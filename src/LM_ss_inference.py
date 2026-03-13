import argparse
import json
import os
from typing import Dict, Any, List, Tuple, Optional

import torch
import pandas as pd
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, WordEmbeddings, BytePairEmbeddings, FlairEmbeddings

from LM_ss_utilities import (
    get_label_maps,
    UnifiedAM_Conv,
    decode_units_from_bio,
    decode_relations_from_token_labels,
    )


def repair_bio_bits(token_multihot: torch.Tensor, label_to_id: Dict[Any, int]) -> torch.Tensor:
    """
    Repair BIO bits with *unit merging* semantics:

    Rules:
      1) If both B and I are active on the same token → keep B, drop I.
      2) If I occurs when we are not inside a unit → convert to B.
      3) If B occurs immediately after another B or I (no O gap) → convert to I (merge).
      4) O resets unit state.

    token_multihot: (T, num_labels) tensor with thresholded {0,1} values
    """
    y = token_multihot.clone()

    b = label_to_id["B"]
    i = label_to_id["I"]

    T = y.size(0)

    # Rule: Prefer B over I if both active
    both = (y[:, b] >= 0.5) & (y[:, i] >= 0.5)
    y[both, i] = 0.0

    inside = False

    for t in range(T):
        is_b = y[t, b].item() >= 0.5
        is_i = y[t, i].item() >= 0.5

        if is_b:
            if inside:
                # Rule: B immediately after B/I → merge into same unit
                y[t, b] = 0.0
                y[t, i] = 1.0
            else:
                # Legitimate unit start
                inside = True
            continue

        if is_i:
            if not inside:
                # Rule: stray I → convert to B
                y[t, b] = 1.0
                y[t, i] = 0.0
                inside = True
            # else: valid continuation
            continue

        # Rule 4: outside token
        inside = False

    return y


def decode_units_and_relations(
        conversation_id: int,
        conversation_text: str,
        token_multihot_raw: torch.Tensor,
        label_to_id: Dict[Any, int],
        ) -> Dict[str, Any]:
    """
    Robust decoding:
      1) Repair BIO bits
      2) Decode unit token spans
      3) Majority-vote unit attributes (Support/Attack/Distance)
      4) Extract unit texts
      5) Build relations based on unit-level (type + distance)

    Returns unified JSON structure.
    """
    sent = Sentence(conversation_text)

    # 1) unit BIO tagging
    token_multihot = repair_bio_bits(token_multihot_raw, label_to_id)

    # 2) Unit spans
    predicted_units = decode_units_from_bio(
        sentence=sent,
        token_multihot=token_multihot,
        label_to_id=label_to_id,
        min_unit_tokens=1
        )

    # Sentence segmentation for distance mapping
    relations = decode_relations_from_token_labels(
        conversation_text=conversation_text,
        predicted_units=predicted_units,
        token_multihot=token_multihot,
        label_to_id=label_to_id,
        )

    return {
        "argument_units": predicted_units,
        "relations": relations
        }


def _threshold_logits_to_multihot(
        logits: torch.Tensor,
        threshold: float = 0.5
        ) -> torch.Tensor:
    """
    logits: (T, num_labels) or (B, T, num_labels)

    Returns:
        FloatTensor with same shape, values 0.0 or 1.0 after sigmoid + threshold.
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).float()


def infer_and_save_json(
        csv_path: str,
        model_weights_path: str,
        output_json_path: str,
        threshold: float = 0.3,
        device: Optional[str] = None,
        ) -> None:
    """
    Run token-level multi-label inference over a QT30-style CSV and save unified
    JSON predictions.

    CSV is expected to have at least:
      - 'conversation_id'
      - 'conversation_text'

    Model is expected to be a UnifiedAM_Conv trained with:
      - input_dim = 4096 (news-forward-fast + news-backward-fast)
      - out_features = 29 (all labels from get_label_maps)
    """
    # --- Device ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # --- Load label maps ---
    id_to_label, label_to_id = get_label_maps()
    num_labels = len(id_to_label)

    # --- Build Flair embeddings stack (must match training) ---
    flair_embedding = StackedEmbeddings([
        # WordEmbeddings('en'),
        # # Byte pair embeddings for English to handle OOV
        # BytePairEmbeddings('en'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
        ]
        )
    embedding_dim = flair_embedding.embedding_length  # should be 4096

    # --- Init model & load weights ---
    model = UnifiedAM_Conv(input_dim=embedding_dim, num_labels=num_labels)
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Load data ---
    df = pd.read_csv(csv_path)

    predictions = {}

    for _, row in df.iterrows():
        conv_id = int(row["conversation_id"])
        conv_text = str(row["conversation_text"])

        # Build Flair sentence and embed
        sent = Sentence(conv_text)
        flair_embedding.embed(sent)
        emb_tensor = torch.stack([t.embedding for t in sent])  # (T, D)

        # Prepare model inputs
        inputs = emb_tensor.unsqueeze(0).to(device)  # (1, T, D)
        # lengths = torch.tensor([emb_tensor.size(0)], dtype=torch.long).to(device)
        batch_size = 1

        # hidden, hidden_final = model.init_hidden(batch_size, device)

        with torch.no_grad():
            logits, _, _ = model(inputs)

        # Handle both (B,T,L) and legacy (B*T,L) just in case
        if logits.dim() == 3:
            logits_t = logits[0]  # (T, num_labels)
        else:
            # legacy: (B*T, num_labels) -> reshape, then trim to T
            logits_t = logits.view(batch_size, -1, num_labels)[0][:emb_tensor.size(0)]

        token_multihot = _threshold_logits_to_multihot(logits_t, threshold=threshold)  # (T, num_labels)

        # Build unified JSON for this conversation
        predictions[conv_id] = decode_units_and_relations(
            conversation_id=conv_id,
            conversation_text=conv_text,
            token_multihot_raw=token_multihot,
            label_to_id=label_to_id,
            )

    # --- Save all predictions ---
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"[infer_and_save_json] Saved predictions for {len(predictions)} conversations to {output_json_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    data_name = ""

    if "QT" in args.data_path:
        data_name = "qt"
    elif "reddit" in args.data_path:
        data_name = "reddit"
    elif "RIP" in args.data_path:
        data_name = "rip"
    else:
        ValueError(f"Unrecognized data path: {args.data_path}")

    filename = f"ss_LM_{data_name}_{args.seed}.json"

    output_json_path = os.path.join(args.run_dir, filename)

    model_weights_path = os.path.join(args.run_dir, "best_bilstm_er_model.pth")

    infer_and_save_json(
        csv_path=args.data_path,
        model_weights_path=model_weights_path,
        output_json_path=output_json_path,
        threshold=args.threshold,
        )
