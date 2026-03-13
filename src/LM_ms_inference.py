import json
from pathlib import Path

import torch
import pandas as pd
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer

from system_utilities import parse_args, load_config
from LM_ms_utilities import get_label_maps_ms_AUE, get_label_maps_ms_RTC, ArgumentClassifier

import re

# Very cheap heuristic: at least one alphabetic “word-like” token of length >= 2
_UNIT_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def is_valid_unit_text(text: str, min_chars: int = 8) -> bool:
    """
    Returns True if the unit text contains at least one word-like English token.
    This filters out units that are pure punctuation, numbers, or garbage.
    """
    if not text:
        return False

    stripped = text.strip()
    if len(stripped) < min_chars:
        return False

    # Extract alphabetic "words"
    words = re.findall(r"[A-Za-z]+", stripped)
    if not words:
        return False

    # If there's only one short word (<= 3 chars), treat it as non-argument
    if len(words) == 1 and len(words[0]) <= min_chars:
        return False

    # if all conditions satisfy then it is a valid unit!
    return True


def decode_aue_to_units(
        text: str,
        encoding,
        pred_label_ids: List[int],
        id_to_label: Dict[int, str]
        ) -> List[Dict[str, Any]]:
    """
    Decode token-level O/B/I predictions into argument units with char spans.
    """
    offsets = encoding["offset_mapping"]
    units = []
    current_start = None
    current_end = None

    for label_id, (s, e) in zip(pred_label_ids, offsets):
        if s is None or e is None:
            continue

        lab = id_to_label.get(label_id, "O")

        if lab == "B":
            # close previous
            if current_start is not None:
                u_start, u_end = current_start, current_end
                u_text = text[u_start:u_end]
                units.append({"start": u_start, "end": u_end, "text": u_text})
            current_start, current_end = s, e

        elif lab == "I" and current_start is not None:
            # extend current span
            current_end = e

        else:  # O
            if current_start is not None:
                u_start, u_end = current_start, current_end
                u_text = text[u_start:u_end]
                units.append({"start": u_start, "end": u_end, "text": u_text})
                current_start, current_end = None, None

        # flush last
    if current_start is not None:
        u_start, u_end = current_start, current_end
        u_text = text[u_start:u_end]
        units.append({"start": u_start, "end": u_end, "text": u_text})

    return units


def merge_adjacent_units(
        units: List[Dict[str, Any]],
        text: str,
        max_gap_chars: int = 2
        ) -> List[Dict[str, Any]]:
    """
    Merge adjacent units into one unit.
    :param units:
    :param text:
    :param max_gap_chars:
    :return:
    """

    if not units:
        return []

        # Ensure chronological order
    units_sorted = sorted(units, key=lambda u: u["start"])
    merged: List[Dict[str, Any]] = []
    current = units_sorted[0]

    for nxt in units_sorted[1:]:
        gap_start = current["end"]
        gap_end = nxt["start"]

        # Overlapping or contiguous spans: just merge them
        if gap_end <= gap_start:
            new_start = min(current["start"], nxt["start"])
            new_end = max(current["end"], nxt["end"])
            current = {
                "start": new_start,
                "end": new_end,
                "text": text[new_start:new_end],
                }
            continue

        gap_text = text[gap_start:gap_end]
        stripped = gap_text.strip()

        # If the gap is tiny and contains no alphanumeric chars, merge
        if len(stripped) <= max_gap_chars and all(not c.isalnum() for c in stripped):
            new_start = current["start"]
            new_end = nxt["end"]
            current = {
                "start": new_start,
                "end": new_end,
                "text": text[new_start:new_end],
                }
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def run_aue_for_conversation(
        model: ArgumentClassifier,
        tokenizer,
        text: str,
        max_length: int,
        device: torch.device
        ) -> List[Dict[str, Any]]:
    """
    Apply AUE token classifier to a single conversation and return predicted units.
    """
    id_to_label, _ = get_label_maps_ms_AUE()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt"
        )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # ArgumentClassifier will output [B, T, C] for AUE
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    preds = logits.argmax(dim=-1)[0].cpu().tolist()  # (T,)
    encoding_for_decode = {
        "offset_mapping": enc["offset_mapping"][0].tolist()
        }

    # raw units with id and text
    raw_units = decode_aue_to_units(text, encoding_for_decode, preds, id_to_label)

    # merge units
    merged_units = merge_adjacent_units(raw_units, text, max_gap_chars=2)

    # fix malformed units (no proper English words)
    filtered_units = [
        u for u in merged_units
        if is_valid_unit_text(u["text"])
        ]

    # sort chronologically by start position
    filtered_units.sort(key=lambda u: u["start"])

    # reassign IDs
    for new_id, u in enumerate(filtered_units):
        u["id"] = new_id

    return filtered_units


def build_rtc_inputs_from_units(
        full_text: str,
        units: List[Dict[str, Any]],
        window_size: int = 150,
        use_context: bool = False,
        max_ctx_chars: int = 100,
        ) -> Tuple[
            List[str],  # seq_a_list
            List[str],  # seq_b_list
            List[Tuple[int, int]],  # pairs (src_id, tgt_id)
            List[Tuple[int, int]],  # tgt_char_spans
            List[Tuple[int, int]],  # src_char_spans
        ]:
    """
    Construct RTC input texts in the same format as RTCHFDataset:

      - Directionality: source must appear AFTER target in the conversation text.
      - seq_a: context (optional, around target) + target unit text
      - seq_b: source unit text

    Returns:
      seq_a_list, seq_b_list, pairs, tgt_char_spans, src_char_spans
    """

    seq_a_list: List[str] = []
    seq_b_list: List[str] = []
    pairs: List[Tuple[int, int]] = []
    tgt_char_spans: List[Tuple[int, int]] = []
    src_char_spans: List[Tuple[int, int]] = []

    if not units or len(units) < 2:
        return seq_a_list, seq_b_list, pairs, tgt_char_spans, src_char_spans

    # units are already sorted by "start" in run_aue_for_conversation
    for src_idx, src in enumerate(units):
        for tgt_idx, tgt in enumerate(units):
            if src_idx == tgt_idx:
                continue

            # enforce: source appears AFTER target in the text
            if src["start"] <= tgt["start"]:
                continue

            t_start, t_end = tgt["start"], tgt["end"]

            # context window around target (same idea as RTCHFDataset)
            c_start = max(0, t_start - window_size)
            c_end = min(len(full_text), t_end + window_size)
            context = full_text[c_start:c_end]

            # clip context length as in RTCHFDataset (max_ctx_chars=80 there)
            if context and len(context) > max_ctx_chars:
                context = context[:max_ctx_chars]

            tgt_text = tgt["text"]
            src_text = src["text"]

            if use_context:
                seq_a = f"{context}\n\n{tgt_text}"
            else:
                seq_a = f"{tgt_text}"

            seq_b = f"{src_text}"

            # char spans of unit texts inside seq_a / seq_b
            tgt_s = seq_a.rfind(tgt_text)
            if tgt_s < 0:
                tgt_s = 0
            tgt_e = tgt_s + len(tgt_text)

            src_s = seq_b.rfind(src_text)
            if src_s < 0:
                src_s = 0
            src_e = src_s + len(src_text)

            seq_a_list.append(seq_a)
            seq_b_list.append(seq_b)
            pairs.append((src["id"], tgt["id"]))
            tgt_char_spans.append((tgt_s, tgt_e))
            src_char_spans.append((src_s, src_e))

    return seq_a_list, seq_b_list, pairs, tgt_char_spans, src_char_spans


def run_rtc_for_pairs(
        model: ArgumentClassifier,
        tokenizer,
        seq_a_list: List[str],
        seq_b_list: List[str],
        pairs: List[Tuple[int, int]],
        tgt_char_spans: List[Tuple[int, int]],
        src_char_spans: List[Tuple[int, int]],
        max_length: int,
        device: torch.device,
        batch_size: int = 8,
        ) -> List[Dict[str, Any]]:
    """
    Run the RTC classifier on unit pairs and return predicted relations.
    """
    id_to_label, _ = get_label_maps_ms_RTC()

    if not pairs:
        print("no pairs detected!")
        return []

    all_preds: List[int] = []

    for i in range(0, len(pairs), batch_size):

        batch_seqa = seq_a_list[i: i + batch_size]
        batch_seqb = seq_b_list[i: i + batch_size]
        batch_tgt_chars = tgt_char_spans[i: i + batch_size]
        batch_src_chars = src_char_spans[i: i + batch_size]

        enc = tokenizer(
            batch_seqa,
            batch_seqb,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
            )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"]

        # build token-level masks as in RTCHF dataset
        tgt_masks = []
        src_masks = []

        for b_idx in range(input_ids.size(0)):
            seq_ids = enc.sequence_ids(b_idx)
            tgt_s, tgt_e = batch_tgt_chars[b_idx]
            src_s, src_e = batch_src_chars[b_idx]

            tgt_mask = [0] * input_ids.size(1)
            src_mask = [0] * input_ids.size(1)

            for j, (seq_id, (s, e)) in enumerate(zip(seq_ids, offsets[b_idx])):
                if seq_id is None:
                    continue
                if s is None or e is None:
                    continue
                if e <= s:
                    continue

                if seq_id == 0:
                    # token belongs to seq_a (target side + context)
                    # NOTE: logic mirrors RTCHFDataset (even if it's generous).
                    if not (e <= tgt_s and s >= tgt_e):
                        tgt_mask[j] = 1
                elif seq_id == 1:
                    # token belongs to seq_b (source side)
                    if not (e <= src_s and s >= src_e):
                        src_mask[j] = 1

            tgt_masks.append(tgt_mask)
            src_masks.append(src_mask)

        # Drop offsets before sending to model
        enc.pop("offset_mapping", None)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        tgt_masks_tensor = torch.tensor(tgt_masks, dtype=torch.long).to(device)
        src_masks_tensor = torch.tensor(src_masks, dtype=torch.long).to(device)

        # Long model global attention, as in training
        global_attention_mask = None
        if model.is_long_model:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
            global_attention_mask = global_attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           global_attention_mask=global_attention_mask,
                           tgt_mask=tgt_masks_tensor,
                           src_mask=src_masks_tensor)

        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

    relations = []
    for (src_id, tgt_id), label_id in zip(pairs, all_preds):
        lbl_str = id_to_label[label_id]
        if lbl_str == "non":
            continue
        relations.append({
            "source_id": src_id,
            "target_id": tgt_id,
            "type": lbl_str
            }
            )
    return relations


def main():
    args = parse_args()
    seed = args.seed
    config = load_config(args)

    # --- Decide which run directory to load checkpoints from ---
    # We use inference.model_name_or_path to point to the training run dir.
    inf_cfg = config.get("inference", {})
    run_dir_str = inf_cfg.get("model_name_or_path", "")
    if not run_dir_str:
        raise ValueError(
            "Please set 'inference.model_name_or_path' in your YAML to the run directory "
            "containing 'best_model_AUE.pt' and 'best_model_RTC.pt'"
        )

    run_dir = Path(run_dir_str)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load AUE model & tokenizer ---
    aue_cfg = config["AUE"]
    aue_backbone = aue_cfg.get("backbone", aue_cfg["model_name_or_path"])

    aue_tokenizer = AutoTokenizer.from_pretrained(aue_backbone)
    aue_model = ArgumentClassifier(
        aue_backbone,
        num_labels=aue_cfg["num_labels"],
        is_long_model=aue_cfg.get("is_long_model", False),
        task_type="AUE",
    )
    aue_best_path = run_dir / "best_model_AUE.pt"
    aue_model.load_state_dict(torch.load(aue_best_path, map_location="cpu"))
    aue_model.to(device)
    aue_model.eval()
    print("AUE model loaded")

    # --- Load RTC model & tokenizer ---
    rtc_cfg = config["RTC"]
    rtc_backbone = rtc_cfg.get("backbone", rtc_cfg["model_name_or_path"])

    rtc_tokenizer = AutoTokenizer.from_pretrained(rtc_backbone)
    rtc_model = ArgumentClassifier(
        rtc_backbone,
        num_labels=rtc_cfg["num_labels"],
        is_long_model=rtc_cfg.get("is_long_model", False),
        task_type="RTC",
    )
    rtc_best_path = run_dir / "best_model_RTC.pt"
    rtc_model.load_state_dict(torch.load(rtc_best_path, map_location="cpu"))
    rtc_model.to(device)
    rtc_model.eval()
    print(f"RTC Model loaded")

    # --- Load inference data ---
    inf_path = config["experiment"]["data_path"]
    # set corpus id.
    if "QT" in inf_path:
        data_name = "qt"
    elif "reddit" in inf_path:
        data_name = "reddit"
    else:
        data_name = "rip"

    df = pd.read_csv(inf_path)

    predictions = {}

    for _, row in df.iterrows():
        conv_id = int(row.get("conversation_id", _))
        text = str(row["conversation_text"])

        # AUE model to produce argument units (post-filtering included)
        units = run_aue_for_conversation(
            model=aue_model,
            tokenizer=aue_tokenizer,
            text=text,
            max_length=aue_cfg["max_length"],
            device=device
            )

        # Build RTC pair inputs from predicted units
        seq_a_list, seq_b_list, pairs, tgt_char_spans, src_char_spans = build_rtc_inputs_from_units(
            full_text=text,
            units=units,
            window_size=200,
            use_context=True,
            max_ctx_chars=100
            )

        # RTC model to predicted relations (post-filtering included)
        relations = run_rtc_for_pairs(
            model=rtc_model,
            tokenizer=rtc_tokenizer,
            seq_a_list=seq_a_list,
            seq_b_list=seq_b_list,
            pairs=pairs,
            tgt_char_spans=tgt_char_spans,
            src_char_spans=src_char_spans,
            max_length=rtc_cfg["max_length"],
            device=device
            )

        # Unified output for this conversation
        predictions[conv_id] = {
            "argument_units": units,
            "relations": relations
            }

    out_path = run_dir / f"FT_ms_{data_name}_{seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved multi-model predictions to {out_path}")


if __name__ == "__main__":
    main()
