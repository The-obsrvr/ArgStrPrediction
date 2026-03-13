import re
import json
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import torch
from torch import Tensor, nn
from axial_positional_embedding import AxialPositionalEmbedding
from torch.utils.data import Dataset
from flair.data import Sentence

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
_END_PUNCT_RE = re.compile(r"[.!?]\s*$")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def english_word_count(text: str) -> int:
    """Count roughly English-like words."""
    return len(_WORD_RE.findall(text))


def word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def split_into_sentences_with_offsets(text: str) -> List[Tuple[int, int]]:
    """
    Returns [(start_char, end_char), ...] for each sentence in `text`.
    Uses a simple punctuation/newline splitter (fast, dependency-free).
    """
    spans = []
    start = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    # Remove empty/whitespace sentences
    spans = [(s, e) for s, e in spans if text[s:e].strip()]
    return spans


def char_to_sentence_index(char_pos: int, sent_spans: List[Tuple[int, int]]) -> int:
    """
    Return the sentence index containing char_pos.
    If not found, return the last sentence index.
    """
    for i, (s, e) in enumerate(sent_spans):
        if s <= char_pos < e:
            return i
    return max(0, len(sent_spans) - 1)


def find_unit_span(conversation_text: str, unit_text: str) -> Optional[Tuple[int, int]]:
    """
    Find a best-effort character span for unit_text in conversation_text.
    Returns (start, end) or None if not found.
    """
    if not unit_text or not unit_text.strip():
        return None

    # Exact match first
    pattern = re.escape(unit_text.strip())
    m = re.search(pattern, conversation_text)
    if m:
        return m.span()

    # Case-insensitive fallback
    m = re.search(pattern, conversation_text, flags=re.IGNORECASE)
    if m:
        return m.span()

    return None


def build_multihot_token_labels_for_conversation(
        sentence: Sentence,
        argument_objects: Dict[str, Any],
        label_to_id: Dict[Any, int],
        ) -> Tensor:
    """
    Build multi-hot (0/1) token labels with shape (num_tokens, 30).

    Rules:
    - Tokens outside any argument unit: O=1.
    - Tokens inside an argument unit:
        - BIO: B for first token of unit span, I for continuation (no O)
        - If unit has outgoing relations:
            - Support=1 and/or Attack=1 (depending on types present)
            - Distance class (integer) for each outgoing relation:
                dist = target_sentence_idx - source_sentence_idx
                clipped to the available integer label keys.
        - If unit has no outgoing relations:
            - root=1

    Notes:
    - If a unit has multiple relations, this becomes multi-hot (multiple distances can be 1).
    - If a token belongs to multiple overlapping units (rare), we OR the labels.
    """
    text = sentence.to_plain_string()
    tokens = sentence.tokens
    n_tokens = len(tokens)
    n_labels = len(label_to_id)

    y = torch.zeros((n_tokens, n_labels), dtype=torch.float32)

    # Default all tokens to O=1
    o_idx = label_to_id["O"]
    y[:, o_idx] = 1.0

    # Sentence spans for distance computation
    sent_spans = split_into_sentences_with_offsets(text)

    # Index argument units by id and compute sentence index per unit
    units = argument_objects.get("argument_units", []) or []
    relations = argument_objects.get("relations", []) or []

    unit_id_to_sent_idx: Dict[int, int] = {}
    unit_id_to_char_span: Dict[int, Tuple[int, int]] = {}

    for u in units:
        uid = int(u.get("id"))
        span = find_unit_span(text, str(u.get("text", "")))
        if span is None:
            continue
        unit_id_to_char_span[uid] = span
        unit_id_to_sent_idx[uid] = char_to_sentence_index(span[0], sent_spans)

    # Build outgoing relation lists per source unit
    outgoing: Dict[int, List[Dict[str, Any]]] = {}
    for r in relations:
        try:
            s_id = int(r.get("source_id"))
            t_id = int(r.get("target_id"))
        except Exception:
            continue
        outgoing.setdefault(s_id, []).append(r)

    # Precompute token spans (character positions)
    token_spans = [(tok.start_position, tok.end_position) for tok in tokens]

    b_idx = label_to_id["B"]
    i_idx = label_to_id["I"]
    sup_idx = label_to_id["Support"]
    att_idx = label_to_id["Attack"]
    root_idx = label_to_id["root"]

    # Helper: integer distance label id (clip to available ints)
    int_distance_labels = sorted([k for k in label_to_id.keys() if isinstance(k, int)])
    min_d, max_d = int_distance_labels[0], int_distance_labels[-1]

    def _dist_to_label_id(d: int) -> int:
        d_clip = max(min_d, min(max_d, d))
        return label_to_id[d_clip]

    # Apply labels unit-by-unit
    for u in units:
        uid = int(u.get("id"))
        if uid not in unit_id_to_char_span or uid not in unit_id_to_sent_idx:
            continue

        u_start, u_end = unit_id_to_char_span[uid]
        src_sent_idx = unit_id_to_sent_idx[uid]

        # Identify token indices overlapping unit span
        covered = []
        for ti, (ts, te) in enumerate(token_spans):
            if te <= u_start or ts >= u_end:
                continue
            covered.append(ti)

        if not covered:
            continue

        # Remove O for covered tokens and set BIO
        # (multi-hot implies we keep other labels too)
        for j, ti in enumerate(covered):
            y[ti, o_idx] = 0.0
            if j == 0:
                y[ti, b_idx] = 1.0
            else:
                y[ti, i_idx] = 1.0

        outs = outgoing.get(uid, [])
        if not outs:
            # Root unit
            for ti in covered:
                y[ti, root_idx] = 1.0
            continue

        # Has relations: type + distance labels
        for rel in outs:
            rel_type = str(rel.get("type", "")).strip().lower()
            tgt_id = int(rel.get("target_id"))

            if tgt_id not in unit_id_to_sent_idx:
                continue
            tgt_sent_idx = unit_id_to_sent_idx[tgt_id]
            dist = tgt_sent_idx - src_sent_idx

            dist_label_id = _dist_to_label_id(dist)

            for ti in covered:
                if rel_type == "support":
                    y[ti, sup_idx] = 1.0
                elif rel_type == "attack":
                    y[ti, att_idx] = 1.0

                y[ti, dist_label_id] = 1.0

    return y


def token_gap_between_spans(span1: Tuple[int, int], span2: Tuple[int, int]) -> int:
    """
    Number of tokens between two non-overlapping spans.
    If they overlap or touch, gap = 0.
    """
    s1, e1 = span1
    s2, e2 = span2

    # Ensure span1 is left, span2 is right (by start index)
    if s2 < s1:
        s1, e1, s2, e2 = s2, e2, s1, e1

    if s2 <= e1 + 1:
        # overlapping or immediately adjacent
        return 0

    return max(0, s2 - e1 - 1)


def merge_single_word_argumentative_units(
        unit_meta: Dict[int, Dict[str, Any]],
        conversation_text: str,
        *,
        max_word_gap: int = 1,
        ) -> Dict[int, Dict[str, Any]]:
    """
    Merge single-English-word argumentative units into a nearby argumentative unit
    in the same sentence, if they are close in token space (gap <= max_word_gap).

    unit_meta (input and output): {unit_id: unit_dict}
    Each unit_dict must contain:
      - "id", "sent_idx", "tok_span", "char_span", "text", "attrs".

    Returns:
      new_unit_meta: updated dict with merged units, where:
        - small single-word units have been absorbed into neighbours;
        - neighbour units have expanded spans and text.
    """
    # Work on a copy to avoid in-place surprises
    new_meta: Dict[int, Dict[str, Any]] = {uid: dict(u) for uid, u in unit_meta.items()}

    # Group units by sentence index
    sent_to_units: Dict[int, List[int]] = {}
    for uid, u in new_meta.items():
        sent_to_units.setdefault(u["sent_idx"], []).append(uid)

    # We’ll mark small units for removal after merging
    to_remove: set[int] = set()

    for sent_idx, unit_ids in sent_to_units.items():
        # sort by token start for deterministic behaviour
        unit_ids_sorted = sorted(unit_ids, key=lambda uid: new_meta[uid]["tok_span"][0])

        # Precompute some stats
        for uid in unit_ids_sorted:
            u = new_meta[uid]
            u_text = (u.get("text") or "").strip()
            u["__english_words__"] = english_word_count(u_text)
            attrs = u.get("attrs") or {}
            u["__is_argumentative__"] = bool(attrs.get("support") or attrs.get("attack") or attrs.get("root"))

        # For each single-word argumentative unit, try to find a close neighbour
        for uid in unit_ids_sorted:
            if uid in to_remove:
                continue

            u = new_meta[uid]
            if not u.get("__is_argumentative__", False):
                continue

            if u.get("__english_words__", 0) != 1:
                continue  # not a single English word

            u_span = u["tok_span"]

            # Find the closest other argumentative unit in this sentence
            best_neighbor_id: Optional[int] = None
            best_gap: Optional[int] = None

            for vid in unit_ids_sorted:
                if vid == uid or vid in to_remove:
                    continue
                v = new_meta[vid]
                if not v.get("__is_argumentative__", False):
                    continue

                gap = token_gap_between_spans(u_span, v["tok_span"])
                if gap <= max_word_gap:
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_neighbor_id = vid

            if best_neighbor_id is None:
                continue  # no suitable neighbour

            # Merge u into its best neighbour
            v = new_meta[best_neighbor_id]

            # Compute merged spans
            u_s, u_e = u["tok_span"]
            v_s, v_e = v["tok_span"]
            new_s = min(u_s, v_s)
            new_e = max(u_e, v_e)

            uc_s, uc_e = u["char_span"]
            vc_s, vc_e = v["char_span"]
            new_cs = min(uc_s, vc_s)
            new_ce = max(uc_e, vc_e)

            merged_text = conversation_text[new_cs:new_ce].strip()

            # Update neighbour unit in-place
            v["tok_span"] = (new_s, new_e)
            v["char_span"] = (new_cs, new_ce)
            v["text"] = merged_text

            # Mark the tiny single-word unit for removal
            to_remove.add(uid)

        # Clean up temporary flags
        for uid in unit_ids_sorted:
            if uid in new_meta:
                new_meta[uid].pop("__english_words__", None)
                new_meta[uid].pop("__is_argumentative__", None)

    # Actually remove all merged single-word units
    for uid in to_remove:
        new_meta.pop(uid, None)

    return new_meta


def load_qt30_flair_multilabel_dataset(csv_path: str, flair_embedding):
    """
    Load QT30 CSV into FlairDataset with:
      - sentence = Sentence(conversation_text)
      - target = FloatTensor (num_tokens, 28) multi-hot

    Returns (dataset, conversation_ids).
    """
    df = pd.read_csv(csv_path)
    _, label_to_id = get_label_maps()

    sentences = []
    targets = []
    conversation_ids = []

    for _, row in df.iterrows():
        conv_id = int(row["conversation_id"])
        conv_text = str(row["conversation_text"])

        arg_obj = json.loads(row["argument_objects"])
        sent = Sentence(conv_text)

        y = build_multihot_token_labels_for_conversation(
            sentence=sent,
            argument_objects=arg_obj,
            label_to_id=label_to_id,
            )

        sentences.append(sent)
        targets.append(y)
        conversation_ids.append(conv_id)

    dataset = FlairDataset(sentences, targets, flair_embedding)

    return dataset, conversation_ids


def print_label_distribution(dataset, label_to_id, id_to_label):
    """
    dataset: FlairDataset (or iterable returning (tokens, labels, mask))
    labels: multi-hot tensor of shape (T, num_labels)
    """
    num_labels = len(label_to_id)
    label_counts = torch.zeros(num_labels)
    total_tokens = 0

    for _, labels in dataset:
        # labels: (T, num_labels)
        active = torch.ones(labels.size(0), dtype=torch.bool)
        label_counts += labels[active].sum(dim=0)
        total_tokens += active.sum().item()

    print("\n===== LABEL DISTRIBUTION =====")
    print(f"Total labeled tokens: {total_tokens}\n")

    for idx in range(num_labels):
        label = id_to_label[idx]
        count = int(label_counts[idx].item())
        ratio = count / max(total_tokens, 1)
        print(f"{str(label):>8} | count = {count:8d} | ratio = {ratio:.6f}")

    print("================================\n")

    return label_counts


def decode_units_from_bio(
        sentence: Sentence,
        token_multihot: torch.Tensor,
        label_to_id: Dict[Any, int],
        min_unit_tokens: int = 3,
        ) -> List[Dict[str, Any]]:
    """
    Decode predicted argument units from BIO bits in token_multihot.

    token_multihot: (num_tokens, 30) with 0/1 (or probabilities thresholded).
    Returns our Argument Structure-style [{"reason": ..., "id": ..., "text": ...}, ...]
    """
    b_idx = label_to_id["B"]
    i_idx = label_to_id["I"]

    sup_idx = label_to_id.get("Support", 0)
    att_idx = label_to_id.get("Attack", 0)

    tokens = sentence.tokens
    text = sentence.to_plain_string()
    n = len(tokens)

    units = []

    def _flush(start_i: int, end_i: int, uid: int, min_span_len: int = 8):
        """
        Find valid units from predicted token indices
        :param start_i:
        :param end_i:
        :param uid:
        :return:
        """
        if end_i < start_i:
            # INVALID: end should be greater than its start!
            return uid
        if (end_i - start_i + 1) < min_unit_tokens:
            # INVALID: span is potentially too small
            return uid

        start_char = tokens[start_i].start_position
        end_char = tokens[end_i].end_position
        span_text = text[start_char:end_char].strip()
        if not span_text and len(span_text) > min_span_len:
            # INVALID: span_text must be certain char length
            return uid

        # valid units are added here!
        units.append({"id": uid, "text": span_text})
        return uid + 1

    uid = 0
    i = 0

    while i < n:
        # check ARG (B/I)
        is_arg = (
                token_multihot[i, b_idx].item() >= 0.5
                or token_multihot[i, i_idx].item() >= 0.5
        )
        if not is_arg:
            i += 1
            continue

        # find unit spans by ARG value
        span_start = i
        j = i
        while j + 1 < n:
            is_arg_next = (
                    token_multihot[j, b_idx].item() >= 0.5
                    or token_multihot[j, i_idx].item() >= 0.5
            )
            if not is_arg_next:
                break
            j = j + 1
        span_end = j

        # review span by Relation value
        rel_classes = []
        for k in range(span_start, span_end+1):
            if sup_idx is None or att_idx is None:
                # checks if support or attack exists in the label_map (always true)
                rel_classes.append("neu")
                continue

            sup_on = token_multihot[k, sup_idx].item() >= 0.5
            att_on = token_multihot[k, att_idx].item() >= 0.5

            if sup_on and not att_on:
                rel_classes.append("sup")
            elif att_on and not sup_on:
                rel_classes.append("att")
            else:
                rel_classes.append("neu")

        # split span by relation if applicable
        seg_start = span_start
        curr_rel = rel_classes[0] if rel_classes else "neu"

        for offset in range(1, len(rel_classes)):
            k = span_start + offset
            new_rel = rel_classes[offset]
            # split only on clear relations
            if (curr_rel in {"sup", "att"}
                    and new_rel in {"sup", "att"}
                    and new_rel != curr_rel):
                uid = _flush(seg_start, k - 1, uid)
                seg_start = k
                curr_rel = new_rel
            else:
                if new_rel in {"sup", "att"}:
                    curr_rel = new_rel

        uid = _flush(seg_start, span_end, uid)

        i = span_end + 1

    if len(units) == 0:
        print("No valid units found!")

    return units


def decode_relations_from_token_labels(
        conversation_text: str,
        predicted_units: List[Dict[str, Any]],
        token_multihot: torch.Tensor,
        label_to_id: Dict[Any, int],
        ) -> List[Dict[str, Any]]:
    """
    Build relations by:
      - determining each unit's sentence index (via span matching),
      - checking whether unit tokens have Support/Attack set,
      - reading which distance class(es) are set,
      - mapping (source_sent + dist) -> choose a target unit in that sentence.

    Returns QT30 relations: [{"source_id": i, "target_id": j, "type": "support"/"attack"}, ...]
    """
    # Sentence segmentation for mapping distances
    sent_spans = split_into_sentences_with_offsets(conversation_text)

    # Map predicted unit id -> (char_span, sentence_idx)
    unit_meta = {}
    for u in predicted_units:
        uid = int(u["id"])
        span = find_unit_span(conversation_text, u["text"])
        if span is None:
            print("Unit text not found in conversation text")
            continue
        # map its first occurrence
        s_idx = char_to_sentence_index(span[0], sent_spans)
        unit_meta[uid] = {"span": span, "sent_idx": s_idx}

    # Build sentence_idx -> list of unit_ids
    sent_to_units: Dict[int, List[int]] = {}
    for uid, m in unit_meta.items():
        sent_to_units.setdefault(m["sent_idx"], []).append(uid)

    sup_idx = label_to_id["Support"]
    att_idx = label_to_id["Attack"]

    # distance integer labels present in map
    distance_vals = sorted([k for k in label_to_id.keys() if isinstance(k, int)])
    distance_label_indices = [(d, label_to_id[d]) for d in distance_vals]

    relations = []

    tokens = Sentence(conversation_text).tokens  # quick tokenisation to use start_pos/end_pos
    token_spans = [(t.start_position, t.end_position) for t in tokens]

    def _tokens_overlapping_span(span: Tuple[int, int]) -> List[int]:
        a, b = span
        out = []
        for i, (ts, te) in enumerate(token_spans):
            if te <= a or ts >= b:
                continue
            out.append(i)
        return out

    # For each unit, inspect its token label bits
    for u in predicted_units:
        src_id = int(u["id"])
        if src_id not in unit_meta:
            continue

        src_sent = unit_meta[src_id]["sent_idx"]
        span = unit_meta[src_id]["span"]
        tok_ids = _tokens_overlapping_span(span)
        if not tok_ids:
            continue

        # Aggregate: if ANY token in unit has sup/att bit set -> treat as that relation present
        has_sup = any(token_multihot[t, sup_idx].item() >= 0.5 for t in tok_ids)
        has_att = any(token_multihot[t, att_idx].item() >= 0.5 for t in tok_ids)

        # Collect distances marked in any token (multi-hot)
        marked_dists = set()
        for d, idx in distance_label_indices:
            if any(token_multihot[t, idx].item() >= 0.5 for t in tok_ids):
                marked_dists.add(d)

        # If no distances, skip relation construction
        if not marked_dists:
            print("No distance predicted by model.")
            continue

        for d in sorted(marked_dists):
            tgt_sent = src_sent + d
            if tgt_sent not in sent_to_units:
                continue
            # Choose a target unit in that sentence; simplest: first one in that sentence
            tgt_id = sent_to_units[tgt_sent][0]

            if has_sup:
                relations.append({"source_id": src_id, "target_id": tgt_id, "type": "support"})
            if has_att:
                relations.append({"source_id": src_id, "target_id": tgt_id, "type": "attack"})

    # Optionally deduplicate
    uniq = []
    seen = set()
    for r in relations:
        key = (r["source_id"], r["target_id"], r["type"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    if len(uniq) == 0:
        print("No relations found!")

    return uniq


def get_label_maps():
    """
    Returns:
    id_to_label: Dict[int, Union[str, int]] - Mapping for decoding model output
    label_to_id: Dict[Union[str, int], int] - Mapping for encoding training data
    """
    id_to_label = {
        0: 'O', 1: 'B', 2: 'I', 3: 'Support', 4: 'Attack',
        5: -11, 6: -10, 7: -9, 8: -8, 9: -7, 10: -6, 11: -5,
        12: -4, 13: -3, 14: -2, 15: -1, 16: 0, 17: 1, 18: 2,
        19: 3, 20: 4, 21: 5, 22: 6, 23: 7, 24: 8, 25: 9,
        26: 10, 27: 11, 28: 'root'
        }
    # Create the reverse map to encode your training data
    label_to_id = {v: k for k, v in id_to_label.items()}
    return id_to_label, label_to_id


# --- 2. Data Handling & Embeddings ---
class FlairDataset(Dataset):
    def __init__(self, sentences, labels, embedding_stack):
        """
        sentences: List of flair.data.Sentence objects
        labels: List of LongTensors (indices) corresponding to tokens
        """
        self.sentences = sentences
        self.labels = labels
        self.embedding_stack = embedding_stack

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # Embed on demand (or pre-calculate for speed)
        self.embedding_stack.embed(sentence)

        # Extract embeddings as tensor (Seq_Len, Embedding_Dim)
        # Note: This gathers embeddings from tokens into a single tensor
        emb_tensor = torch.stack([token.embedding for token in sentence])

        return emb_tensor, self.labels[idx]


class UnifiedAM_Conv(nn.Module):
    """
    Flair-based single-step model:
      - Project Flair embeddings
      - Axial positional embedding
      - Multi-head attention
      - BiLSTM
      - Token-level multi-label classifier
    """

    def __init__(self, input_dim, num_labels, dropout_p=0.5):
        super(UnifiedAM_Conv, self).__init__()

        # Project Flair embeddings to a manageable size
        self.projection = nn.Linear(input_dim, 512)

        # Axial positional embedding (expects (B, T, dim))
        # we set (40,40) for up to maximum sequence length [~1600]
        # axial dims should add up to dim: 256+256=512
        self.pos_emb = AxialPositionalEmbedding(
            dim=512,
            axial_shape=(40, 40),
            axial_dims=(256, 256),
            )

        # 3. Multi-head self-attention
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        # 4. BiLSTM over [attn_output ; projected_input]
        #    LSTM input: 512 (attn) + 512 (projection) = 1024
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            bidirectional=True,
            num_layers=2,
            dropout=0.65,
            )

        # 5. Classifier: BiLSTM outputs 512*2 = 1024
        self.classifier = nn.Linear(in_features=1024, out_features=num_labels)

        self.dropout = nn.Dropout(p=dropout_p)

    def init_hidden(self, batch_size, device):
        """
        not used since we internally define the hidden state in our LSTM object above.
        """
        hidden = (
            torch.zeros(2 * 2, batch_size, 512).cuda(),
            torch.zeros(2 * 2, batch_size, 512).cuda()
            )
        hidden_final = (
            torch.zeros(2 * 2, batch_size, 512).cuda(),
            torch.zeros(2 * 2, batch_size, 512).cuda()
            )
        return hidden, hidden_final

    def forward(self, src_embedding, lengths: Optional = None, hidden: Optional = None, hidden_state: Optional = None):
        """
        src_embedding: (B, T, input_dim)
        """
        # Projection (B, T, input_dim) -> (B, T, 512)
        x = self.projection(src_embedding)

        # Positional Embedding: Axial expects (B, T, 512)
        pos_out = self.pos_emb(x)  # (B, T, 512)

        # Attention: MultiheadAttention expects (T, B, E)
        pos_out = pos_out.permute(1, 0, 2)  # (T, B, 512)
        attn_output, attn_weights = self.multi_head_attn(
            pos_out, pos_out, pos_out
            )  # attn_output: (T, B, 512)

        # Residual concat with projected input
        x_permuted = x.permute(1, 0, 2)  # (T, B, 512)
        combined = torch.cat((attn_output, x_permuted), dim=2)  # (T, B, 1024)

        # BiLSTM directly on padded input
        output, hidden = self.lstm(combined)  # output: (T, B, 1024)

        # Classification back to (B, T, label)
        output = output.permute(1, 0, 2).contiguous()  # (B, T, 1024)
        logits = self.classifier(self.dropout(output))  # (B, T, label)
        return logits, hidden, attn_weights
