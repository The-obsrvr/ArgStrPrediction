import inspect
import json
import re
from typing import Dict, Tuple, List, Optional
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.utils import compute_class_weight
from transformers import AutoTokenizer, XLMRobertaTokenizerFast, AutoModel


def get_label_maps_ms_AUE():
    """
    Returns:
    id_to_label: Dict[int, Union[str, int]] - Mapping for decoding model output
    label_to_id: Dict[Union[str, int], int] - Mapping for encoding training data
    """
    id_to_label = {
        0: 'O', 1: 'B', 2: 'I'
        }
    # Create the reverse map to encode your training data
    label_to_id = {v: k for k, v in id_to_label.items()}
    return id_to_label, label_to_id


def get_label_maps_ms_RTC():
    """
    Returns:
    id_to_label: Dict[int, Union[str, int]] - Mapping for decoding model output
    label_to_id: Dict[Union[str, int], int] - Mapping for encoding training data
    """
    id_to_label = {
        0: 'support', 1: 'attack', 2: 'non'
        }
    # Create the reverse map to encode your training data
    label_to_id = {v: k for k, v in id_to_label.items()}
    return id_to_label, label_to_id


def load_dataframe(config) -> pd.DataFrame:
    """
    Load the common QT30 CSV once.

    Expects config["experiment"]["data_path"] to point to QT30_training(1).csv
    with columns:
      - conversation_id
      - conversation_text
      - argument_objects (JSON string with 'argument_units' and 'relations')
    """
    path = config["experiment"].get("data_path")
    if not path:
        raise ValueError("config['experiment']['data_path'] must be set to QT30_training(1).csv")

    df = pd.read_csv(path)
    if "conversation_text" not in df.columns or "argument_objects" not in df.columns:
        raise ValueError("QT30 file must have 'conversation_text' and 'argument_objects' columns.")
    return df


def _find_unit_span_in_text(text: str, unit_text: str) -> Tuple[int, int]:
    """
    Best-effort character span for a unit's 'text' inside conversation_text.
    Returns (start, end). Raises ValueError if not found.
    """
    if not unit_text or not unit_text.strip():
        raise ValueError("Empty unit text")

    pattern = re.escape(unit_text.strip())
    m = re.search(pattern, text)
    if not m:
        # fallback: case-insensitive
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Unit text not found in conversation: {unit_text[:50]!r}")

    return m.start(), m.end()


class AUEHFDataset(Dataset):
    """
    HuggingFace-style dataset for token classification (O/B/I).
    Expects `data` as a DataFrame with:
      - 'text' column
      - 'units' column: JSON string or list of dicts with 'start', 'end'
    """

    def __init__(self, data: pd.DataFrame, tokenizer_name: str, max_length: int):
        self.data = data.reset_index(drop=True)

        tokenizer_name_lc = tokenizer_name.lower()

        # Explicitly route XLM-R models to the *slow* tokenizer to avoid the fast-conversion bug
        if "xlm-roberta" in tokenizer_name_lc:
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

        self.max_length = max_length

        _, self.label_to_id = get_label_maps_ms_AUE()

        self.encodings = []
        self.labels = []

        for _, row in self.data.iterrows():
            text = str(row["conversation_text"])
            arg_obj = json.loads(row["argument_objects"])
            units = arg_obj.get("argument_units", []) or []

            # 1) compute unit spans in char space
            spans = []
            for u in units:
                u_text = str(u.get("text", ""))
                try:
                    start, end = _find_unit_span_in_text(text, u_text)
                    spans.append((start, end))
                except ValueError:
                    # skip units that fail to align
                    continue

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                padding=False,  # use dynamic padding
                )

            offsets = encoding["offset_mapping"]

            # default: O
            tags = [self.label_to_id['O']] * len(offsets)

            # mark B/I according to char spans
            for (u_start, u_end) in spans:
                first = True
                for i, (s, e) in enumerate(offsets):
                    if s is None or e is None:
                        continue
                    if e <= u_start or s >= u_end:
                        continue
                    if first:
                        tags[i] = self.label_to_id['B']
                        first = False
                    else:
                        tags[i] = self.label_to_id['I']

            # build labels at token level (ignore subword splitting with -100)
            word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else None
            if word_ids is not None:
                label_ids = []
                for idx, (wi, (s, e)) in enumerate(zip(word_ids, offsets)):
                    if wi is None:
                        # special tokens / padding
                        label_ids.append(-100)
                    else:
                        # Use the tag for THIS TOKEN index, not the word index
                        label_ids.append(tags[idx])
            else:
                # Fallback: no word_ids; just use tags as-is
                label_ids = tags

            encoding = {k: torch.tensor(v) for k, v in encoding.items() if k != "offset_mapping"}
            self.encodings.append(encoding)
            self.labels.append(torch.tensor(label_ids, dtype=torch.long))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] if v.ndim > 1 else v for k, v in self.encodings[idx].items()}
        item["labels"] = self.labels[idx]
        return item


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _sample_nonarg_spans(
        full_text: str,
        occupied_spans: List[Tuple[int, int]],
        num_samples: int,
        span_char_len: int = 80,
        max_tries: int = 500,
        ) -> List[Tuple[int, int]]:
    """
    Sample character spans that do NOT overlap any argument unit spans.
    """
    n = len(full_text)
    if n < span_char_len + 5:
        return []

    samples = []
    tries = 0
    while len(samples) < num_samples and tries < max_tries:
        tries += 1
        start = random.randint(0, max(0, n - span_char_len - 1))
        end = min(n, start + span_char_len)
        cand = (start, end)

        # reject overlaps with argument spans
        if any(_spans_overlap(cand, s) for s in occupied_spans):
            continue

        # reject spans with no letters (very weak quality gate)
        snippet = full_text[start:end]
        if not any(c.isalpha() for c in snippet):
            continue

        samples.append(cand)

    return samples


class RTCHFDataset(Dataset):
    """
    RTC dataset with:
      - gold positives (support/attack)
      - sampled arg-arg negatives
      - hard arg-arg negatives (nearby confusers)
      - nonarg-arg negatives (sampled non-arg spans)
    """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer_name: str,
            max_length: int,
            window_size: int = 50,
            neg_per_pos: int = 1,  # random arg-arg negatives per positive
            hard_neg_per_pos: int = 2,  # hard arg-arg negatives per positive
            hard_pool_k: int = 4,  # how many "nearby" candidates to consider around src/tgt
            num_nonarg_spans: int = 2,  # non-arg spans sampled per conversation
            nonarg_span_char_len: int = 30,  # length of each non-arg span
            seed: int = 42,
            max_pairs_conv: int = 30,
            use_context: bool = False,
            ):
        self.data = data.reset_index(drop=True)

        tokenizer_name_lc = tokenizer_name.lower()
        if "xlm-roberta" in tokenizer_name_lc:
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

        self.max_length = max_length
        self.window_size = window_size

        self.id_to_label, self.label_to_id = get_label_maps_ms_RTC()

        # self.texts = []
        self.seq_a_list = []
        self.seq_b_list = []

        # mark the argument spans inside the sequences
        self.tgt_char_spans: List[Tuple[int, int]] = []
        self.src_char_spans: List[Tuple[int, int]] = []

        self.labels = []

        for _, row in self.data.iterrows():
            full_text = str(row["conversation_text"])
            arg_obj = json.loads(row["argument_objects"])

            units = arg_obj.get("argument_units", []) or []
            rels = arg_obj.get("relations", []) or []

            # 1) compute spans for units
            unit_spans: Dict[int, Tuple[int, int]] = {}
            unit_texts: Dict[int, str] = {}
            for u in units:
                uid = int(u.get("id"))
                u_text = str(u.get("text", ""))
                try:
                    start, end = _find_unit_span_in_text(full_text, u_text)
                except ValueError:
                    continue
                unit_spans[uid] = (start, end)
                unit_texts[uid] = u_text

            # if less than 2 units exist in the conversation skip it,
            # though all conversations in training data should have more than two.
            if len(unit_spans) < 2:
                continue

            # 2) gold relations map
            rel_map: Dict[Tuple[int, int], str] = {}
            pos_pairs: List[Tuple[int, int]] = []
            for r in rels:
                try:
                    s_id = int(r.get("source_id"))
                    t_id = int(r.get("target_id"))
                    r_type = str(r.get("type", "")).lower()
                except Exception:
                    continue
                if r_type not in ("support", "attack"):
                    continue
                if s_id in unit_spans and t_id in unit_spans:
                    rel_map[(s_id, t_id)] = r_type
                    pos_pairs.append((s_id, t_id))

            # sort unit ids by appearance
            unit_ids = sorted(unit_spans.keys(), key=lambda u_id: unit_spans[u_id][0])

            all_pairs = []

            # 3) candidate directed pairs (arg-arg) respecting your directionality
            for src_id in unit_ids:
                for tgt_id in unit_ids:
                    if src_id == tgt_id:
                        continue
                    s_start, _ = unit_spans[src_id]
                    t_start, _ = unit_spans[tgt_id]
                    #  ensure directionality
                    if s_start <= t_start:
                        continue
                    all_pairs.append((src_id, tgt_id))

            if not all_pairs:
                continue

            # positives that actually respect directionality (keep consistent with inference)
            dir_pos_pairs = [p for p in pos_pairs if p in all_pairs]

            # random arg-arg negatives
            neg_pool = [p for p in all_pairs if p not in rel_map]

            # # 5) hard negatives around each positive
            # hard_negs = []
            # # index mapping for "nearby in text"
            # idx_of = {uid: i for i, uid in enumerate(unit_ids)}
            #
            # for (s_id, t_id) in dir_pos_pairs:
            #     s_i = idx_of.get(s_id, None)
            #     t_i = idx_of.get(t_id, None)
            #     if s_i is None or t_i is None:
            #         continue
            #
            #     # nearby targets around t_id
            #     tgt_candidates = []
            #     for j in range(max(0, t_i - hard_pool_k), min(len(unit_ids), t_i + hard_pool_k + 1)):
            #         cand_t = unit_ids[j]
            #         if cand_t == t_id:
            #             continue
            #         cand_pair = (s_id, cand_t)
            #         if cand_pair in all_pairs and cand_pair not in rel_map:
            #             tgt_candidates.append(cand_pair)
            #
            #     # nearby sources around s_id
            #     src_candidates = []
            #     for j in range(max(0, s_i - hard_pool_k), min(len(unit_ids), s_i + hard_pool_k + 1)):
            #         cand_s = unit_ids[j]
            #         if cand_s == s_id:
            #             continue
            #         cand_pair = (cand_s, t_id)
            #         if cand_pair in all_pairs and cand_pair not in rel_map:
            #             src_candidates.append(cand_pair)
            #
            #     # sample a few from each side
            #     if tgt_candidates:
            #         hard_negs.extend(random.sample(tgt_candidates, k=min(hard_neg_per_pos, len(tgt_candidates))))
            #     if src_candidates:
            #         hard_negs.extend(random.sample(src_candidates, k=min(hard_neg_per_pos, len(src_candidates))))
            #
            # # de-dup
            # hard_negs = list(dict.fromkeys(hard_negs))

            # non-arg spans and nonarg-arg pairs (label=non)
            occupied = list(unit_spans.values())
            nonarg_spans = _sample_nonarg_spans(
                full_text,
                occupied_spans=occupied,
                num_samples=num_nonarg_spans,
                span_char_len=nonarg_span_char_len,
                )

            # assign pseudo ids (negative, to avoid collisions)
            nonarg_ids = []
            next_nonarg_id = -1
            for (ns, ne) in nonarg_spans:
                nid = next_nonarg_id
                next_nonarg_id -= 1
                nonarg_ids.append(nid)
                unit_spans[nid] = (ns, ne)
                unit_texts[nid] = full_text[ns:ne]

            # build nonarg-arg pairs (both directions, respecting directionality rule)
            nonarg_pairs = []
            for nid in nonarg_ids:
                for uid in unit_ids:
                    # (arg -> nonarg)
                    s_start, _ = unit_spans[uid]
                    t_start, _ = unit_spans[nid]
                    if s_start > t_start:
                        nonarg_pairs.append((uid, nid))  # non
                    # (nonarg -> arg)
                    s_start, _ = unit_spans[nid]
                    t_start, _ = unit_spans[uid]
                    if s_start > t_start:
                        nonarg_pairs.append((nid, uid))  # non

            # keep positive pairs (sup or attack)
            pos_pairs_unique = list(dict.fromkeys(dir_pos_pairs))
            num_pos = len(pos_pairs_unique)

            # pool the negative examples
            rand_neg_candidates = []
            if neg_pool:
                want_rand_neg = neg_per_pos * max(1, num_pos)
                rand_neg_candidates.extend(
                    random.sample(neg_pool, k=min(want_rand_neg, len(neg_pool)))
                    )

            # combine negatives: random arg-arg, and nonarg-arg
            neg_candidates = []
            for p in rand_neg_candidates + nonarg_pairs:
                # ensure we don't treat a gold positive as negative
                if p in rel_map:
                    continue
                neg_candidates.append(p)

            # de-dup negatives while preserving order, and avoid duplicating positives
            seen_neg = set()
            neg_final = []
            for p in neg_candidates:
                if p in seen_neg:
                    continue
                if p in pos_pairs_unique:
                    continue
                seen_neg.add(p)
                neg_final.append(p)

            if max_pairs_conv is not None:
                if num_pos >= max_pairs_conv:
                    # More positives than budget: keep ALL positives, no negatives
                    selected_pairs = pos_pairs_unique
                else:
                    remaining = max_pairs_conv - num_pos
                    if remaining > 0 and neg_final:
                        chosen_negs = random.sample(
                            neg_final,
                            k=min(remaining, len(neg_final))
                            )
                    else:
                        chosen_negs = []
                    selected_pairs = pos_pairs_unique + chosen_negs
            else:
                # no cap: keep everything
                selected_pairs = pos_pairs_unique + neg_final

            # 8) materialize examples + labels
            for (src_id, tgt_id) in selected_pairs:
                t_start, t_end = unit_spans[tgt_id]
                c_start = max(0, t_start - self.window_size)
                c_end = min(len(full_text), t_end + self.window_size)
                context = full_text[c_start:c_end]
                # further limit the size of context (extracted from the window)
                # to reduce misclassification chances by LM.
                max_ctx_chars = 80
                if context and len(context) > max_ctx_chars:
                    context = context[:max_ctx_chars]

                # context ideally includes both arguments but may not be
                # large enough to include source. It is extremely unlikely.
                if use_context:
                    seq_a = f"{context}\n\n{unit_texts[tgt_id]}"
                else:
                    seq_a = f"{unit_texts[tgt_id]}"

                seq_b = f"{unit_texts[src_id]}"

                # store char spans of argument units in seq a and seq b
                tgt_text = unit_texts[tgt_id]
                src_text = unit_texts[src_id]
                # Check that the src and tgt exist in their seq and find their start and end.
                tgt_s = seq_a.rfind(tgt_text)
                if tgt_s < 0:
                    tgt_s = 0
                tgt_e = tgt_s + len(tgt_text)
                src_s = seq_b.rfind(src_text)
                if src_s < 0:
                    src_s = 0
                src_e = src_s + len(src_text)

                # note the labels using the gold relation map, otherwise "non"
                label_string = rel_map.get((src_id, tgt_id), "non")
                label_id = self.label_to_id.get(label_string, self.label_to_id["non"])

                self.seq_a_list.append(seq_a)
                self.seq_b_list.append(seq_b)
                self.tgt_char_spans.append((tgt_s, tgt_e))
                self.src_char_spans.append((src_s, src_e))
                self.labels.append(label_id)

        encodings = self.tokenizer(
            self.seq_a_list,
            self.seq_b_list,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True
            )

        # build token-level masks for pooling
        input_ids = encodings["input_ids"]
        offsets = encodings["offset_mapping"]

        tgt_masks = []
        src_masks = []

        for i in range(len(input_ids)):
            seq_ids = encodings.sequence_ids(i)
            tgt_s, tgt_e = self.tgt_char_spans[i]
            src_s, src_e = self.src_char_spans[i]

            tgt_mask = [0] * len(input_ids[i])
            src_mask = [0] * len(input_ids[i])

            for j, (seq_id, (s,e)) in enumerate(zip(seq_ids, offsets[i])):
                if seq_id is None:
                    continue
                if s is None or e is None:
                    continue
                if e <= s:
                    continue

                if seq_id == 0:
                    if not (e <= tgt_s and s >= tgt_e):
                        tgt_mask[j] = 1
                elif seq_id == 1:
                    if not (e <= src_s and s >= src_e):
                        src_mask[j] = 1
            tgt_masks.append(tgt_mask)
            src_masks.append(src_mask)

        # remove offset mapping before tensorization
        encodings.pop("offset_mapping", None)

        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        self.encodings["tgt_mask"] = torch.tensor(tgt_masks, dtype=torch.long)
        self.encodings["src_mask"] = torch.tensor(src_masks, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def prepare_data_for_AUE(config):
    """
    Returns a Dataset for AUE token classification.
    Expects:
      - config["AUE"]["data_path"]
      - config["AUE"]["model_name_or_path"]
      - config["AUE"]["max_length"]
    """
    data = load_dataframe(config)
    dataset = AUEHFDataset(
        data=data,
        tokenizer_name=config["AUE"]["model_name_or_path"],
        max_length=config["AUE"]["max_length"]
        )

    all_labels_flat = []
    for lab in dataset.labels:
        # lab is a 1D tensor per example including -100; filter those out
        all_labels_flat.extend([x for x in lab.tolist() if x != -100])

    classes = np.unique(all_labels_flat)
    weights = compute_class_weight("balanced", classes=classes, y=all_labels_flat)
    # map to [O, B, I] order
    id_to_label, label_to_id = get_label_maps_ms_AUE()
    class_weights = [0.0] * len(label_to_id)
    for cls, w in zip(classes, weights):
        class_weights[cls] = float(w)

    return dataset, class_weights


def prepare_data_for_RTC(config):
    """
    Returns a Dataset for RTC classification.
    Expects:
      - config["RTC"]["data_path"]
      - config["RTC"]["model_name_or_path"]
      - config["RTC"]["max_length"]
    """
    data = load_dataframe(config)
    rtc_cfg = config.get("RTC", {})
    dataset = RTCHFDataset(
        data=data,
        tokenizer_name=rtc_cfg.get("model_name_or_path", "FacebookAI/xlm-roberta-large"),
        max_length=config["RTC"]["max_length"],
        window_size=rtc_cfg.get("window_size", 50),
        neg_per_pos=rtc_cfg.get("neg_per_pos", 2),
        hard_neg_per_pos=rtc_cfg.get("hard_neg_per_pos", 2),
        hard_pool_k=rtc_cfg.get("hard_pool_k", 4),
        num_nonarg_spans=rtc_cfg.get("num_nonarg_spans", 2),
        nonarg_span_char_len=rtc_cfg.get("nonarg_span_char_len", 80),
        seed=config.get("experiment", {}).get("seed", 42),
        max_pairs_conv=rtc_cfg.get("max_pairs_conv", 30),
        use_context=rtc_cfg.get("use_context", False),
        )
    # for debugging
    print("For debugging")
    for i in range(1):
        print(dataset.seq_a_list[i])
        print(dataset.seq_b_list[i])
        print(dataset.labels[i])

    return dataset


class ArgumentClassifier(nn.Module):
    """
    Wraps AutoModelForMaskedLM to perform Classification tasks.
    Adapts to both standard and Longformer-based architectures.
    """

    def __init__(
            self, model_name,
            num_labels: int,
            is_long_model: bool = False,
            task_type: str = "AUE"
            ):
        super().__init__()
        self.task_type = task_type
        self.is_long_model = is_long_model

        load_kwargs = {}
        if self.is_long_model:
            load_kwargs["trust_remote_code"] = True

        self.backbone = AutoModel.from_pretrained(model_name, **load_kwargs)

        self.backbone.gradient_checkpointing_enable()
        self.backbone.config.use_cache = False

        try:
            self._supports_global_attention = (
                    "global_attention_mask" in inspect.signature(self.backbone.forward).parameters
            )
        except (TypeError, ValueError):
            # Extremely defensive fallback (some wrapped forwards can break signature())
            self._supports_global_attention = False

        # Classifier Head
        hidden = self.backbone.config.hidden_size
        if self.task_type == "RTC":
            # we pool separately over target, source and combine them
            self.classifier = nn.Linear(3 * hidden, num_labels)
        else:
            self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(
            self, input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            src_mask: Optional[torch.Tensor] = None,
            ):
        """

        :param input_ids:
        :param attention_mask:
        :param global_attention_mask:
        :param tgt_mask:
        :param src_mask:
        :return:
        """
        # Base kwargs
        model_kwargs = {
            "input_ids": input_ids,
            "output_hidden_states": True,
            }
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask

        # Only include global_attention_mask when BOTH:
        # - config says it's a long model
        # - caller provided a GAM
        # - backbone forward actually supports it
        if (
                self.is_long_model
                and global_attention_mask is not None
                and self._supports_global_attention
        ):
            model_kwargs["global_attention_mask"] = global_attention_mask

        # outputs = self.encoder(**model_kwargs)
        outputs = self.backbone(**model_kwargs)

        # Extract last hidden state: [batch, seq_len, hidden]
        last_hidden_state = outputs.hidden_states[-1]
        sequence_output = self.dropout(last_hidden_state)

        if self.task_type == "AUE":
            # token-level classification for every position in the discussion
            # shape: [batch, seq_len, num_labels]
            logits = self.classifier(sequence_output)
        else:
            # process for RTC (Sequence classification)
            # shape [batch, num_labels]
            # span pooling over the target and source units only (not over context)!
            def span_mean(x: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
                if m is None:
                    # do masked mean over all tokens
                    if attention_mask is None:
                        return x.mean(dim=1)
                    am = attention_mask.unsqueeze(-1).to(x.dtype)
                    summed = (x * am).sum(dim=1)
                    denom = am.sum(dim=1).clamp(min=1.0)
                    return summed / denom

                m = m.unsqueeze(-1).to(x.dtype)
                summed = (x * m).sum(dim=1)
                denom = m.sum(dim=1).clamp(min=1.0)
                return summed / denom

            tgt_unit = span_mean(sequence_output, tgt_mask)
            src_unit = span_mean(sequence_output, src_mask)

            pair_representation = torch.cat([tgt_unit, src_unit, tgt_unit * src_unit], dim=-1)
            logits = self.classifier(self.dropout(pair_representation))

        return logits