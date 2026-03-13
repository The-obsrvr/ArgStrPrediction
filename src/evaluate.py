import os
import json
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score


# --------- Data structures ---------
@dataclass
class Span:
    id: int
    start: int
    end: int
    text: str


@dataclass
class Relation:
    source_id: int
    target_id: int
    rel_type: str  # "support" or "attack"


# --------- Span / relation utilities ---------

def find_offsets(conversation_text: str, span_text: str) -> Tuple[int, int]:
    """
    Case-insensitive substring match to recover character offsets of a span.
    Returns (-1, -1) if not found.
    """
    conv_low = conversation_text.lower()
    span_low = span_text.lower()
    # search for span in text
    start = conv_low.find(span_low)
    # if not found
    if start == -1:
        return -1, -1
    end = start + len(span_text)
    return start, end


def span_overlap(s1: Span, s2: Span) -> float:
    """Compute overlap ratio between two spans as IoU over character ranges."""
    inter_start = max(s1.start, s2.start)
    inter_end = min(s1.end, s2.end)
    inter = max(0, inter_end - inter_start)
    if inter == 0:
        return 0.0
    union = max(s1.end, s2.end) - min(s1.start, s2.start)
    return inter / union


def greedy_best_match(
        gold_spans: List[Span],
        pred_spans: List[Span],
        overlap_thr: float,
        ) -> Dict[int, int]:
    """
    Greedy maximum-overlap one-to-one matching between gold and predicted spans.
    Returns a mapping: gold_id -> pred_id.
    """
    candidates: List[Tuple[float, int, int]] = []
    for g in gold_spans:
        for p in pred_spans:
            ov = span_overlap(g, p)
            if ov >= overlap_thr:
                candidates.append((ov, g.id, p.id))

    # Sort by overlap descending
    candidates.sort(reverse=True, key=lambda x: x[0])

    gold_matched = set()
    pred_matched = set()
    mapping: Dict[int, int] = {}

    # parse through candidates, sorted in descending order of overlaps, and select the most overlapped matches as true.
    for ov, gid, pid in candidates:
        # skip already matched units
        if gid in gold_matched or pid in pred_matched:
            continue
        mapping[gid] = pid
        gold_matched.add(gid)
        pred_matched.add(pid)

    return mapping


def char_level_labels(text_len: int, spans: List[Span]) -> np.ndarray:
    """
    Build a binary vector (length text_len) with 1 for ARG characters, 0 for NON.
    Overlapping spans are treated as ARG if any span covers the character.
    """
    labels = np.zeros(text_len, dtype=int)
    for s in spans:
        start = max(0, min(text_len, s.start))
        end = max(0, min(text_len, s.end))
        if end > start:
            labels[start:end] = 1
    return labels


# --------- AUE metrics ---------


def evaluate_aue_char_level(
        text: str,
        gold_spans: List[Span],
        pred_spans: List[Span],
        ) -> Dict[str, float]:
    """
    NOT USED
    Character-level evaluation for AUE:
    ARG = 1, NON = 0
    Reports weighted / macro precision, recall, F1, and class-wise values.
    """
    L = len(text)
    y_true = char_level_labels(L, gold_spans)
    y_pred = char_level_labels(L, pred_spans)

    # Macro + weighted
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        labels=[0, 1],
        zero_division=0,
        )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        labels=[0, 1],
        zero_division=0,
        )

    # Class-wise: [NON, ARG]
    p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=[0, 1],
        zero_division=0,
        )

    return {
        # weighted
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted),
        # macro
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        # class-wise
        "precision_NON": float(p_cls[0]),
        "recall_NON": float(r_cls[0]),
        "f1_NON": float(f_cls[0]),
        "precision_ARG": float(p_cls[1]),
        "recall_ARG": float(r_cls[1]),
        "f1_ARG": float(f_cls[1]),
        }


def evaluate_aue_span_level(
        gold_spans: List[Span],
        pred_spans: List[Span],
        overlap_thr: float,
        ) -> Dict[str, float]:
    """
    Span-level AUE evaluation for a given overlap threshold.

    Returns precision, recall, F1 for the ARG class, plus TP/FP/FN counts.
    """
    mapping = greedy_best_match(gold_spans, pred_spans, overlap_thr=overlap_thr)

    tp = len(mapping)
    fp = len(pred_spans) - len(set(mapping.values()))
    fn = len(gold_spans) - len(mapping)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        }


# --------- RTC metrics ---------
def evaluate_rtc(
        gold_rels: List[Relation],
        pred_rels: List[Relation],
        gold_to_pred_span: Dict[int, int],
        ) -> Dict[str, float]:
    """
    Full RTC evaluation: a predicted relation is correct iff:
    - both endpoints are matched to gold units, and
    - the predicted edge connects those matched units with the correct type.
    """
    pred_set = {(r.source_id, r.target_id, r.rel_type) for r in pred_rels}
    gold_set = {(r.source_id, r.target_id, r.rel_type) for r in gold_rels}

    # Build reverse mapping: pred_id -> gold_id
    pred_to_gold_span = {pid: gid for gid, pid in gold_to_pred_span.items()}

    # Convert predicted relations into "gold-indexed" triples
    # (only for relations where both endpoints correspond to some gold unit)
    pred_as_gold_space = set()
    for (ps, pt, t) in pred_set:
        if ps in pred_to_gold_span and pt in pred_to_gold_span:
            gs = pred_to_gold_span[ps]
            gt = pred_to_gold_span[pt]
            pred_as_gold_space.add((gs, gt, t))

    tp = len(gold_set & pred_as_gold_space)
    fp = len(pred_as_gold_space - gold_set)
    fn = len(gold_set - pred_as_gold_space)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision_full": float(precision),
        "recall_full": float(recall),
        "f1_full": float(f1),
        "tp_full": int(tp),
        "fp_full": int(fp),
        "fn_full": int(fn),
        }


def evaluate_rtc_type_only(
        gold_rels: List[Relation],
        pred_rels: List[Relation],
        gold_to_pred_span: Dict[int, int],
        ) -> Dict[str, float]:
    """
    Type-only RTC evaluation for error analysis:
    - Restrict to edges whose endpoints are correctly matched & paired.
    - Evaluate whether the relation type is correct.
    """
    pred_to_gold_span = {pid: gid for gid, pid in gold_to_pred_span.items()}

    # Gold relations indexed by endpoint pair only
    gold_by_pair: Dict[Tuple[int, int], str] = {
        (r.source_id, r.target_id): r.rel_type for r in gold_rels
        }

    y_true: List[str] = []
    y_pred: List[str] = []

    for r in pred_rels:
        # keep only if both endpoints mapped to gold
        if r.source_id not in pred_to_gold_span or r.target_id not in pred_to_gold_span:
            continue
        gs = pred_to_gold_span[r.source_id]
        gt = pred_to_gold_span[r.target_id]
        pair = (gs, gt)
        if pair not in gold_by_pair:
            # endpoint pairing is wrong; ignore for type-only analysis
            continue

        y_true.append(gold_by_pair[pair])
        y_pred.append(r.rel_type)

    if not y_true:
        return {
            "f1_weighted": 0.0,
            "f1_macro": 0.0,
            "support_f1": 0.0,
            "attack_f1": 0.0,
            }

    labels = ["support", "attack"]
    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0,
        )
    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=labels,
        zero_division=0,
        )
    f1_per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0,
        )

    return {
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "support_f1": float(f1_per_class[0]),
        "attack_f1": float(f1_per_class[1]),
        }


# --------- Gold loading ---------
def load_gold(csv_path) -> Dict[str, Dict[str, Any]]:
    """
    Load gold data from CSV into:
      gold[conversation_id] = {
        "text": ...,
        "spans": List[Span],
        "relations": List[Relation]
      }
    """
    df = pd.read_csv(csv_path)
    gold: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        conv_id = str(row["conversation_id"])
        conv_text = row["conversation_text"]
        arg_obj = json.loads(row["argument_objects"])

        spans: List[Span] = []
        for unit in arg_obj.get("argument_units", []):
            uid = int(unit["id"])
            utext = unit["text"]
            start, end = find_offsets(conv_text, utext)
            if start == -1:
                # If we fail to recover offsets, fall back to a zero-length span
                start = 0
                end = 0
            spans.append(Span(id=uid, start=start, end=end, text=utext))

        rels: List[Relation] = []
        for r in arg_obj.get("relations", []):
            rels.append(
                Relation(
                    source_id=int(r["source_id"]),
                    target_id=int(r["target_id"]),
                    rel_type=r["type"],
                    )
                )

        gold[conv_id] = {"text": conv_text, "spans": spans, "relations": rels}

    return gold


# --------- Prediction loading (handles LM formats too) ---------

def _build_spans_from_units(
        units_raw: List[Dict[str, Any]],
        conv_text: str,
        ) -> List[Span]:
    spans: List[Span] = []
    for u in units_raw:
        uid = int(u["id"])
        utext = u.get("text", "")
        start = int(u.get("start", -1))
        end = int(u.get("end", -1))
        if start < 0 or end < 0:
            start, end = find_offsets(conv_text, utext)
        if start == -1:
            start = 0
            end = 0
        spans.append(Span(id=uid, start=start, end=end, text=utext))
    return spans


def _build_relations_from_obj(obj: Dict[str, Any]) -> List[Relation]:
    rels: List[Relation] = []

    # Case 1: unified "relations" field
    if "relations" in obj:
        for r in obj["relations"]:
            rels.append(
                Relation(
                    source_id=int(r["source_id"]),
                    target_id=int(r["target_id"]),
                    rel_type=r["type"],
                    )
                )
        return rels

    # Case 2: separate support / attack relationship lists (old version)
    for r in obj.get("support_relationships", []):
        rels.append(
            Relation(
                source_id=int(r["source_id"]),
                target_id=int(r["target_id"]),
                rel_type="support",
                )
            )
    for r in obj.get("attack_relationships", []):
        rels.append(
            Relation(
                source_id=int(r["source_id"]),
                target_id=int(r["target_id"]),
                rel_type="attack",
                )
            )

    return rels


def load_predictions(
        pred_path: str,
        gold: Dict[str, Dict[str, Any]],
        ) -> Dict[str, Dict[str, Any]]:
    """
    Load predictions from JSON in multiple possible formats and
    normalize them to:
      preds[conversation_id] = {
        "spans": List[Span],
        "relations": List[Relation]
      }

    Supported formats:

    1) Dict keyed by conversation id:
       {
         "21585": {
           "argument_units": [...],
           "relations": [...]
         },
         ...
       }

       or with "support_relationships"/"attack_relationships" instead of "relations".

       If the value for a conversation id is null or not a dict, it is skipped.

    2) Dict keyed by conversation id with only "argument_units"
       (LM extraction-only outputs).

    3) List of conversation records (LM-style):
       [
         {
           "conversation_id": ...,
           "conversation_text": ...,
           "argument_objects": {
             "argument_units": [...],
             "relations": [...]
            }
         },
         ...
       ]
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        preds_raw: Any = json.load(f)

    preds: Dict[str, Dict[str, Any]] = {}

    # --- Format 1 & 2: dict keyed by conversation id ---
    if isinstance(preds_raw, dict):
        for conv_id, obj in preds_raw.items():
            conv_id_str = str(conv_id)

            # Skip if no gold for this id
            if conv_id_str not in gold:
                continue

            # Robustness: skip null / non-dict entries
            if not isinstance(obj, dict):
                # e.g. "12345": null → no prediction for this conversation
                continue

            conv_text = gold[conv_id_str]["text"]

            # argument_units may be directly under obj, or under obj["argument_objects"]
            if "argument_units" in obj:
                units_raw = obj["argument_units"]
            else:
                units_raw = obj.get("argument_objects", {}).get("argument_units", [])

            spans = _build_spans_from_units(units_raw, conv_text)
            rels = _build_relations_from_obj(obj)

            preds[conv_id_str] = {"spans": spans, "relations": rels}

        return preds

    # --- Format 3: list of conversation records (LM-style) ---
    if isinstance(preds_raw, list):
        for rec in preds_raw:
            # Robustness: skip null / non-dict records
            if not isinstance(rec, dict):
                continue

            conv_id_val = rec.get("conversation_id")
            if conv_id_val is None:
                continue
            conv_id_str = str(conv_id_val)
            if conv_id_str not in gold:
                continue

            conv_text = gold[conv_id_str]["text"]
            arg_obj = rec.get("argument_objects", rec)

            units_raw = arg_obj.get("argument_units", [])
            spans = _build_spans_from_units(units_raw, conv_text)
            rels = _build_relations_from_obj(arg_obj)

            preds[conv_id_str] = {"spans": spans, "relations": rels}

        return preds

    raise ValueError(f"Unsupported prediction JSON structure in {pred_path!r}")


# --------- Dataset-level aggregation + unmatched units ---------
# def evaluate_dataset(
#         gold: Dict[str, Dict[str, Any]],
#         preds: Dict[str, Dict[str, Any]],
#         ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, List[int]]]]]:
#     """
#     Run all metrics over the intersection of gold/pred conversation IDs
#     and return:
#       - a dict of aggregate scores
#       - a dict recording unmatched units per conversation and threshold
#         for error analysis.
#     """
#     common_ids = sorted(set(gold.keys()) & set(preds.keys()))
#
#     aue_results_50: List[Dict[str, float]] = []
#     aue_results_100: List[Dict[str, float]] = []
#
#     rtc_results_full_50: List[Dict[str, float]] = []
#     rtc_results_full_100: List[Dict[str, float]] = []
#
#     rtc_type_only_50: List[Dict[str, float]] = []
#     rtc_type_only_100: List[Dict[str, float]] = []
#
#     unmatched_50: Dict[str, Dict[str, List[int]]] = {}
#     unmatched_100: Dict[str, Dict[str, List[int]]] = {}
#
#     for conv_id in common_ids:
#         g = gold[conv_id]
#         p = preds[conv_id]
#
#         text = g["text"]
#         gold_spans: List[Span] = g["spans"]
#         pred_spans: List[Span] = p["spans"]
#         gold_rels: List[Relation] = g["relations"]
#         pred_rels: List[Relation] = p["relations"]
#
#         m50 = greedy_best_match(gold_spans, pred_spans, overlap_thr=0.5)
#         m100 = greedy_best_match(gold_spans, pred_spans, overlap_thr=1.0)
#
#         unmatched_gold_50 = [s.id for s in gold_spans if s.id not in m50]
#         unmatched_pred_50 = [s.id for s in pred_spans if s.id not in m50.values()]
#         unmatched_50[conv_id] = {
#             "unmatched_gold_ids": unmatched_gold_50,
#             "unmatched_pred_ids": unmatched_pred_50,
#             }
#
#         unmatched_gold_100 = [s.id for s in gold_spans if s.id not in m100]
#         unmatched_pred_100 = [s.id for s in pred_spans if s.id not in m100.values()]
#         unmatched_100[conv_id] = {
#             "unmatched_gold_ids": unmatched_gold_100,
#             "unmatched_pred_ids": unmatched_pred_100,
#             }
#
#         aue_50 = evaluate_aue_span_level(gold_spans, pred_spans, overlap_thr=0.5)
#         aue_100 = evaluate_aue_span_level(gold_spans, pred_spans, overlap_thr=1.0)
#
#         aue_results_50.append(aue_50)
#         aue_results_100.append(aue_100)
#
#         rtc_full_50 = evaluate_rtc(gold_rels, pred_rels, m50)
#         rtc_full_100 = evaluate_rtc(gold_rels, pred_rels, m100)
#
#         rtc_results_full_50.append(rtc_full_50)
#         rtc_results_full_100.append(rtc_full_100)
#
#         rtc_type_50 = evaluate_rtc_type_only(gold_rels, pred_rels, m50)
#         rtc_type_100 = evaluate_rtc_type_only(gold_rels, pred_rels, m100)
#
#         rtc_type_only_50.append(rtc_type_50)
#         rtc_type_only_100.append(rtc_type_100)
#
#         # _ = evaluate_aue_char_level(text, gold_spans, pred_spans)
#
#     def mean_metric(results: List[Dict[str, float]], key: str) -> float:
#         vals: List[float] = []
#         for r in results:
#             v = r.get(key, 0.0)
#             try:
#                 if not math.isnan(v):
#                     vals.append(float(v))
#             except TypeError:
#                 continue
#         return float(np.mean(vals)) if vals else 0.0
#
#     summary: Dict[str, float] = {
#         # AUE span-level @ 50%
#         "aue_50_precision": mean_metric(aue_results_50, "precision"),
#         "aue_50_recall": mean_metric(aue_results_50, "recall"),
#         "aue_50_f1": mean_metric(aue_results_50, "f1"),
#         # AUE span-level @ 100%
#         "aue_100_precision": mean_metric(aue_results_100, "precision"),
#         "aue_100_recall": mean_metric(aue_results_100, "recall"),
#         "aue_100_f1": mean_metric(aue_results_100, "f1"),
#         # RTC full @ 50%
#         "rtc_full_50_precision": mean_metric(rtc_results_full_50, "precision_full"),
#         "rtc_full_50_recall": mean_metric(rtc_results_full_50, "recall_full"),
#         "rtc_full_50_f1": mean_metric(rtc_results_full_50, "f1_full"),
#         # RTC full @ 100%
#         "rtc_full_100_precision": mean_metric(rtc_results_full_100, "precision_full"),
#         "rtc_full_100_recall": mean_metric(rtc_results_full_100, "recall_full"),
#         "rtc_full_100_f1": mean_metric(rtc_results_full_100, "f1_full"),
#         # RTC type-only @ 50%
#         "rtc_type_50_f1_weighted": mean_metric(rtc_type_only_50, "f1_weighted"),
#         "rtc_type_50_f1_macro": mean_metric(rtc_type_only_50, "f1_macro"),
#         "rtc_type_50_support_f1": mean_metric(rtc_type_only_50, "support_f1"),
#         "rtc_type_50_attack_f1": mean_metric(rtc_type_only_50, "attack_f1"),
#         # RTC type-only @ 100%
#         "rtc_type_100_f1_weighted": mean_metric(rtc_type_only_100, "f1_weighted"),
#         "rtc_type_100_f1_macro": mean_metric(rtc_type_only_100, "f1_macro"),
#         "rtc_type_100_support_f1": mean_metric(rtc_type_only_100, "support_f1"),
#         "rtc_type_100_attack_f1": mean_metric(rtc_type_only_100, "attack_f1"),
#         }
#
#     unmatched_info: Dict[str, Dict[str, Dict[str, List[int]]]] = {
#         "50": unmatched_50,
#         "100": unmatched_100,
#         }
#
#     return summary, unmatched_info

def evaluate_dataset(
        gold: Dict[str, Dict[str, Any]],
        preds: Dict[str, Dict[str, Any]],
        ) -> Tuple[
            Dict[str, float],
            Dict[str, Dict[str, Dict[str, List[int]]]],
            Dict[str, int],
        ]:
    """
    Run all metrics over the intersection of gold/pred conversation IDs
    and return:
      - a dict of aggregate scores
      - a dict recording unmatched units per conversation and threshold
        for error analysis.
      - a dict of aggregated error-analysis counts across the dataset.
    """
    common_ids = sorted(set(gold.keys()) & set(preds.keys()))

    aue_results_50: List[Dict[str, float]] = []
    aue_results_100: List[Dict[str, float]] = []

    rtc_results_full_50: List[Dict[str, float]] = []
    rtc_results_full_100: List[Dict[str, float]] = []

    rtc_type_only_50: List[Dict[str, float]] = []
    rtc_type_only_100: List[Dict[str, float]] = []

    unmatched_50: Dict[str, Dict[str, List[int]]] = {}
    unmatched_100: Dict[str, Dict[str, List[int]]] = {}

    # Aggregated error-analysis counts (50% and 100% overlap thresholds)
    error_counts: Dict[str, int] = {
        # span-level unit errors
        "false_positive_units_50": 0,
        "false_negative_units_50": 0,
        "false_positive_units_100": 0,
        "false_negative_units_100": 0,
        # relations whose endpoints cannot be mapped to any gold units
        "endpoint_mismatched_relations_50": 0,
        "endpoint_mismatched_relations_100": 0,
        # pair-level (endpoint-correct) relation errors
        "false_positive_pairs_50": 0,
        "false_negative_pairs_50": 0,
        "incorrectly_classified_pairs_50": 0,
        "false_positive_pairs_100": 0,
        "false_negative_pairs_100": 0,
        "incorrectly_classified_pairs_100": 0,
    }

    for conv_id in common_ids:
        g = gold[conv_id]
        p = preds[conv_id]

        text = g["text"]
        gold_spans: List[Span] = g["spans"]
        pred_spans: List[Span] = p["spans"]
        gold_rels: List[Relation] = g["relations"]
        pred_rels: List[Relation] = p["relations"]

        # --- span matching at two thresholds ---
        m50 = greedy_best_match(gold_spans, pred_spans, overlap_thr=0.5)
        m100 = greedy_best_match(gold_spans, pred_spans, overlap_thr=1.0)

        unmatched_gold_50 = [s.id for s in gold_spans if s.id not in m50]
        unmatched_pred_50 = [s.id for s in pred_spans if s.id not in m50.values()]
        unmatched_50[conv_id] = {
            "unmatched_gold_ids": unmatched_gold_50,
            "unmatched_pred_ids": unmatched_pred_50,
        }

        unmatched_gold_100 = [s.id for s in gold_spans if s.id not in m100]
        unmatched_pred_100 = [s.id for s in pred_spans if s.id not in m100.values()]
        unmatched_100[conv_id] = {
            "unmatched_gold_ids": unmatched_gold_100,
            "unmatched_pred_ids": unmatched_pred_100,
        }

        # Aggregate span-level FP/FN units
        error_counts["false_positive_units_50"] += len(unmatched_pred_50)
        error_counts["false_negative_units_50"] += len(unmatched_gold_50)
        error_counts["false_positive_units_100"] += len(unmatched_pred_100)
        error_counts["false_negative_units_100"] += len(unmatched_gold_100)

        # --- AUE metrics ---
        aue_50 = evaluate_aue_span_level(gold_spans, pred_spans, overlap_thr=0.5)
        aue_100 = evaluate_aue_span_level(gold_spans, pred_spans, overlap_thr=1.0)

        aue_results_50.append(aue_50)
        aue_results_100.append(aue_100)

        # --- RTC metrics (full) ---
        rtc_full_50 = evaluate_rtc(gold_rels, pred_rels, m50)
        rtc_full_100 = evaluate_rtc(gold_rels, pred_rels, m100)

        rtc_results_full_50.append(rtc_full_50)
        rtc_results_full_100.append(rtc_full_100)

        # --- RTC metrics (type-only) ---
        rtc_type_50 = evaluate_rtc_type_only(gold_rels, pred_rels, m50)
        rtc_type_100 = evaluate_rtc_type_only(gold_rels, pred_rels, m100)

        rtc_type_only_50.append(rtc_type_50)
        rtc_type_only_100.append(rtc_type_100)

        # --- relation-level error counts per threshold ---
        def accumulate_relation_errors(mapping: Dict[int, int], suffix: str) -> None:
            # mapping: gold_id -> pred_id
            pred_to_gold_span = {pid: gid for gid, pid in mapping.items()}

            # Endpoint-mismatched relations: predicted edges with at least one
            # endpoint that does not map to any gold unit.
            mismatched = 0
            for r in pred_rels:
                if (r.source_id not in pred_to_gold_span) or (r.target_id not in pred_to_gold_span):
                    mismatched += 1
            error_counts[f"endpoint_mismatched_relations_{suffix}"] += mismatched

            # Now restrict to relations whose endpoints can be mapped
            gold_pairs = {(r.source_id, r.target_id) for r in gold_rels}
            pred_pairs = set()
            gold_type_by_pair: Dict[Tuple[int, int], str] = {
                (r.source_id, r.target_id): r.rel_type for r in gold_rels
            }

            incorrectly_classified = 0

            for r in pred_rels:
                if (r.source_id not in pred_to_gold_span) or (r.target_id not in pred_to_gold_span):
                    continue
                gs = pred_to_gold_span[r.source_id]
                gt = pred_to_gold_span[r.target_id]
                pair = (gs, gt)
                pred_pairs.add(pair)

                # Pair exists in gold, but type is wrong → incorrectly classified pair
                if pair in gold_type_by_pair and r.rel_type != gold_type_by_pair[pair]:
                    incorrectly_classified += 1

            fp_pairs = len(pred_pairs - gold_pairs)
            fn_pairs = len(gold_pairs - pred_pairs)

            error_counts[f"false_positive_pairs_{suffix}"] += fp_pairs
            error_counts[f"false_negative_pairs_{suffix}"] += fn_pairs
            error_counts[f"incorrectly_classified_pairs_{suffix}"] += incorrectly_classified

        accumulate_relation_errors(m50, "50")
        accumulate_relation_errors(m100, "100")

        # _ = evaluate_aue_char_level(text, gold_spans, pred_spans)

    def mean_metric(results: List[Dict[str, float]], key: str) -> float:
        vals: List[float] = []
        for r in results:
            v = r.get(key, 0.0)
            try:
                if not math.isnan(v):
                    vals.append(float(v))
            except TypeError:
                continue
        return float(np.mean(vals)) if vals else 0.0

    summary: Dict[str, float] = {
        # AUE span-level @ 50%
        "aue_50_precision": mean_metric(aue_results_50, "precision"),
        "aue_50_recall": mean_metric(aue_results_50, "recall"),
        "aue_50_f1": mean_metric(aue_results_50, "f1"),
        # AUE span-level @ 100%
        "aue_100_precision": mean_metric(aue_results_100, "precision"),
        "aue_100_recall": mean_metric(aue_results_100, "recall"),
        "aue_100_f1": mean_metric(aue_results_100, "f1"),
        # RTC full @ 50%
        "rtc_full_50_precision": mean_metric(rtc_results_full_50, "precision_full"),
        "rtc_full_50_recall": mean_metric(rtc_results_full_50, "recall_full"),
        "rtc_full_50_f1": mean_metric(rtc_results_full_50, "f1_full"),
        # RTC full @ 100%
        "rtc_full_100_precision": mean_metric(rtc_results_full_100, "precision_full"),
        "rtc_full_100_recall": mean_metric(rtc_results_full_100, "recall_full"),
        "rtc_full_100_f1": mean_metric(rtc_results_full_100, "f1_full"),
        # RTC type-only @ 50%
        "rtc_type_50_f1_weighted": mean_metric(rtc_type_only_50, "f1_weighted"),
        "rtc_type_50_f1_macro": mean_metric(rtc_type_only_50, "f1_macro"),
        "rtc_type_50_support_f1": mean_metric(rtc_type_only_50, "support_f1"),
        "rtc_type_50_attack_f1": mean_metric(rtc_type_only_50, "attack_f1"),
        # RTC type-only @ 100%
        "rtc_type_100_f1_weighted": mean_metric(rtc_type_only_100, "f1_weighted"),
        "rtc_type_100_f1_macro": mean_metric(rtc_type_only_100, "f1_macro"),
        "rtc_type_100_support_f1": mean_metric(rtc_type_only_100, "support_f1"),
        "rtc_type_100_attack_f1": mean_metric(rtc_type_only_100, "attack_f1"),
    }

    unmatched_info: Dict[str, Dict[str, Dict[str, List[int]]]] = {
        "50": unmatched_50,
        "100": unmatched_100,
    }

    return summary, unmatched_info, error_counts


# --------- Experiment discovery & LM handling ---------

def _has_lm_token(name: str) -> bool:
    """
    Return True if the given string has a token 'lm' (case-insensitive)
    when split on underscores / hyphens / dots.
    """
    tokens = re.split(r"[_.\-]", name)
    return any(tok.lower() == "lm" for tok in tokens if tok)


def parse_experiment_name(folder_name: str) -> Dict[str, Any]:
    """
    Parse folder name into a small metadata dict.

    Handles both "normal" and LM-style names, and extracts a 'data' token
    if present.
    """
    parts = folder_name.split("_")
    is_lm = _has_lm_token(folder_name)

    parsed: Dict[str, Any] = {
        "task": "",
        "model_and_context": "",
        "seed": "",
        "timestamp": "",
        "data": "",
        "is_lm": bool(is_lm),
        }

    data_token_map = {
        "qt": "qt",
        "rip": "rip",
        "reddit": "reddit",
        "red": "reddit",  # treat 'red' as 'reddit'
        }

    if not parts:
        return parsed

    # detect data token index
    data_idx = None
    for i, tok in enumerate(parts):
        key = tok.lower()
        if key in data_token_map:
            data_idx = i
            parsed["data"] = data_token_map[key]
            break

    # detect timestamp block (trailing long numeric tokens)
    ts_indices: List[int] = []
    for i in range(len(parts) - 1, -1, -1):
        tok = parts[i]
        if tok.isdigit() and len(tok) >= 6:
            ts_indices.append(i)
            continue
        break
    ts_indices.sort()

    # detect seed index
    seed_idx = None
    if ts_indices:
        seed_search_end = ts_indices[0]
        for i in range(seed_search_end - 1, -1, -1):
            if parts[i].isdigit():
                seed_idx = i
                break
    if seed_idx is None:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                seed_idx = i
                break

    # timestamp
    if ts_indices:
        parsed["timestamp"] = "_".join(parts[ts_indices[0]:])

    # task & LM flag
    if is_lm and parts[0].lower() == "lm":
        parsed["task"] = "lm"
        start_model = 1
    else:
        parsed["task"] = parts[0]
        start_model = 1

    # model_and_context
    end_model = seed_idx if seed_idx is not None else len(parts)
    model_tokens = [
        parts[i]
        for i in range(start_model, end_model)
        if i != data_idx
        ]
    parsed["model_and_context"] = "_".join(model_tokens) if model_tokens else ""

    # seed
    if seed_idx is not None:
        parsed["seed"] = parts[seed_idx]

    return parsed


def parse_lm_pred_filename(filename: str) -> Dict[str, Any]:
    """
    Parse LM (non LLM) folder prediction file names such as:
      - FT_ms_qt_42.json
      - ss_LM_reddit_42.json
    """
    base = os.path.splitext(filename)[0]
    tokens = base.split("_")

    data_token_map = {
        "qt": "qt",
        "rip": "rip",
        "reddit": "reddit",
        "red": "reddit",  # treat 'red' as 'reddit'
        }

    data = ""
    data_idx = None
    for i, tok in enumerate(tokens):
        key = tok.lower()
        if key in data_token_map:
            data = data_token_map[key]
            data_idx = i
            break

    is_lm_file = any(tok.lower() == "lm" for tok in tokens)
    variant = tokens[0] if tokens else ""

    file_seed = ""
    for tok in reversed(tokens):
        if tok.isdigit():
            file_seed = tok
            break

    file_model_ctx_tokens = [
        t
        for i, t in enumerate(tokens)
        if i != data_idx and not (t.isdigit() and i == len(tokens) - 1)
        ]
    file_model_ctx = "_".join(file_model_ctx_tokens) if file_model_ctx_tokens else ""

    return {
        "file_base": base,
        "file_variant": variant,
        "file_data": data,
        "file_seed": file_seed,
        "file_model_and_context": file_model_ctx,
        "file_is_lm": is_lm_file,
        }


def discover_experiments(experiments_root: str) -> List[Dict[str, Any]]:
    """
    Recursively walk `experiments_root` and detect experiment setups.

    Non-LM experiment folder:
      - must directly contain at least one *.config file
      - must directly contain at least one *.json file

    LM experiment folder (name has token 'lm'):
      - must directly contain at least one *.json file
      - *.config is optional
      - each *.json file inside becomes its own "experiment row"
    """
    experiments: List[Dict[str, Any]] = []

    for root, _dirs, files in os.walk(experiments_root):
        folder_name = os.path.basename(root)
        meta = parse_experiment_name(folder_name)

        config_files = [f for f in files if f.endswith(".config")]
        json_files = [f for f in files if f.endswith(".json")]

        if not json_files:
            continue

        is_lm_folder = bool(meta.get("is_lm", False))

        if not is_lm_folder and not config_files:
            continue

        config_files.sort()
        json_files.sort()

        config_path = os.path.join(root, config_files[0]) if config_files else ""

        if is_lm_folder:
            for jf in json_files:
                file_meta = parse_lm_pred_filename(jf)

                if "unmatched" in file_meta["file_base"]:
                    continue

                exp_info: Dict[str, Any] = {
                    "name": f"{folder_name}",
                    "path": root,
                    "config_path": config_path,
                    "pred_path": os.path.join(root, jf),
                    }
                exp_info.update(meta)

                if file_meta["file_data"]:
                    exp_info["data"] = file_meta["file_data"]

                exp_info["lm_variant"] = file_meta["file_variant"]
                exp_info["lm_file_seed"] = file_meta["file_seed"]
                exp_info["lm_file_model_and_context"] = file_meta["file_model_and_context"]
                exp_info["is_lm_file"] = file_meta["file_is_lm"]

                experiments.append(exp_info)
        else:
            pred_path = os.path.join(root, json_files[0])
            exp_info = {
                "name": folder_name,
                "path": root,
                "config_path": config_path,
                "pred_path": pred_path,
                }
            exp_info.update(meta)
            experiments.append(exp_info)

    return experiments


# --------- LLM log parsing & efficiency reporting ---------
def parse_llm_experiment_log(log_path: str, task_type: str = "ms") -> List[Dict[str, Any]]:
    """
    Parse experiment.log for non-LM (LLM) experiments.

    For ms (multi-stage) runs:
      - We expect separate AUE and RTC tasks.
      - We track attempts & tokens per task.

    For ss (single-shot) runs:
      - We treat all SUCCESS lines as belonging to a single combined stage.
      - We do NOT force an RTC stage to exist; AUE/RTC statuses mirror the overall status.

    Expected patterns (per conversation/prediction), especially for ms:
      - [DEBUG] current_task=AUE ...
      - [Attempt 1/2] Generating...
      - SUCCESS on attempt k. Total tokens: N   (for AUE)
      - [DEBUG] current_task=RTC ...
      - [Attempt ...]
      - SUCCESS on attempt k. Total tokens: M   (for RTC)
      - Prediction i took X seconds

    Returns a list of per-prediction records:
      {
        "prediction_index": int,
        "time_seconds": float,
        "total_tokens": int,
        "aue_tokens": int,
        "rtc_tokens": int,
        "aue_status": "success_first" | "success_second" | "failure",
        "rtc_status": "success_first" | "success_second" | "failure",
        "overall_status": "success_first" | "success_second" | "failure",
      }
    """
    if not os.path.exists(log_path):
        return []

    task_type_norm = (task_type or "").lower()
    is_ms = task_type_norm == "ms"

    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    records: List[Dict[str, Any]] = []
    seen_prediction_indices = set()
    last_prediction_index = -1
    current_task = None
    current: Dict[str, Any] = {
        "aue_attempts": [],  # list of {"attempt": int, "tokens": int, "status": "success"}
        "rtc_attempts": [],
        "combined_attempts": [],  # used for ss, or as a fallback
        }

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ---- detect current task (for ms) ----
        m_task = re.search(r"current_task=(\w+)", line)
        if m_task:
            current_task = m_task.group(1).upper()

        # ---- detect a successful attempt + tokens ----
        m_succ = re.search(r"SUCCESS on attempt (\d+)\. Total tokens: (\d+)", line)
        if m_succ:
            attempt = int(m_succ.group(1))
            tokens = int(m_succ.group(2))
            rec = {"attempt": attempt, "tokens": tokens, "status": "success"}

            if is_ms:
                # In ms runs we care which task (AUE vs RTC) this belongs to
                if current_task == "AUE":
                    current["aue_attempts"].append(rec)
                elif current_task == "RTC":
                    current["rtc_attempts"].append(rec)
                else:
                    # Fallback: if task not tagged for some reason, treat as combined
                    current["combined_attempts"].append(rec)
            else:
                # ss: treat everything as a single combined stage
                current["combined_attempts"].append(rec)

        # ---- detect prediction completion ----
        m_pred = re.search(r"Prediction (\d+) took ([0-9.]+) seconds", line)
        if m_pred:
            pred_idx = int(m_pred.group(1))
            time_sec = float(m_pred.group(2))

            # ---- fill gaps: emit empty records for skipped predictions ----
            # This handles null / aborted predictions
            for missing_idx in range(last_prediction_index + 1, pred_idx):
                records.append(
                    {
                        "prediction_index": missing_idx,
                        "time_seconds": 0.0,
                        "total_tokens": 0,
                        "aue_tokens": 0,
                        "rtc_tokens": 0,
                        "aue_status": "failure",
                        "rtc_status": "failure",
                        "overall_status": "failure",
                        }
                    )
                seen_prediction_indices.add(missing_idx)

            last_prediction_index = pred_idx
            seen_prediction_indices.add(pred_idx)

            aue_attempts = current["aue_attempts"]
            rtc_attempts = current["rtc_attempts"]
            combined_attempts = current["combined_attempts"]

            if is_ms:
                aue_tokens = sum(a["tokens"] for a in aue_attempts)
                rtc_tokens = sum(a["tokens"] for a in rtc_attempts)
                total_tokens = aue_tokens + rtc_tokens

                def task_status(attempts):
                    if not attempts:
                        return "failure"
                    min_attempt = min(a["attempt"] for a in attempts)
                    return "success_first" if min_attempt == 1 else "success_second"

                aue_status = task_status(aue_attempts)
                rtc_status = task_status(rtc_attempts)

                if aue_status == "success_first" and rtc_status == "success_first":
                    overall = "success_first"
                elif aue_status.startswith("success") and rtc_status.startswith("success"):
                    overall = "success_second"
                else:
                    overall = "failure"
            else:
                # ss
                total_tokens = sum(a["tokens"] for a in combined_attempts)
                if not combined_attempts:
                    overall = "failure"
                else:
                    min_attempt = min(a["attempt"] for a in combined_attempts)
                    overall = "success_first" if min_attempt == 1 else "success_second"

                aue_tokens = total_tokens
                rtc_tokens = 0
                aue_status = overall
                rtc_status = overall

            records.append(
                {
                    "prediction_index": pred_idx,
                    "time_seconds": time_sec,
                    "total_tokens": total_tokens,
                    "aue_tokens": aue_tokens,
                    "rtc_tokens": rtc_tokens,
                    "aue_status": aue_status,
                    "rtc_status": rtc_status,
                    "overall_status": overall,
                    }
                )

            current = {
                "aue_attempts": [],
                "rtc_attempts": [],
                "combined_attempts": [],
                }
            current_task = None

    if current["aue_attempts"] or current["rtc_attempts"] or current["combined_attempts"]:
        missing_idx = last_prediction_index + 1
        records.append(
            {
                "prediction_index": missing_idx,
                "time_seconds": 0.0,
                "total_tokens": 0,
                "aue_tokens": 0,
                "rtc_tokens": 0,
                "aue_status": "failure",
                "rtc_status": "failure",
                "overall_status": "failure",
                }
            )

    return records


def summarize_efficiency(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate efficiency statistics over a list of per-prediction records.
    """
    if not records:
        return {
            "num_predictions": 0,
            "num_success_first": 0,
            "num_success_second": 0,
            "num_failures": 0,
            "total_time_seconds": 0.0,
            "mean_time_seconds": 0.0,
            "total_tokens": 0,
            "mean_tokens": 0.0,
            "tokens_per_second": 0.0,
            }

    num_preds = len(records)
    num_success_first = sum(1 for r in records if r["overall_status"] == "success_first")
    num_success_second = sum(1 for r in records if r["overall_status"] == "success_second")
    num_failures = sum(1 for r in records if r["overall_status"] == "failure")
    total_time = sum(r["time_seconds"] for r in records)
    total_tokens = sum(r["total_tokens"] for r in records)

    return {
        "num_predictions": num_preds,
        "num_success_first": num_success_first,
        "num_success_second": num_success_second,
        "num_failures": num_failures,
        "total_time_seconds": float(total_time),
        "mean_time_seconds": float(total_time / num_preds),
        "total_tokens": int(total_tokens),
        "mean_tokens": float(total_tokens / num_preds),
        "tokens_per_second": float(total_tokens / total_time) if total_time > 0 else 0.0,
        }


def summarize_efficiency_ms(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate efficiency statistics for multi-stage (ms) LLM runs,
    including separate success/failure counts for AUE and RTC.

    Assumes each record has:
      - aue_status: "success_first" | "success_second" | "failure"
      - rtc_status: "success_first" | "success_second" | "failure"
      - overall_status, time_seconds, total_tokens  (as in summarize_efficiency)
    """
    base = summarize_efficiency(records)

    def stage_counts(prefix: str, key: str) -> Dict[str, Any]:
        num_first = sum(1 for r in records if r[key] == "success_first")
        num_second = sum(1 for r in records if r[key] == "success_second")
        num_fail = sum(1 for r in records if r[key] == "failure")
        return {
            f"{prefix}_success_first": num_first,
            f"{prefix}_success_second": num_second,
            f"{prefix}_failures": num_fail,
        }

    result = dict(base)
    if records:
        result.update(stage_counts("aue", "aue_status"))
        result.update(stage_counts("rtc", "rtc_status"))
    else:
        # Explicit zeros if no records
        result.update({
            "aue_success_first": 0,
            "aue_success_second": 0,
            "aue_failures": 0,
            "rtc_success_first": 0,
            "rtc_success_second": 0,
            "rtc_failures": 0,
        })

    return result


# --------- Experiment-level orchestration ---------

def evaluate_experiment(
        gold: Dict[str, Dict[str, Any]],
        experiment: Dict[str, Any],
        ) -> Dict[str, Any]:
    """
    Run the evaluation for a single experiment descriptor produced by
    `discover_experiments`. Also writes unmatched units to a JSON file
    inside the experiment folder for error analysis.
    """
    pred_path = experiment["pred_path"]
    preds = load_predictions(pred_path, gold)
    summary_metrics, unmatched_info, error_counts = evaluate_dataset(gold, preds)

    # Save unmatched units per threshold for this experiment
    unmatched_filename = f"unmatched_units_{experiment['name']}.json"
    unmatched_path = os.path.join(experiment["path"], unmatched_filename)
    with open(unmatched_path, "w", encoding="utf-8") as f:
        json.dump(unmatched_info, f, indent=2, ensure_ascii=False)

    result: Dict[str, Any] = {
        "experiment_name": experiment["name"],
        "experiment_path": experiment["path"],
        "config_path": experiment.get("config_path", ""),
        "pred_path": pred_path,
        "unmatched_units_path": unmatched_path,
        # high-level experimental configuration
        "task": experiment.get("task", ""),
        "model_and_context": experiment.get("model_and_context", ""),
        "seed": experiment.get("seed", ""),
        "timestamp": experiment.get("timestamp", ""),
        "data": experiment.get("data", ""),
        "is_lm": bool(experiment.get("is_lm", False)),
        "lm_variant": experiment.get("lm_variant", ""),
        "lm_file_seed": experiment.get("lm_file_seed", ""),
        "lm_file_model_and_context": experiment.get("lm_file_model_and_context", ""),
        "is_lm_file": bool(experiment.get("is_lm_file", False)),
    }

    # Attach scalar summary metrics
    for key, value in summary_metrics.items():
        result[key] = value

    # Attach aggregated error-analysis counts
    for key, value in error_counts.items():
        result[key] = value

    return result



def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate AUE/RTC experiments and LLM efficiency over prediction JSON files.",
        )
    parser.add_argument(
        "--data_root",
        type=str,
        default="Data",
        help="Folder containing gold CSV files (QT30_test.csv, RIP1.csv, US2016reddit.csv).",
        )
    parser.add_argument(
        "--experiments_root",
        type=str,
        default="experiments",
        help="Root folder containing experiment subfolders.",
        )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_summary.csv",
        help="Where to save the aggregated AUE/RTC experiment metrics.",
        )
    parser.add_argument(
        "--efficiency_all_csv",
        type=str,
        default="llm_efficiency_all.csv",
        help="Where to save the combined efficiency report for all LLM predictions.",
        )
    parser.add_argument(
        "--efficiency_first_csv",
        type=str,
        default="llm_efficiency_first_attempt.csv",
        help="Where to save the efficiency report restricted to 1st-attempt successes.",
        )
    parser.add_argument(
        "--error_csv",
        type=str,
        default="error_analysis.csv",
        help="Where to save the per-experiment graph error analysis counts.",
        )
    args = parser.parse_args()

    data_root = args.data_root
    experiments_root = args.experiments_root

    DATA_FILES = {
        "qt": "QT30_test.csv",
        "rip": "RIP1.csv",
        "reddit": "US2016reddit.csv",
        }

    if not os.path.isdir(data_root):
        raise NotADirectoryError(f"Data root not found: {data_root}")

    if not os.path.isdir(experiments_root):
        raise NotADirectoryError(f"Experiments root not found: {experiments_root}")

    print(f"Discovering experiments under: {experiments_root}")
    experiments = discover_experiments(experiments_root)
    if not experiments:
        print("No experiments found (no folders with suitable *.json/*.config combinations).")
        return

    print(f"Found {len(experiments)} experiment(s).")

    gold_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    all_eval_results: List[Dict[str, Any]] = []
    efficiency_rows_all: List[Dict[str, Any]] = []
    efficiency_rows_first: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    # keys we will pull out of each experiment's evaluation result
    error_metric_keys = [
        "false_positive_units_50",
        "false_negative_units_50",
        "false_positive_units_100",
        "false_negative_units_100",
        "endpoint_mismatched_relations_50",
        "endpoint_mismatched_relations_100",
        "false_positive_pairs_50",
        "false_negative_pairs_50",
        "incorrectly_classified_pairs_50",
        "false_positive_pairs_100",
        "false_negative_pairs_100",
        "incorrectly_classified_pairs_100",
        ]

    for exp in experiments:
        data_key = exp.get("data", "").lower()
        if data_key not in DATA_FILES:
            print(f"\n[SKIP] Experiment {exp['name']} has unknown or missing data type: {data_key!r}")
            continue

        gold_csv_name = DATA_FILES[data_key]
        gold_csv_path = os.path.join(data_root, gold_csv_name)

        if not os.path.exists(gold_csv_path):
            print(f"\n[SKIP] Gold CSV for data={data_key} not found at {gold_csv_path}")
            continue

        if data_key not in gold_cache:
            print(f"\nLoading gold data for '{data_key}' from: {gold_csv_path}")
            gold_cache[data_key] = load_gold(gold_csv_path)

        gold = gold_cache[data_key]

        print(f"\n=== Evaluating experiment: {exp['name']} ===")
        print(f"  Data: {data_key} ({gold_csv_name})")
        print(f"  LM-folder: {bool(exp.get('is_lm', False))}")
        if exp.get("config_path"):
            print(f"  Config: {exp['config_path']}")
        else:
            print("  Config: <none>")
        print(f"  Predictions: {exp['pred_path']}")
        if exp.get("lm_variant"):
            print(f"  LM variant (file): {exp.get('lm_variant')}")

        eval_metrics = evaluate_experiment(gold, exp)

        print("  AUE span @50% F1:  {aue_50_f1:.4f}".format(**eval_metrics))
        print("  AUE span @100% F1: {aue_100_f1:.4f}".format(**eval_metrics))
        print("  RTC full @50% F1:  {rtc_full_50_f1:.4f}".format(**eval_metrics))
        print("  RTC full @100% F1: {rtc_full_100_f1:.4f}".format(**eval_metrics))
        print("  RTC type @50% F1w: {rtc_type_50_f1_weighted:.4f}".format(**eval_metrics))
        print("  RTC type @100% F1w:{rtc_type_100_f1_weighted:.4f}".format(**eval_metrics))
        print("  Unmatched units JSON:", eval_metrics["unmatched_units_path"])

        all_eval_results.append(eval_metrics)

        # Build a compact per-experiment row for error analysis CSV
        error_row = {
            "experiment_name": eval_metrics.get("experiment_name", ""),
            "experiment_path": eval_metrics.get("experiment_path", ""),
            "config_path": eval_metrics.get("config_path", ""),
            "pred_path": eval_metrics.get("pred_path", ""),
            "data": eval_metrics.get("data", ""),
            "task": eval_metrics.get("task", ""),
            "model_and_context": eval_metrics.get("model_and_context", ""),
            "seed": eval_metrics.get("seed", ""),
            "timestamp": eval_metrics.get("timestamp", ""),
            "is_lm": eval_metrics.get("is_lm", ""),
            "lm_variant": eval_metrics.get("lm_variant", ""),
            "lm_file_seed": eval_metrics.get("lm_file_seed", ""),
            "lm_file_model_and_context": eval_metrics.get("lm_file_model_and_context", ""),
            "is_lm_file": eval_metrics.get("is_lm_file", ""),
            }
        for key in error_metric_keys:
            error_row[key] = eval_metrics.get(key, 0)
        error_rows.append(error_row)

        # # --------- LLM efficiency for non-LM experiments ---------
        # if not bool(exp.get("is_lm", False)):
        #     log_path = os.path.join(exp["path"], "experiment.log")
        #     if os.path.exists(log_path):
        #         print(f"  Parsing LLM log at: {log_path}")
        #         records = parse_llm_experiment_log(log_path)

        # --------- LLM efficiency for non-LM experiments ---------
        if not bool(exp.get("is_lm", False)):
            log_path = os.path.join(exp["path"], "experiment.log")
            if os.path.exists(log_path):
                print(f"  Parsing LLM log at: {log_path}")

                # Use experiment 'task' to distinguish ms vs ss behaviour
                task_type = (exp.get("task", "") or "").lower()
                if task_type not in ("ms", "ss"):
                    # Default to ss-style (single stage) if task is unknown
                    task_type = "ss"

                records = parse_llm_experiment_log(log_path, task_type=task_type)

                # --- Summaries: overall vs 1st-attempt-only ---
                if task_type == "ms":
                    # Multi-stage: use AUE/RTC-aware summaries
                    summary_all = summarize_efficiency_ms(records)
                    first_records = [r for r in records if r["overall_status"] == "success_first"]
                    summary_first = summarize_efficiency_ms(first_records)
                else:
                    # Single-shot: keep original behaviour (overall only)
                    summary_all = summarize_efficiency(records)
                    first_records = [r for r in records if r["overall_status"] == "success_first"]
                    summary_first = summarize_efficiency(first_records)

                base_row = {
                    "experiment_name": exp["name"],
                    "experiment_path": exp["path"],
                    "data": data_key,
                    "task": exp.get("task", ""),
                    "model_and_context": exp.get("model_and_context", ""),
                    "seed": exp.get("seed", ""),
                    "timestamp": exp.get("timestamp", ""),
                    }

                row_all = {**base_row, **summary_all}
                row_first = {**base_row, **summary_first}

                efficiency_rows_all.append(row_all)
                efficiency_rows_first.append(row_first)

                # --- Printing ---
                print(
                    "  Efficiency (all): "
                    f"{summary_all['num_predictions']} preds, "
                    f"{summary_all['num_success_first']} first-try, "
                    f"{summary_all['num_success_second']} second-try, "
                    f"{summary_all['num_failures']} failures"
                    )

                if task_type == "ms":
                    # Separate reporting for AUE and RTC
                    print(
                        "    AUE: "
                        f"{summary_all['aue_success_first']} first-try, "
                        f"{summary_all['aue_success_second']} second-try, "
                        f"{summary_all['aue_failures']} failures"
                        )
                    print(
                        "    RTC: "
                        f"{summary_all['rtc_success_first']} first-try, "
                        f"{summary_all['rtc_success_second']} second-try, "
                        f"{summary_all['rtc_failures']} failures"
                        )

                # Time reporting:
                #  - summary_all: averaged over ALL predictions
                #  - summary_first: averaged only over fully 1st-attempt successes
                print(
                    "  Efficiency (1st-only): "
                    f"{summary_first['num_predictions']} preds, "
                    f"mean tokens={summary_first['mean_tokens']:.1f}, "
                    f"mean time={summary_first['mean_time_seconds']:.2f}s"
                    )

    if all_eval_results:
        df_results = pd.DataFrame(all_eval_results)
        df_results.to_csv(args.output_csv, index=False)
        print(f"\nSaved aggregated AUE/RTC results to: {args.output_csv}")
    else:
        print("\nNo experiments were successfully evaluated (AUE/RTC).")

    if efficiency_rows_all:
        df_eff_all = pd.DataFrame(efficiency_rows_all)
        df_eff_all.to_csv(args.efficiency_all_csv, index=False)
        print(f"Saved LLM efficiency report (all predictions) to: {args.efficiency_all_csv}")
    else:
        print("No LLM efficiency data collected (no non-LM logs found?).")

    if efficiency_rows_first:
        df_eff_first = pd.DataFrame(efficiency_rows_first)
        df_eff_first.to_csv(args.efficiency_first_csv, index=False)
        print(
            "Saved LLM efficiency report (1st-attempt successes only) "
            f"to: {args.efficiency_first_csv}"
            )
    else:
        print("No 1st-attempt-only efficiency data collected.")

    if error_rows:
        df_error = pd.DataFrame(error_rows)
        df_error.to_csv(args.error_csv, index=False)
        print(f"Saved graph error analysis report to: {args.error_csv}")
    else:
        print("No graph error analysis data collected.")


if __name__ == "__main__":
    main()
