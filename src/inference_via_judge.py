# LM_ms_with_LLM_judge_inference.py

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import pandas as pd
from transformers import AutoTokenizer

from system_utilities import parse_args, load_config, setup_experiment_dir
from LM_ms_finetuning import MaskedLMClassifier
from LM_ms_inference import (
    run_aue_for_conversation,
    build_rtc_inputs_from_units,
    run_rtc_for_pairs,
)
from LLM_utilities import load_outlines_llm, generate_fn
from LLM_argument_schema import ArgumentGraph


# -----------------------------
# LLM Judge prompt helpers
# -----------------------------

def build_judge_instructions_ms() -> str:
    """
    Multi-step judge instructions: remove incorrect units/relations only.
    The candidate units were produced by AUE (token classification) and
    relations by RTC (pair classification).
    """
    return """
You are an argument mining expert acting as a strict **judge**.

You are given:
1) A discussion text.
2) A candidate argument graph produced by two fine-tuned models:
   - AUE model extracted argument units
   - RTC model predicted support/attack relations between units

Your task:
- Remove any argument units that are not clearly supported by the text.
- Remove any relations that are not clearly supported by the text.
- You MUST NOT add new units.
- You MUST NOT add new relations.
- You MUST NOT edit the `text` field of any unit.
- You MUST keep the same unit IDs for any units you keep.
- If you remove a unit, remove any relation that references it.

Keep an argument unit only if:
- Its `text` appears verbatim in the discussion (minor whitespace/punctuation differences are okay).
- It expresses a clear argumentative claim/premise/rebuttal (not just a topic label or greeting).

Keep a relation only if:
- The source clearly supports or attacks the target in context.
- The relationship is explicit or strongly implied (e.g., “because”, “but”, “however”, direct rebuttal).

Output:
Return exactly ONE JSON object matching this schema:

{
  "argument_units": [
    {"reason": "...", "id": 0, "text": "..."},
    ...
  ],
  "relations": [
    {"source_id": 3, "target_id": 1, "type": "support"},
    ...
  ]
}

Additional constraints:
- Ensure all relation IDs exist in the remaining units.
- Relation type MUST be exactly "support" or "attack".
- Do NOT output any text before or after the JSON.
"""


def build_judge_input_text(
    conversation_text: str,
    candidate_units: List[Dict[str, Any]],
    candidate_relations: List[Dict[str, Any]],
) -> str:
    """
    Build the judge "user" input text. We pass discussion + candidate graph.
    """
    candidate_graph = {
        "argument_units": candidate_units,
        "relations": candidate_relations,
    }
    candidate_json = json.dumps(candidate_graph, ensure_ascii=False, indent=2)

    return (
        "DISCUSSION:\n"
        f"{conversation_text}\n\n"
        "CANDIDATE ARGUMENT GRAPH (from multi-step LM AUE→RTC):\n"
        f"{candidate_json}\n\n"
        "TASK:\n"
        "Review the candidate graph and remove any incorrect units or relations.\n"
        "Return ONLY the cleaned JSON graph.\n"
    )


def _units_for_judge_schema(units_ms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert multi-step LM units ({id,start,end,text}) into the schema required by ArgumentGraph:
    ({reason,id,text}). We keep IDs and text unchanged.
    """
    out = []
    for u in units_ms:
        out.append(
            {
                "reason": "Candidate argumentative unit extracted by AUE model.",
                "id": int(u["id"]),
                "text": u["text"],
            }
        )
    return out


def _relations_for_judge_schema(relations_ms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    RTC already uses {source_id,target_id,type} and type is support/attack/non.
    We will only pass support/attack to the judge (non already filtered in LM_ms_inference).
    """
    out = []
    for r in relations_ms:
        if r.get("type") not in ("support", "attack"):
            continue
        out.append(
            {
                "source_id": int(r["source_id"]),
                "target_id": int(r["target_id"]),
                "type": r["type"],
            }
        )
    return out


def _apply_judge_selection_back_to_ms_units(
    original_ms_units: List[Dict[str, Any]],
    judge_units: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Judge outputs units without spans. We project the keep/remove decision back
    onto original units by ID, so we preserve start/end fields.
    """
    keep_ids = {int(u["id"]) for u in judge_units}
    kept = [u for u in original_ms_units if int(u["id"]) in keep_ids]
    return kept


# -----------------------------
# Multi-step LM inference
# -----------------------------

def run_multistep_lm_for_df(
    df: pd.DataFrame,
    aue_model: MaskedLMClassifier,
    aue_tokenizer,
    aue_max_length: int,
    rtc_model: MaskedLMClassifier,
    rtc_tokenizer,
    rtc_max_length: int,
    device: torch.device,
    window_size: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs AUE then RTC for each row in the dataframe.
    Expects:
      - conversation_text column name can be either "text" or "conversation_text"
      - conversation_id optional
    Returns dict keyed by conv_id (string).
    """
    outputs: Dict[str, Dict[str, Any]] = {}

    for idx, row in df.iterrows():
        conv_id = row.get("conversation_id", idx)
        conv_id = str(int(conv_id)) if isinstance(conv_id, (int, float)) else str(conv_id)

        # accept either column name
        if "conversation_text" in row:
            text = str(row["conversation_text"])
        else:
            text = str(row["text"])

        # 1) AUE → units (includes your validity filtering + chronological IDs)
        units = run_aue_for_conversation(
            model=aue_model,
            tokenizer=aue_tokenizer,
            text=text,
            max_length=aue_max_length,
            device=device,
        )

        # 2) RTC inputs from units (directionality preserved inside build_rtc_inputs_from_units)
        examples, pairs = build_rtc_inputs_from_units(
            full_text=text,
            units=units,
            tokenizer=rtc_tokenizer,
            max_length=rtc_max_length,
            window_size=window_size,
        )

        # 3) RTC → relations
        relations = run_rtc_for_pairs(
            model=rtc_model,
            tokenizer=rtc_tokenizer,
            examples=examples,
            pairs=pairs,
            max_length=rtc_max_length,
            device=device,
        )

        outputs[conv_id] = {
            "conversation_text": text,
            "argument_units": units,      # includes start/end/text/id
            "relations": relations,       # {source_id,target_id,type}
        }

    return outputs


# -----------------------------
# Main
# -----------------------------

def main():
    """
    Multi-step LM (AUE + RTC) + LLM judge inference.

    Minimal config expectations:

    experiment:
      run_name: "ms_lm_llm_judge"
      log_level: "INFO"

    inference:
      data_path: "path/to/inference.csv"
      window_size: 50

    AUE:
      model_name_or_path: "..."
      num_labels: <int>
      max_length: <int>

    RTC:
      model_name_or_path: "..."
      num_labels: <int>
      max_length: <int>

    lm_ms:
      run_dir: "path/to/run_dir_with_best_model_AUE.pt_and_best_model_RTC.pt"

    model:
      model_name_or_path: "unsloth/gpt-oss-20b-unsloth-bnb-4bit"   # judge LLM

    Output:
      <run_dir>/<run_name>_ms_lm_llm_judge.json
    """
    args = parse_args()
    config = load_config(args)

    # your existing helper; makes a run folder and config-driven logging
    run_dir = setup_experiment_dir(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # -----------------------------
    # Load inference CSV
    # -----------------------------
    data_path = config.get("inference", {}).get("data_path")
    if not data_path:
        raise ValueError("Missing config['inference']['data_path'].")

    df = pd.read_csv(data_path)
    logging.info(f"Loaded inference data: {data_path} (rows={len(df)})")

    window_size = int(config.get("inference", {}).get("window_size", 50))

    # -----------------------------
    # Load AUE model/tokenizer
    # -----------------------------
    aue_cfg = config["AUE"]
    aue_tokenizer = AutoTokenizer.from_pretrained(aue_cfg["model_name_or_path"])
    aue_model = MaskedLMClassifier(
        aue_cfg["model_name_or_path"],
        num_labels=int(aue_cfg["num_labels"]),
        task_type="AUE",
    )

    # weights location: run_dir containing best_model_AUE.pt / best_model_RTC.pt
    ms_run_dir = config.get("lm_ms", {}).get("run_dir")
    if not ms_run_dir:
        raise ValueError("Missing config['lm_ms']['run_dir'] (folder containing best_model_AUE.pt and best_model_RTC.pt).")
    ms_run_dir = Path(ms_run_dir)

    aue_best_path = ms_run_dir / "best_model_AUE.pt"
    if not aue_best_path.exists():
        raise FileNotFoundError(f"Could not find AUE weights at: {aue_best_path}")

    aue_model.load_state_dict(torch.load(aue_best_path, map_location=device))
    aue_model.to(device)
    aue_model.eval()

    # -----------------------------
    # Load RTC model/tokenizer
    # -----------------------------
    rtc_cfg = config["RTC"]
    rtc_tokenizer = AutoTokenizer.from_pretrained(rtc_cfg["model_name_or_path"])
    rtc_model = MaskedLMClassifier(
        rtc_cfg["model_name_or_path"],
        num_labels=int(rtc_cfg["num_labels"]),
        task_type="RTC",
    )

    rtc_best_path = ms_run_dir / "best_model_RTC.pt"
    if not rtc_best_path.exists():
        raise FileNotFoundError(f"Could not find RTC weights at: {rtc_best_path}")

    rtc_model.load_state_dict(torch.load(rtc_best_path, map_location=device))
    rtc_model.to(device)
    rtc_model.eval()

    # -----------------------------
    # Stage 1: multi-step LM inference
    # -----------------------------
    logging.info("=== STAGE 1: Running multi-step LM (AUE→RTC) ===")
    lm_graphs = run_multistep_lm_for_df(
        df=df,
        aue_model=aue_model,
        aue_tokenizer=aue_tokenizer,
        aue_max_length=int(aue_cfg["max_length"]),
        rtc_model=rtc_model,
        rtc_tokenizer=rtc_tokenizer,
        rtc_max_length=int(rtc_cfg["max_length"]),
        device=device,
        window_size=window_size,
    )
    logging.info(f"Multi-step LM produced graphs for {len(lm_graphs)} conversations.")

    # -----------------------------
    # Stage 2: LLM-as-a-judge
    # -----------------------------
    logging.info("=== STAGE 2: Loading judge LLM ===")
    judge_model_name = config["model"]["model_name_or_path"]
    judge_generator, judge_tokenizer = load_outlines_llm(judge_model_name, config)
    logging.info(f"Loaded judge model: {judge_model_name}")

    judge_instructions = build_judge_instructions_ms()
    judge_examples: List[Dict[str, Any]] = []

    total_run_metrics = {"prompt": 0, "completion": 0, "total": 0}
    final_out: Dict[str, Dict[str, Any]] = {}

    for conv_id, g in lm_graphs.items():
        text = g["conversation_text"]
        ms_units = g.get("argument_units", [])
        ms_relations = g.get("relations", [])

        # If the LM output is too small, skip judge (ArgumentGraph schema needs >=2 units and >=1 relation)
        if len(ms_units) < 2 or len(ms_relations) < 1:
            final_out[conv_id] = {
                "conversation_text": text,
                "ms_argument_units": ms_units,
                "ms_relations": ms_relations,
                "judge_argument_units": ms_units,
                "judge_relations": ms_relations,
                "judge_note": "Skipped judge due to insufficient LM output (needs >=2 units and >=1 relation).",
            }
            continue

        cand_units_for_judge = _units_for_judge_schema(ms_units)
        cand_rel_for_judge = _relations_for_judge_schema(ms_relations)

        # If RTC predicted zero support/attack (all non filtered out), judge schema would fail; skip
        if len(cand_rel_for_judge) < 1:
            final_out[conv_id] = {
                "conversation_text": text,
                "ms_argument_units": ms_units,
                "ms_relations": ms_relations,
                "judge_argument_units": ms_units,
                "judge_relations": ms_relations,
                "judge_note": "Skipped judge because RTC produced no support/attack relations.",
            }
            continue

        judge_input = build_judge_input_text(text, cand_units_for_judge, cand_rel_for_judge)

        judge_output, token_metrics = generate_fn(
            input_text=judge_input,
            generator=judge_generator,
            tokenizer=judge_tokenizer,
            output_type=ArgumentGraph,
            instructions=judge_instructions,
            selected_examples=judge_examples,
            config=config,
            current_task="ASP",
        )

        if not judge_output:
            # fallback to MS output
            final_out[conv_id] = {
                "conversation_text": text,
                "ms_argument_units": ms_units,
                "ms_relations": ms_relations,
                "judge_argument_units": ms_units,
                "judge_relations": ms_relations,
                "judge_note": "Judge failed; fell back to MS output.",
            }
            continue

        judged = judge_output.model_dump()
        judge_units = judged.get("argument_units", [])
        judge_rels = judged.get("relations", [])

        # project back to ms units to preserve spans (start/end)
        judged_ms_units = _apply_judge_selection_back_to_ms_units(ms_units, judge_units)

        # relations are already in judge format; keep as-is
        judged_ms_relations = judge_rels

        final_out[conv_id] = {
            "conversation_text": text,
            "ms_argument_units": ms_units,
            "ms_relations": ms_relations,
            "judge_argument_units": judged_ms_units,
            "judge_relations": judged_ms_relations,
        }

        total_run_metrics["prompt"] += token_metrics.get("prompt_tokens", 0)
        total_run_metrics["completion"] += token_metrics.get("completion_tokens", 0)
        total_run_metrics["total"] += token_metrics.get("total_tokens", 0)

    out_path = run_dir / f"{config['experiment']['run_name']}_ms_lm_llm_judge.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_out, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved MS+Judge predictions to: {out_path}")
    logging.info(
        "Judge token usage: "
        f"prompt={total_run_metrics['prompt']} "
        f"completion={total_run_metrics['completion']} "
        f"total={total_run_metrics['total']}"
    )
    print(f"Saved MS+Judge predictions to: {out_path}")


if __name__ == "__main__":
    main()
