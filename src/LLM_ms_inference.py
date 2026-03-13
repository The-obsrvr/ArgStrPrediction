import sys
import json
import logging
import time
import os

import torch
import pandas as pd

# --- Disable TorchDynamo / torch.compile globally to avoid Unsloth Gemma3 graph breaks ---
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
    dynamo.disable()
except Exception:
    # If running on an older PyTorch without torch._dynamo, just ignore.
    pass

from LLM_argument_schema import ArgumentUnits, ArgumentRelations
from system_utilities import parse_args, setup_experiment_dir, load_config
from LLM_utilities import (filter_and_sort_argument_units, generate_fn, build_ms_fewshot_examples_from_single_file,
                           load_outlines_llm
                           )


def main():
    """
    Main function for LLM-M-Inference.
    :return:
    """
    args = parse_args()
    config = load_config(args)
    run_dir = setup_experiment_dir(config)

    logging.info(f"Loaded config: {config}")

    # Load few-shot examples for units and for relations
    args_few_shot_examples = []
    rels_few_shot_examples = []

    if config['inference']['num_context_examples'] > 0:
        # Prefer args_context_examples_path; if rels path is different you can still use it,
        # but user wants single file, so we default to args path.
        with open(config["inference"]["context_examples_path"], "r", encoding="utf-8") as f:
            str_examples = json.load(f)

        args_few_shot_examples, rels_few_shot_examples = build_ms_fewshot_examples_from_single_file(
            raw_examples=str_examples,
            num_examples=config['inference']['num_context_examples'],
            rtc_input_style="compact"
            )

        logging.info(f"Loaded {len(args_few_shot_examples)} AUE few-shot examples")
        logging.info(f"Loaded {len(rels_few_shot_examples)} RTC few-shot examples")
    else:
        logging.warning("num_context_examples > 0 but no context_examples_path provided; running zero-shot.")

    # load argument unit extraction and relation classification models.
    args_target_model_name = config['model']['args_model_name_or_path']
    rels_target_model_name = config['model']['rels_model_name_or_path']

    print(f"--- INITIALIZING ---")
    print(f"Mode: {'Default LLM' if config['experiment']['do_inference'] else 'fine-tuning the model'}")
    print(f"AUE Model:  {args_target_model_name}")
    print(f"RTC Model:  {rels_target_model_name}")

    try:
        same_model = (args_target_model_name == rels_target_model_name)

        # Load AUE model (Unsloth if needed)
        args_model, args_tokenizer = load_outlines_llm(args_target_model_name, config)
        logging.info("AUE Model loaded successfully.")

        # If same model used for relations, just reuse
        if same_model:
            rels_model, rels_tokenizer = args_model, args_tokenizer
            logging.info("Same Model for RTC.")
        else:
            rels_model, rels_tokenizer = load_outlines_llm(rels_target_model_name, config)
            print("RTC Model loaded successfully.")
            logging.info("RTC Model loaded successfully.")

    except Exception as e:
        print(f"FATAL ERROR: Could not load model. Details: {e}")
        logging.error(f"FATAL ERROR: Could not load one of the models. Details: {e}")
        sys.exit(1)

    # Define General Instructions
    instructions_for_args = """
You are an argument mining expert. Extract **argument units** from the discussion.

Return ONE JSON object with this exact shape:

{
  "argument_units": [
    {"reason": "...", "id": 0, "text": "..."},
    ...
  ]
}

Rules:
- Copy `text` **verbatim** from the discussion (no paraphrasing).
- Each unit MUST be a single argumentative idea (example: asserts / questions / rejects/ accepts / defends / challenges topic or another statement).
- Assign IDs in order of appearance: 0, 1, 2,...
- `reason` is a short explanation of the unit’s intent/role (e.g. claim/premise/opinion/stance/counterclaim).
- Exclude non-argumentative chatter, jokes, greetings, or pure rhetoric.

Schema constraints:
- The root object MUST have ONLY the key `"argument_units"`.
- Each conversation should RETURN at least 3 units.
- Each element MUST have ONLY `"id"`, `"text"`, `"reason"`.
- Do NOT output any text before or after the JSON.
"""

    instructions_for_rels = """
You are an argument mining expert. Identify **directional relations** between the given units.

Return ONE JSON object with this exact shape:

{
  "relations": [
    {"source_id": , "target_id": , "type": },
    ...
  ] 
}

Meaning:
- `support`: source gives reasons/evidence/clarification for target.
- `attack`: source challenges, rejects, undercuts or undermines target.

Rules to AVOID spurious relations:
- Only create a relation if there is a clear indication in the text.
- Only treat a pair as “no relation” if they are clearly unrelated or just share a topic
  without one supporting or attacking the other.
- Only create a relation if the text shows a **clear, explicit** link to its target unit.
- If unsure or only loosely related by topic, **do NOT** create a relation.
- ENSURE you consider the conversation text for added context and not only units.

Constraints:
- Use ONLY IDs from the provided units; never invent new IDs.
- By default, the source should appear **after** the target in the discussion.
- `type` MUST be exactly `"support"` or `"attack"` (no other labels).
- The root object MUST have ONLY the key `"relations"`.
- `"relations"` MUST be a list.
    """

    # START INFERENCE
    output_graphs = {}

    # Load inference file (csv)
    inference_file = pd.read_csv(config['inference']['data_path'])
    logging.info(f"Inference file loaded with length: {len(inference_file)}")

    total_run_metrics = {"prompt": 0, "completion": 0, "rels_prompt": 0, "rels_completion": 0}

    for i, disc_row in inference_file.iterrows():
        start_time = time.time()
        conv_id = disc_row.get("conversation_id", f"conv_{i}")
        logging.info(f"--- [Conversation {conv_id}] START ---")

        # Start Unit Extraction
        input_text_for_args = disc_row["conversation_text"]

        output_args, args_token_metrics = generate_fn(input_text_for_args, args_model,
                                                      tokenizer=args_tokenizer,
                                                      output_type=ArgumentUnits,
                                                      instructions=instructions_for_args,
                                                      selected_examples=args_few_shot_examples,
                                                      config=config,
                                                      current_task="AUE"
                                                      )

        if not output_args:
            # if no argument units were successfully extracted, then skip relation extraction and proceed to next
            # discussion.
            output_graphs[conv_id] = None
            print("No arg units extracted")
            logging.info(f"No arg units extracted. Proceed to next discussion.")
            # FAILED: go to the next discussion
            continue

        # START Relation Identification

        # --- 1b) Filter malformed units and enforce chronological order ---
        try:
            filtered_args = filter_and_sort_argument_units(output_args, input_text_for_args)
        except Exception as e:
            logging.error(f"Error while filtering/sorting units: {e}")
            output_graphs[conv_id] = None
            continue

        # If, after filtering, we don't have enough units, skip RTC
        if len(filtered_args.argument_units) < 2:
            logging.info("Too few valid units after filtering; skipping RTC for this discussion.")
            output_graphs[conv_id] = None
            continue

        import gc
        torch.cuda.empty_cache()
        gc.collect()

        # Prepare a compact JSON representation for the RTC prompt
        # units_for_prompt = [
        #     {"id": u.id, "text": u.text, "reason": u.reason}
        #     for u in filtered_args.argument_units
        #     ]
        # units_json_str = json.dumps(units_for_prompt, ensure_ascii=False, separators=(",", ":"))

        args_dict = filtered_args.model_dump()
        units_for_rtc = args_dict.get("argument_units", args_dict.get("units", []))

        # --- 2) Relation Identification (RTC) ---
        # Directionality is handled in instructions_for_rels:
        # Source should generally appear after Target in the text.
        input_text_for_rels = (
            "Discussion Text:\n"
            f"{input_text_for_args}\n\n"
            "Extracted Argument Units (JSON):\n"
            f"{units_for_rtc}\n\n"
            "Identified Relations:"
        )

        #  Start Relation Extraction
        output_rels, rels_token_metrics = generate_fn(input_text_for_rels, rels_model,
                                                      tokenizer=rels_tokenizer,
                                                      output_type=ArgumentRelations,
                                                      instructions=instructions_for_rels,
                                                      selected_examples=rels_few_shot_examples,
                                                      config=config,
                                                      current_task="RTC"
                                                      )

        if not output_rels:
            output_graphs[conv_id] = None
            print("No rels units extracted")
            logging.info(f"No rels units extracted. Proceed to next discussion.")
            # FAILED: go to the next discussion
            continue

        # Convert Pydantic models to plain dicts
        units_dict = filtered_args.model_dump()
        rels_dict = output_rels.model_dump()

        argument_units = units_dict.get("argument_units", [])
        relations = rels_dict.get("relations", [])

        final_graph = {
            "argument_units": argument_units,
            "relations": relations,
            }

        # SUCCESS: save the complete graph
        output_graphs[conv_id] = final_graph

        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Prediction {i} took {round(time_taken, 4)} seconds")
        logging.info(f"Prediction {i} took {round(time_taken, 4)} seconds")
        # Accumulate totals
        total_run_metrics["prompt"] += args_token_metrics["prompt_tokens"]
        total_run_metrics["completion"] += args_token_metrics["completion_tokens"]
        total_run_metrics["rels_prompt"] += rels_token_metrics["prompt_tokens"]
        total_run_metrics["rels_completion"] += rels_token_metrics["completion_tokens"]

    # save output_graphs
    output_path_name = run_dir / f"{config['experiment']['run_name']}.json"

    with open(output_path_name, "w") as f:
        json.dump(output_graphs, f, indent=4)

    logging.info(f"Output graphs saved to {output_path_name}")
    total_tokens = total_run_metrics["prompt"] + total_run_metrics["completion"] + total_run_metrics["rels_prompt"] + \
                   total_run_metrics["rels_completion"]

    logging.info(f"FINAL RUN STATS: Prompt: {total_run_metrics['prompt']}, "
                 f"Completion: {total_run_metrics['completion']}, "
                 f"Rels Prompt: {total_run_metrics['rels_prompt']}, "
                 f"Rels Completion: {total_run_metrics['rels_completion']}, "
                 f" Total (collective): {total_tokens}"
                 )

    # save updated config file in run folder
    config_file_path = run_dir / f"{config['experiment']['run_name']}.config"

    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=2)

    return None


if __name__ == '__main__':
    main()
