import sys
import json
import logging
import time

import pandas as pd
import torch
import gc

from LLM_argument_schema import ArgumentGraph
from system_utilities import parse_args, setup_experiment_dir, load_config
from LLM_utilities import generate_fn, load_outlines_llm


def main():
    """
    Main function for LLM-S-Inference.
    :return:
    """
    args = parse_args()
    config = load_config(args)
    run_dir = setup_experiment_dir(config)

    logging.info(f"Loaded config: {config}")

    # ARGUMENT GRAPH EXAMPLES
    # Load few-shot examples for graphs
    str_few_shot_examples = []
    if config["inference"]["num_context_examples"] > 0:
        with open(config["inference"]["context_examples_path"], "r", encoding="utf-8") as f:
            str_examples = json.load(f)

            # Convert your format -> expected format
            converted = []
            for ex in str_examples:
                converted.append({
                    "input": ex["conversation_text"],
                    "output": json.dumps(ex["argument_objects"], ensure_ascii=False)
                    }
                    )
            str_few_shot_examples = converted[:config["inference"]["num_context_examples"]]
            logging.info(f"Loaded {len(str_few_shot_examples)} structure examples for in-context learning")

    # load argument unit extraction and relation classification models.
    target_model_name = config['model']['model_name_or_path']
    print(f"--- INITIALIZING ---")
    print(f"Mode: {'Default LLM' if config['experiment']['do_inference'] else 'fine-tuning the model'}")
    print(f"Model: {target_model_name}")

    try:
        str_model, str_tokenizer = load_outlines_llm(target_model_name, config)
        logging.info(f"Loaded model: {target_model_name}")
    except Exception as e:
        print(f"FATAL ERROR: Could not load model. Details: {e}")
        logging.error(f"FATAL ERROR: Could not load model. Details: {e}")
        sys.exit(1)

    # Define General Instructions
    instructions_for_str = """
You are an argument mining expert. From the discussion, extract argument units and their relations.

Return ONE JSON object with this exact shape:

{
  "argument_units": [
    {"reason": "...", "id": 0, "text": "..."},
    ...
  ],
    "relations": [
    {"source_id": 1, "target_id": 0, "type": "support"},
    ...
  ]
}

Argument units:
- `text`: copy **verbatim** from the discussion (no paraphrasing).
- Each unit is a single argumentative idea (example: asserts / questions / rejects/ accepts / defends / challenges topic or another statement).
- Assign IDs in order of appearance: 0, 1, 2...
- `reason`: short explanation of the unit’s intent or expressed idea.

Relations:
- Only create a relation if there is a clear indication in the conversation text.
- Only treat a pair as “no relation” if they are clearly unrelated or just share a topic
  without one supporting or attacking the other.
- `support`: source accepts or gives reasons/evidence/clarification for target.
- `attack`: source challenges, rejects, undercuts, or undermines target.
- Only create a relation if there is a **clear, explicit** link (e.g. “because”, “but”, “however”, “in response to”).
- If unsure or only loosely related by topic, do **not** create a relation.
- Use ONLY IDs from `argument_units`; NEVER invent new IDs.
- By default, the source should appear **after** the target in the discussion.
- Do NOT create symmetric duplicates.

Schema constraints:
- Root object MUST have ONLY: "argument_units" and "relations".
- "argument_units" and "relations" MUST each be lists.
- Elements of "argument_units" MUST have ONLY: "reason", "id", "text".
- Elements of "relations" MUST have ONLY: "source_id", "target_id", "type".
- `type` MUST be exactly "support" or "attack".
- Do NOT output any text before or after the JSON.
"""

    # START INFERENCE
    output_graphs = {}

    if not config['inference']['data_path']:
        raise ValueError("Inference data_path is empty in config.")

    # Load inference file (csv)
    inference_file = pd.read_csv(config['inference']['data_path'])
    logging.info(f"Inference file loaded with length: {len(inference_file)}")

    # Inside main() loop:
    total_run_metrics = {"prompt": 0, "completion": 0}

    for i, disc_row in inference_file.iterrows():
        start_time = time.time()
        conv_id = disc_row.get("conversation_id", f"conv_{i}")
        logging.info(f"--- [Conversation {conv_id}] START ---")

        # Start Structure Prediction
        input_text_for_str = disc_row["conversation_text"]

        output_str, token_metrics = generate_fn(input_text_for_str, str_model,
                                                tokenizer=str_tokenizer,
                                                output_type=ArgumentGraph,
                                                instructions=instructions_for_str,
                                                selected_examples=str_few_shot_examples,
                                                config=config,
                                                current_task="ASP"
                                                )

        if not output_str:
            output_graphs[conv_id] = None
            print("No structure predicted.")
            logging.info(f"No structure predicted. Proceed to next discussion.")
            continue

        output_graphs[conv_id] = output_str.model_dump()

        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Prediction {i} took {round(time_taken, 4)} seconds")
        logging.info(f"Prediction {i} took {round(time_taken, 4)} seconds")
        # Accumulate totals
        total_run_metrics["prompt"] += token_metrics["prompt_tokens"]
        total_run_metrics["completion"] += token_metrics["completion_tokens"]

        torch.cuda.empty_cache()
        gc.collect()

    # save output_graphs
    output_path_name = run_dir / f"{config['experiment']['run_name']}.json"

    with open(output_path_name, "w") as f:
        json.dump(output_graphs, f, indent=4)

    logging.info(f"Output graphs saved to {output_path_name}")
    total_tokens = total_run_metrics["prompt"] + total_run_metrics["completion"]
    logging.info(f"FINAL RUN STATS: Prompt: {total_run_metrics['prompt']}, "
                 f"Completion: {total_run_metrics['completion']}, Total: {total_tokens}"
                 )

    # save updated config file in run folder
    config_file_path = run_dir / f"{config['experiment']['run_name']}.config"

    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=2)

    return None


if __name__ == '__main__':
    main()
