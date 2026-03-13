import logging

from system_utilities import count_tokens
import re
import json
from typing import Any, Dict, List, Optional, Literal, Tuple
from pathlib import Path

from pydantic import ValidationError
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
import torch

from LLM_argument_schema import ArgumentUnits, ArgumentUnit, ArgumentRelations

# Extend/adjust as needed
ModelFamily = Literal["gpt_oss", "deepseek_r1", "gemma3", "generic_chat"]
HF_ACCESS_TOKEN = "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"


def _schema_units_key() -> str:
    """
    Return the field name ArgumentUnits expects ('units' vs 'argument_units').
    This prevents schema/instruction drift from breaking your few-shot shaping.
    """
    try:
        fields = getattr(ArgumentUnits, "model_fields", {})
        if "argument_units" in fields:
            return "argument_units"
        if "units" in fields:
            return "units"
    except Exception:
        pass
    # default fallback (your context examples file uses this)
    return "argument_units"


def _schema_relations_key() -> str:
    try:
        fields = getattr(ArgumentRelations, "model_fields", {})
        if "relations" in fields:
            return "relations"
    except Exception:
        pass
    return "relations"


def build_ms_fewshot_examples_from_single_file(
        raw_examples: List[Dict[str, Any]],
        num_examples: int,
        rtc_input_style: str = "compact",
        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build two few-shot pools for multi-prompt inference *from the same source list*.

    Source example format (like qt30_context_examples.json):
      {
        "conversation_id": "...",
        "conversation_text": "...",
        "argument_objects": {
          "argument_units": [...],
          "relations": [...]
        }
      }

    Returns:
      (aue_examples, rtc_examples)
      Each item is {"input": <string>, "output": <dict|list|string>}
      compatible with generate_fn() / build_prompt().

    rtc_input_style:
      - "compact": just JSON of {"conversation_text":..., "argument_units":[...]}
      - "promptlike": matches your actual RTC prompt template (Discussion Text + Units JSON)
    """
    num = max(0, int(num_examples or 0))
    if num == 0:
        return [], []

    units_key = _schema_units_key()
    rels_key = _schema_relations_key()

    aue_examples: List[Dict[str, Any]] = []
    rtc_examples: List[Dict[str, Any]] = []

    for ex in raw_examples[:num]:
        conv_text = ex.get("conversation_text", "") or ""
        arg_objs = ex.get("argument_objects", {}) or {}

        # raw file typically stores these keys
        raw_units = arg_objs.get("argument_units", []) or arg_objs.get("units", []) or []
        raw_rels = arg_objs.get("relations", []) or []

        # ---- AUE few-shot ----
        # Input: conversation text (string)
        # Output: only units object (dict) matching schema field name
        aue_examples.append(
            {
                "input": conv_text,
                "output": {units_key: raw_units},
                }
            )

        # ---- RTC few-shot ----
        # Input: conversation + units
        if rtc_input_style == "promptlike":
            units_for_prompt = [
                {"id": u.get("id"), "text": u.get("text")}
                for u in raw_units
                ]
            units_json_str = json.dumps(units_for_prompt, ensure_ascii=False, indent=2)
            rtc_input = (
                "Conversation Text:\n"
                f"{conv_text}\n\n"
                "Extracted Argument Units (JSON):\n"
                f"{units_json_str}\n\n"
                "Identified Relations:"
            )
        else:
            rtc_input = json.dumps(
                {"conversation_text": conv_text, units_key: raw_units},
                ensure_ascii=False,
                )

        rtc_examples.append(
            {
                "input": rtc_input,
                "output": {rels_key: raw_rels},
                }
            )

    return aue_examples, rtc_examples


def _require_example_fields(ex: Dict[str, Any]) -> None:
    """Fail early if the few-shot example doesn't match the builder's expectation."""
    if "input" not in ex or "output" not in ex:
        raise KeyError("Each example must have keys: 'input' and 'output'.")


def _normalize_output_to_text(out: Any) -> str:
    """Ensure assistant example output is a string (JSON text is best)."""
    if isinstance(out, (dict, list)):
        return json.dumps(out, ensure_ascii=False)
    return str(out)


def _normalize_examples(selected_examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    norm_examples: List[Dict[str, str]] = []
    for ex in selected_examples:
        _require_example_fields(ex)
        norm_examples.append({
            "input": str(ex["input"]),
            "output": _normalize_output_to_text(ex["output"]),
            }
            )
    return norm_examples


def _detect_model_family_from_name(model_name_or_path: str) -> ModelFamily:
    """
    Best-effort detection from model name/path.
    Falls back to "generic_chat".
    """
    name = (model_name_or_path or "").lower()
    if "gpt-oss" in name or "gpt_oss" in name or ("gpt" in name and "oss" in name):
        return "gpt_oss"
    if "deepseek" in name and ("r1" in name or "deepseek-r1" in name):
        return "deepseek_r1"
    if "gemma" in name and ("3" in name or "gemma3" in name or "gemma-3" in name):
        return "gemma3"
    return "generic_chat"


def build_prompt(
        model_family: ModelFamily,
        input_text: str,
        instructions: str,
        selected_examples: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        repeat_instructions: bool = True,
        tokenizer: Optional[Any] = None,
        ) -> str:
    """
    Unified prompt builder for:
      - GPT Harmony-style tokens: <|system|> <|user|> <|assistant|>
      - DeepSeek chat tokens: <｜begin▁of▁sentence｜> <｜User｜> <｜Assistant｜> <｜end▁of▁sentence｜>
      - Gemma 3 chat tokens: <bos> <start_of_turn>user ... <end_of_turn> etc.
      - Generic fallback.

    If repeat_instructions=False:
      - instructions are included ONCE (system/preface)
      - each example user turn contains only "Input:\n{...}"
      - final user turn contains only "Input:\n{...}"
    """

    instructions = instructions.strip()
    input_text = input_text.strip()

    # ---------- Helpers ----------
    def _build_system_content() -> str:
        chunks: List[str] = []
        if system_message:
            chunks.append(system_message.strip())
        # If we are NOT repeating instructions in every user turn, put them here once.
        if not repeat_instructions and instructions:
            chunks.append(instructions.strip())
        return "\n\n".join(chunks).strip()

    system_content = _build_system_content()

    # Validate + normalize examples
    norm_examples: List[Dict[str, str]] = []
    for ex in selected_examples:
        _require_example_fields(ex)
        norm_examples.append({
            "input": str(ex["input"]).strip(),
            "output": _normalize_output_to_text(ex["output"]).strip(),
            }
            )

    # Helper: what goes in each user turn
    def _user_turn_text(x: str) -> str:
        if repeat_instructions:
            return f"{instructions}\n\nInput:\n{x}\n"
        return f"Input:\n{x}\n"

    # Helper: what goes once at the top if not repeating
    def _top_preface() -> str:
        chunks = []
        if system_message:
            chunks.append(system_message.strip())
        if not repeat_instructions:
            chunks.append(instructions)
        return ("\n\n".join(chunks).strip() + "\n\n") if chunks else ""

    print("model_family", model_family)

    # ---------------- GPT OSS (Harmony) ----------------
    if model_family == "gpt_oss":
        # Build chat-style messages
        messages: List[Dict[str, str]] = []
        if system_content:
            messages.append({
                "role": "system",
                "content": system_content + "\n Reasoning: High \n",
                }
                )
        for ex in norm_examples:
            messages.append({
                "role": "user",
                "content": _user_turn_text(ex["input"]),
                }
                )
            messages.append({
                "role": "assistant",
                "content": ex["output"].strip(),
                }
                )
        messages.append({
            "role": "user",
            "content": _user_turn_text(input_text),
            }
            )
        # Preferred: let tokenizer build the prompt with its chat template
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                )

        # Fallback: manual Harmony-like tokens
        SYSTEM = "<|system|>"
        USER = "<|user|>"
        ASSISTANT = "<|assistant|>"
        prompt_parts: List[str] = []
        if system_content:
            prompt_parts.append(f"{SYSTEM}\n{system_content}. Reasoning: High")
        for ex in norm_examples:
            prompt_parts.append(
                f"{USER}\n{_user_turn_text(ex['input'])}\n"
                f"{ASSISTANT}\n{ex['output'].strip()}"
                )
        # Final user turn; assistant left open
        prompt_parts.append(
            f"{USER}\n{_user_turn_text(input_text)}\n{ASSISTANT}\n"
            )
        return "\n\n".join(p.strip() for p in prompt_parts if p.strip())

    # ---------------- backup ----------------

    if model_family == "gpt_oss1":
        SYSTEM = "<|system|>"
        USER = "<|user|>"
        ASSISTANT = "<|assistant|>"

        prompt = ""

        # Harmony supports system; put system_message + (optional) instructions there.
        top = _top_preface()
        if top:
            prompt += f"{SYSTEM}\n{top}. Reasoning: High.\n"

        # Examples: closed user->assistant pairs
        for ex in norm_examples:
            prompt += (
                f"{USER}\n{_user_turn_text(ex['input'])}"
                f"{ASSISTANT}\n{ex['output']}\n"
            )

        # Final user; assistant open
        prompt += (
            f"{USER}\n{_user_turn_text(input_text)}"
            f"{ASSISTANT}\n"
        )
        return prompt

    # ---------------- DeepSeek R1 ----------------
    # Keep tokens EXACT. Do not "pretty" them.
    if model_family == "deepseek_r1":
        BOS = "<｜begin▁of▁sentence｜>"
        EOS = "<｜end▁of▁sentence｜>"
        USER = "<｜User｜>"
        ASSISTANT = "<｜Assistant｜>"

        prompt = BOS

        # DeepSeek doesn't have a strict system role in all runtimes.
        # Safest: prepend a plain-text preface once (system_message + optional instructions).
        top = _top_preface()
        if top:
            prompt += top

        # Examples as closed turns (EOS after assistant)
        for ex in norm_examples:
            prompt += (
                f"{USER}{_user_turn_text(ex['input'])}"
                f"{ASSISTANT}{ex['output']}{EOS}"
            )

        # Final user; assistant open (NO EOS)
        prompt += (
            f"{USER}{_user_turn_text(input_text)}"
            f"{ASSISTANT}"
        )
        return prompt

    # ---------------- Gemma 3 ----------------
    if model_family == "gemma3":
        BOS = "<bos>"
        START = "<start_of_turn>"
        END = "<end_of_turn>"

        prompt = BOS

        # Gemma supports system role. Put system_message + (optional) instructions there.
        top = _top_preface().strip()
        if top:
            prompt += (
                f"{START}system\n"
                f"{top}\n"
                f"{END}\n"
            )

        # Examples
        for ex in norm_examples:
            prompt += (
                f"{START}user\n"
                f"{_user_turn_text(ex['input'])}"
                f"{END}\n"
                f"{START}assistant\n"
                f"{ex['output']}\n"
                f"{END}\n"
            )

        # Final user
        prompt += (
            f"{START}user\n"
            f"{_user_turn_text(input_text)}"
            f"{END}\n"
        )

        # Final assistant open
        prompt += f"{START}assistant\n"
        return prompt

    # ---------------- Generic fallback ----------------
    # Designed to be accepted by "most" chat-completion wrappers.
    # Uses simple role headers; safe for plain text prompting too.
    # Build chat-style messages
    messages: List[Dict[str, str]] = []
    if system_content:
        messages.append({
            "role": "system",
            "content": system_content + "\n Reasoning: High \n",
            }
            )
    for ex in norm_examples:
        messages.append({
            "role": "user",
            "content": _user_turn_text(ex["input"]),
            }
            )
        messages.append({
            "role": "assistant",
            "content": ex["output"].strip(),
            }
            )
    messages.append({
        "role": "user",
        "content": _user_turn_text(input_text),
        }
        )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            )
    else:
        ValueError("No tokenizer")


def load_outlines_llm(model_name: str, config: dict):
    """
    Load a CausalLM + tokenizer and wrap it in an Outlines generator.

    - If `model_name` is an Unsloth gpt-oss variant, use FastLanguageModel.from_pretrained.
    - Otherwise fall back to vanilla AutoModelForCausalLM.from_pretrained.
    """

    name_lower = (model_name or "").lower()
    use_unsloth = "gpt-oss" in name_lower and "unsloth" in name_lower

    # You can control this via config too if you want:
    # use_unsloth = config.get("model", {}).get("use_unsloth", False) and "gpt-oss" in name_lower

    if use_unsloth:
        # ---- Unsloth path for gpt-oss-20b ----
        max_seq_len = config.get("inference", {}).get("model_max_length", 8146)

        # dtype=None lets Unsloth auto-detect BF16/FP16/FP32
        extra = {}
        if HF_ACCESS_TOKEN:
            extra["token"] = HF_ACCESS_TOKEN

        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            dtype=None,
            max_seq_length=max_seq_len,
            load_in_4bit=True,  # to fit on 48 GB
            full_finetuning=False,
            **extra,
            )
        # Switch model into inference mode (required by Unsloth)
        FastLanguageModel.for_inference(base_model)

    else:
        # ---- Standard HF path ----
        is_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if is_bf16 else torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_auth_token=HF_ACCESS_TOKEN,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HF_ACCESS_TOKEN,
            )

    # Wrap in Outlines generator (this is what your generate_fn expects)
    generator = outlines.from_transformers(base_model, tokenizer)
    return generator, tokenizer


def _truncate_failed_output_summary(failed_json_text: str, max_chars: int = 2000) -> str:
    """
    Keep the retry prompt bounded. Uses tail+head if huge.
    """
    s = (failed_json_text or "").strip()
    if len(s) <= max_chars:
        return s
    head = s[: max_chars // 2]
    tail = s[-(max_chars // 2):]
    return head + "\n...\n" + tail


def build_repair_instruction(
        error_details: str,
        current_task: str,
        schema_name: str = "ArgumentGraph",
        ) -> str:
    """
    Builds a repair instruction string ONLY (no template tokens).
    Keeps your original hinting style but can be shared across model families.
    """
    error_details = (error_details or "").strip()

    if current_task == "ASP":
        hint = (
            "1. Ensure at least two units and at least one relation.\n"
            "2. Every unit ID must be connected (each appears in at least one relation as source_id or target_id).\n"
            "3. Do not invent IDs not in the argument_units list.\n"
        )
        return (
            "I received the following JSON output in the previous attempt, but it failed validation.\n"
            "You MUST regenerate a corrected JSON object.\n"
            f"The new JSON MUST adhere to the '{schema_name}' Pydantic schema.\n\n"
            "--- VALIDATION ERROR ---\n"
            f"{error_details}\n"
            "--- HINT ---\n"
            f"{hint}"
            "---\n\n"
            "Return ONLY the corrected JSON object (no markdown, no commentary)."
        )

    if current_task == "AUE":
        hint = "1. Ensure at least two argument units.\n"
        return (
            "I received the following JSON output in the previous attempt, but it failed validation.\n"
            "You MUST regenerate a corrected JSON object.\n"
            "The new JSON MUST adhere to the 'ArgumentUnits' Pydantic schema.\n\n"
            "--- VALIDATION ERROR ---\n"
            f"{error_details}\n"
            "--- HINT ---\n"
            f"{hint}"
            "---\n\n"
            "Return ONLY the corrected JSON object (no markdown, no commentary)."
        )

    # Default: relations task
    hint = (
        "1. Predict at least one relation in the conversation.\n"
        "2. Do not invent IDs not in the argument_units list.\n"
    )
    return (
        "I received the following JSON output in the previous attempt, but it failed validation.\n"
        "You MUST regenerate a corrected JSON object.\n"
        "The new JSON MUST adhere to the 'ArgumentRelations' Pydantic schema.\n\n"
        "--- VALIDATION ERROR ---\n"
        f"{error_details}\n"
        "--- HINT ---\n"
        f"{hint}"
        "---\n\n"
        "Return ONLY the corrected JSON object (no markdown, no commentary)."
    )


# def build_repair_prompt_from_scratch(
#         model_family: ModelFamily,
#         original_input_text: str,
#         instructions: str,
#         selected_examples: List[Dict[str, Any]],
#         failed_json_text: str,
#         repair_instruction: str,
#         system_message: Optional[str] = None,
#         ) -> str:
#     """
#     Rebuild-from-scratch repair prompt:
#       - same examples
#       - same base instructions and input text
#       - adds a SHORT failed-output summary + repair instruction as an additional user message
#         right before the final assistant generation.
#
#     Important:
#       - We DO NOT append an ever-growing chain.
#       - We DO NOT mix token conventions between model families.
#     """
#     failed_summary = _truncate_failed_output_summary(failed_json_text, max_chars=2000)
#
#     # We keep the original task instructions in the original user turn,
#     # then add a second user turn that contains the repair context.
#     # This avoids rewriting your original instructions while still giving repair context.
#
#     if model_family == "gpt_oss":
#         SYSTEM = "<|system|>"
#         USER = "<|user|>"
#         ASSISTANT = "<|assistant|>"
#
#         norm_examples = _normalize_examples(selected_examples)
#         prompt = ""
#         if system_message:
#             prompt += f"{SYSTEM}\n{system_message.strip()}\n\n"
#
#         for ex in norm_examples:
#             prompt += (
#                 f"{USER}\n{instructions.strip()}\n\n"
#                 f"Input:\n{ex['input']}\n"
#                 f"{ASSISTANT}\n{ex['output']}\n"
#             )
#
#         # Original input turn
#         prompt += (
#             f"{USER}\n{instructions.strip()}\n\n"
#             f"Input:\n{original_input_text.strip()}\n"
#             f"{ASSISTANT}\n"
#         )
#
#         # Close assistant as if it produced the failed output, then add repair user, open assistant
#         # Since your gpt_oss prompt ends with an open assistant, we complete it then start a new user turn.
#         prompt = prompt.rstrip()  # keep neat
#         prompt += f"{failed_summary}\n\n"
#         prompt += (
#             f"{USER}\n{repair_instruction.strip()}\n\n"
#             f"{ASSISTANT}\n"
#         )
#         return prompt
#
#     if model_family == "deepseek_r1":
#         BOS = "<｜begin▁of▁sentence｜>"
#         EOS = "<｜end▁of▁sentence｜>"
#         USER = "<｜User｜>"
#         ASSISTANT = "<｜Assistant｜>"
#
#         norm_examples = _normalize_examples(selected_examples)
#         prompt = BOS
#         if system_message:
#             prompt += f"{system_message.strip()}\n\n"
#
#         for ex in norm_examples:
#             prompt += (
#                 f"{USER}{instructions.strip()}\n\n"
#                 f"Input:\n{ex['input']}\n"
#                 f"{ASSISTANT}{ex['output']}{EOS}"
#             )
#
#         # Original input turn, assistant open
#         prompt += (
#             f"{USER}{instructions.strip()}\n\n"
#             f"Input:\n{original_input_text.strip()}\n"
#             f"{ASSISTANT}"
#         )
#
#         # Complete failed assistant turn + EOS, add repair user, open assistant
#         prompt += f"{failed_summary}{EOS}"
#         prompt += f"{USER}{repair_instruction.strip()}\n{ASSISTANT}"
#         return prompt
#
#     if model_family == "gemma3":
#         BOS = "<bos>"
#         START = "<start_of_turn>"
#         END = "<end_of_turn>"
#
#         norm_examples = _normalize_examples(selected_examples)
#         prompt = BOS
#         if system_message:
#             prompt += (
#                 f"{START}system\n{system_message.strip()}\n{END}\n"
#             )
#
#         for ex in norm_examples:
#             prompt += (
#                 f"{START}user\n{instructions.strip()}\n\nInput:\n{ex['input']}\n{END}\n"
#                 f"{START}assistant\n{ex['output']}\n{END}\n"
#             )
#
#         # Original input turn + assistant (we will "close" it with the failed output)
#         prompt += (
#             f"{START}user\n{instructions.strip()}\n\nInput:\n{original_input_text.strip()}\n{END}\n"
#             f"{START}assistant\n{failed_summary}\n{END}\n"
#         )
#
#         # Repair user turn + open assistant
#         prompt += (
#             f"{START}user\n{repair_instruction.strip()}\n{END}\n"
#             f"{START}assistant\n"
#         )
#         return prompt
#
#     # generic_chat: plain transcript with explicit roles
#     norm_examples = _normalize_examples(selected_examples)
#     prompt = ""
#     if system_message:
#         prompt += f"SYSTEM:\n{system_message.strip()}\n\n"
#
#     for ex in norm_examples:
#         prompt += (
#             f"USER:\n{instructions.strip()}\n\nInput:\n{ex['input']}\n\n"
#             f"ASSISTANT:\n{ex['output']}\n\n"
#         )
#
#     prompt += (
#         f"USER:\n{instructions.strip()}\n\nInput:\n{original_input_text.strip()}\n\n"
#         f"ASSISTANT:\n{failed_summary}\n\n"
#         f"USER:\n{repair_instruction.strip()}\n\n"
#         f"ASSISTANT:\n"
#     )
#     return prompt


def build_repair_prompt_from_scratch(
        model_family: ModelFamily,
        original_input_text: str,
        instructions: str,
        selected_examples: List[Dict[str, Any]],
        failed_json_text: str,
        repair_instruction: str,
        system_message: Optional[str] = None,
        repeat_instructions: bool = True,
        tokenizer: Optional[Any] = None,
        ) -> str:
    """
    Build a fresh retry prompt after a validation failure.

    Behaviour:
    ----------
    - Uses the SAME few-shot examples and the SAME core `instructions` as the
      original call.
    - Does NOT append an ever-growing turn history.
    - Does NOT mix token conventions; all model-specific chat formatting is
      delegated to `build_prompt(...)`.

    The only change vs. the original prompt is that the final `Input:` now
    contains:
      - the original input text, and
      - a SHORT summary of the previously invalid JSON, plus
      - the new repair instructions.

    This gives the model enough context to correct itself without dragging
    the whole conversation history along with it.
    """
    # Shorten the failed JSON to within length limits.
    failed_summary = _truncate_failed_output_summary(
        failed_json_text,
        max_chars=1000,
        )

    # This block explains (in-user-voice) what went wrong and what to fix.
    repair_block = (
        "NOTE: The previous JSON response for this input was invalid and failed schema validation.\n"
        "Here is the INVALID JSON output (for REFERENCE only – do NOT copy it verbatim):\n"
        f"{failed_summary}\n\n"
        f"{repair_instruction.strip()}"
    )

    # Combine original task input and repair context into a single 'Input:' payload.
    # From the model's POV, this is just a richer final user input.
    combined_input = (
        f"{original_input_text.strip()}\n\n"
        f"{repair_block}"
    )

    # Delegate all chat-token formatting to the unified builder.
    return build_prompt(
        model_family=model_family,
        input_text=combined_input,
        instructions=instructions,
        selected_examples=selected_examples,
        system_message=system_message,
        repeat_instructions=repeat_instructions,
        tokenizer=tokenizer,
        )


def generate_fn(
        input_text: str,
        generator,
        tokenizer,
        output_type,
        instructions: str,
        selected_examples: List[Dict[str, Any]],
        config: Dict[str, Any],
        current_task: str,
        system_message: Optional[str] = None,
        ) -> Tuple[Any, Dict[str, int]]:
    """
    Generates JSON structures using few-shot examples and rebuild-from-scratch repair.

    Assumptions:
      - `generator(prompt, output_type=..., ...)` returns either a JSON string or a dict/list.
      - `output_type.model_validate_json(json_text)` validates the JSON text.

    Returns:
      (validated_graph_or_none, metrics)
    """


    # single-prompt setup
    model_name = (
            config.get("model", {}).get("model_name_or_path")
            or config.get("inference", {}).get("model_name_or_path")
            or ""
    )

    # multi-prompt setup: AUE/RTC
    if not model_name:
        if current_task == "AUE":
            model_name = config.get("model", {}).get("args_model_name_or_path", "")
        elif current_task in "RTC":
            model_name = config.get("model", {}).get("rels_model_name_or_path", "")

    model_family = _detect_model_family_from_name(model_name)
    debug_io = config.get("experiment", {}).get("debug_llm_io", False)

    if debug_io:
        logging.info(f"[DEBUG] current_task={current_task}, model_name={model_name}, model_family={model_family}")

    repeat = config["inference"].get("repeat_instructions", False)

    # Build the base prompt once (few-shot + input; assistant open)
    base_prompt = build_prompt(
        model_family=model_family,
        input_text=input_text,
        instructions=instructions,
        selected_examples=selected_examples,
        system_message=system_message,
        repeat_instructions=repeat,
        tokenizer=tokenizer,
        )

    metrics = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    current_prompt = base_prompt

    if current_task == "AUE":
        task_cap = config["inference"].get("max_tokens_aue", 2048)
    elif current_task in "RTC":
        task_cap = config["inference"].get("max_tokens_rtc", 768)
    else:
        # for joint structure prediction
        task_cap = config["inference"].get("max_tokens_joint", 2048)

    model_max = config["inference"].get("model_max_length", 8192)
    prompt_tokens = count_tokens(current_prompt, tokenizer)
    # reserve a bit of headroom, so we never exactly hit the limit
    headroom = 128
    budget_from_context = max(128, model_max - prompt_tokens - headroom)

    max_new = min(task_cap, budget_from_context)

    use_sampling = True if "gpt_oss" in model_family else False

    for attempt in range(config["inference"]["max_attempts"]):
        # Token accounting
        attempt_prompt_tokens = count_tokens(current_prompt, tokenizer)
        metrics["prompt_tokens"] += attempt_prompt_tokens
        logging.info(f"\n[Attempt {attempt + 1}/{config['inference']['max_attempts']}] Generating...")

        structured_json_output = generator(
            current_prompt,
            output_type=output_type,
            max_new_tokens=max_new,
            do_sample=use_sampling,
            temperature=0.1 if use_sampling else 0.0,
            top_p=0.9 if use_sampling else 1.0,
            repetition_penalty=config["inference"]["repetition_penalty"],
            )

        if isinstance(structured_json_output, (dict, list)):
            structured_json_text = json.dumps(structured_json_output, ensure_ascii=False)
        else:
            structured_json_text = str(structured_json_output)

        if debug_io and model_family == "gpt_oss":
            logging.debug(
                "[DEBUG][GPT-OSS] Raw generator output (first 1200 chars):\n%s",
                structured_json_text[:1200],
                )

        try:
            raw = json.loads(structured_json_text)

            # ONLY continue if it's a dict; anything else is obviously wrong
            if isinstance(raw, dict):

                # --- Normalize top-level keys, still respecting structure ---
                # We do NOT add new semantic keys, just fix common aliasing.
                if current_task == "AUE":
                    # Prefer "argument_units"
                    if "units" in raw and "argument_units" not in raw:
                        raw["argument_units"] = raw.pop("units")

                    # Strip other random top-level keys
                    raw = {k: v for k, v in raw.items() if k == "argument_units"}

                    # Normalize None to []
                    if raw.get("argument_units") is None:
                        raw["argument_units"] = []

                elif current_task == "RTC":
                    raw = {k: v for k, v in raw.items() if k == "relations"}
                    if raw.get("relations") is None:
                        raw["relations"] = []

                else:  # ASP / full graph
                    # Allow only these keys
                    allowed = {"argument_units", "units", "relations"}
                    raw = {k: v for k, v in raw.items() if k in allowed}
                    if "units" in raw and "argument_units" not in raw:
                        raw["argument_units"] = raw.pop("units")

            structured_json_text = json.dumps(raw, ensure_ascii=False)

            metrics["completion_tokens"] += count_tokens(structured_json_text, tokenizer)

            validated_graph = output_type.model_validate_json(structured_json_text)
            metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
            logging.info(f"SUCCESS on attempt {attempt + 1}. Total tokens: {metrics['total_tokens']}")
            return validated_graph, metrics

        except json.JSONDecodeError as e:

            # Treat JSON syntax errors as recoverable and trigger repair
            error_details = f"JSONDecodeError: {str(e)}"
            logging.error(
                f"JSON decoding failed on attempt {attempt + 1}: {error_details.splitlines()[0]}"
                )

            if debug_io:
                logging.error(
                    "[DEBUG][%s] Raw text that failed JSON decoding (first 2000 chars):\n%s",
                    model_family,
                    structured_json_text[:2000],
                    )

            last_failed_text = structured_json_text
            repair_instruction = build_repair_instruction(
                error_details=error_details,
                current_task=current_task,
                schema_name=getattr(output_type, "__name__", "ArgumentGraph"),
                )

            # Rebuild from scratch: base prompt + failed summary + repair instruction (single repair turn)
            current_prompt = build_repair_prompt_from_scratch(
                model_family=model_family,
                original_input_text=input_text,
                instructions=instructions,
                selected_examples=selected_examples,
                failed_json_text=last_failed_text,
                repair_instruction=repair_instruction,
                system_message=system_message,
                repeat_instructions=repeat,
                tokenizer=tokenizer
                )

            # go to next attempt
            continue

        except (ValidationError, ValueError) as e:
            error_details = str(e)
            logging.error(
                f"Validation failed on attempt {attempt + 1}: {e.__class__.__name__}: {error_details.splitlines()[0]}"
                )
            if debug_io and model_family == "gpt_oss":
                logging.error("[DEBUG][GPT-OSS] Full validation error:\n%s", error_details)
                logging.error(
                    "[DEBUG][GPT-OSS] JSON that failed validation (first 2000 chars):\n%s",
                    structured_json_text[:2000],
                    )
            last_failed_text = structured_json_text

            repair_instruction = build_repair_instruction(
                error_details=error_details,
                current_task=current_task,
                schema_name=getattr(output_type, "__name__", "ArgumentGraph"),
                )

            # Rebuild from scratch: base prompt + failed summary + repair instruction (single repair turn)
            current_prompt = build_repair_prompt_from_scratch(
                model_family=model_family,
                original_input_text=input_text,
                instructions=instructions,
                selected_examples=selected_examples,
                failed_json_text=last_failed_text,
                repair_instruction=repair_instruction,
                system_message=system_message,
                repeat_instructions=repeat,
                tokenizer=tokenizer,
                )

    # Exhausted retries
    metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    logging.error("All attempts failed; returning None.")
    return None, metrics


_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+")


def filter_and_sort_argument_units(
        argument_units: ArgumentUnits,
        discussion_text: str,
        ) -> ArgumentUnits:
    """
    - Drop malformed units that do not contain any English-looking token.
    - Sort remaining units chronologically by their first appearance in the discussion text.
    - Reassign ids as 1..N in that chronological order.
    """
    # 1) Filter units with at least one English word
    filtered: List[ArgumentUnit] = [
        u for u in argument_units.argument_units
        if _ENGLISH_WORD_RE.search(u.text or "")
        ]

    if not filtered:
        # Let validator handle the "at least 2 units" condition later, but we return empty container
        return ArgumentUnits(argument_units=[])

    # 2) Sort by textual position in the discussion; fall back to original id if not found
    def sort_key(u: ArgumentUnit):
        idx = discussion_text.find(u.text)
        # If .find fails, send to the end but keep stable ordering via id
        return (idx if idx != -1 else 10 ** 9, u.id)

    filtered.sort(key=sort_key)

    # 3) Reindex ids to be sequential after filtering
    for new_id, u in enumerate(filtered, start=1):
        u.id = new_id

    # 4) Re-wrap in ArgumentUnits; validator will enforce >=2 if you keep that constraint
    return ArgumentUnits(argument_units=filtered)


def build_multiprompt_context_examples(
        source_path: str,
        aue_out_path: str,
        rtc_out_path: str,
        ) -> None:
    """
    Convert a single-prompt context examples file into two multi-prompt files:
      - AUE (Argument Unit Extraction) examples
      - RTC (Relation Classification) examples

    Source schema (per example), e.g. qt30_context_examples.json:
        {
          "conversation_id": "...",
          "conversation_text": "...",
          "argument_objects": {
            "argument_units": [...],
            "relations": [...]
          },
          "dropped_units": 0
        }

    Output schemas:

    AUE file: list of examples
        {
          "conversation_id": "...",
          "input": {
            "conversation_text": "..."
          },
          "output": {
            "argument_units": [...]
          }
        }

    RTC file: list of examples
        {
          "conversation_id": "...",
          "input": {
            "conversation_text": "...",
            "argument_units": [...]
          },
          "output": {
            "relations": [...]
          }
        }
    """
    src = Path(source_path)
    aue_out = Path(aue_out_path)
    rtc_out = Path(rtc_out_path)

    with src.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    aue_examples: List[Dict[str, Any]] = []
    rtc_examples: List[Dict[str, Any]] = []

    for ex in examples:
        conv_id = ex.get("conversation_id")
        conv_text = ex.get("conversation_text", "")
        arg_objs = ex.get("argument_objects", {}) or {}
        units = arg_objs.get("argument_units", []) or []
        relations = arg_objs.get("relations", []) or []

        # --- AUE (argument units only) ---
        aue_examples.append(
            {
                "conversation_id": conv_id,
                "input": {
                    "conversation_text": conv_text,
                    },
                "output": {
                    "argument_units": units,
                    },
                }
            )

        # --- RTC (relations only) ---
        rtc_examples.append(
            {
                "conversation_id": conv_id,
                "input": {
                    "conversation_text": conv_text,
                    "argument_units": units,
                    },
                "output": {
                    "relations": relations,
                    },
                }
            )

    aue_out.write_text(
        json.dumps(aue_examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
        )
    rtc_out.write_text(
        json.dumps(rtc_examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
        )
