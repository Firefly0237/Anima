"""Generate policy rollouts for W4 evaluation.

This module is deliberately separate from the CharacterEval scorer. Policy
generation runs in the training environment; BaichuanCharRM scoring runs later
in the isolated RM env. The output JSONL is the shared input for W4 eval axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from anima.train.common import (
    ChatTemplateError,
    load_config,
    parse_paths,
    read_records,
    record_to_messages,
    render_messages_with_chat_template,
    render_policy_prompt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config.")
    parser.add_argument("--input-jsonl", action="append", help="Eval seed JSON/JSONL path.")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="Stable arm name, e.g. base/sft/dpo/grpo.")
    parser.add_argument("--axis", default="manual", help="Eval axis label stored in each row.")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--prompt-format", choices=("legacy", "chat_template"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Normalize rows without loading a model.")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else {}
    config = _with_cli_overrides(config, args)

    input_paths = parse_paths(args.input_jsonl or config.get("eval_files") or config.get("input_files"))
    if not input_paths:
        raise SystemExit("generate_rollouts needs --input-jsonl or config eval_files/input_files")

    prepared = read_records(input_paths, max_records=_optional_int(config.get("max_records")))
    rows = [_normalize_seed(record, arm=args.arm, axis=str(config.get("axis", args.axis))) for record in prepared.rows]
    if not rows:
        raise SystemExit(f"No rollout seed rows found in {[str(path) for path in input_paths]}")

    summary = {
        "stage": "generate_rollouts",
        "arm": args.arm,
        "axis": str(config.get("axis", args.axis)),
        "input_files": [str(path) for path in prepared.input_files],
        "rows": len(rows),
        "output_jsonl": str(args.output_jsonl),
        "model_name_or_path": str(config.get("model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")),
        "adapter_path": _clean_optional(config.get("adapter_path")),
        "max_prompt_length": int(config.get("max_prompt_length", 768)),
        "max_new_tokens": int(config.get("max_new_tokens", config.get("max_completion_length", 192))),
        "temperature": float(config.get("temperature", 0.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "top_k": _optional_int(config.get("top_k")),
        "prompt_format": str(config.get("prompt_format", "legacy")),
        "seed": int(config.get("seed", 42)),
        "note": "Policy rollouts only; external RM scoring is a separate offline step.",
    }

    if args.dry_run:
        preview = [dict(row, response="", completion_raw="") for row in rows[: int(config.get("dry_run_samples", 2))]]
        print(json.dumps({**summary, "sample": preview}, ensure_ascii=False, indent=2))
        return 0

    generated = generate(config, rows)
    write_jsonl(args.output_jsonl, generated)
    write_summary(args.output_jsonl, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def generate(config: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name = str(config.get("model_name_or_path", "Qwen/Qwen2.5-3B-Instruct"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(config.get("trust_remote_code", True)))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if bool(config.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=_torch_dtype(torch, str(config.get("bnb_4bit_compute_dtype", "bfloat16"))),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        torch_dtype=_torch_dtype(torch, str(config.get("torch_dtype", "bfloat16"))),
        quantization_config=quantization_config,
        device_map=config.get("device_map", "auto"),
    )
    adapter_path = _clean_optional(config.get("adapter_path"))
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    max_prompt_length = int(config.get("max_prompt_length", 768))
    max_new_tokens = int(config.get("max_new_tokens", config.get("max_completion_length", 192)))
    temperature = float(config.get("temperature", 0.0))
    do_sample = temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = float(config.get("top_p", 1.0))
        top_k = _optional_int(config.get("top_k"))
        if top_k is not None:
            generation_kwargs["top_k"] = top_k

    out: list[dict[str, Any]] = []
    for row in rows:
        prompt_text = _render_generation_prompt(row, tokenizer, config)
        should_truncate = str(config.get("prompt_format", "legacy")) == "legacy"
        encode_kwargs: dict[str, Any] = {"return_tensors": "pt", "truncation": should_truncate}
        if should_truncate:
            encode_kwargs["max_length"] = max_prompt_length
        encoded = tokenizer(prompt_text, **encode_kwargs)
        if not should_truncate and encoded["input_ids"].shape[-1] > max_prompt_length:
            raise ChatTemplateError(
                f"chat-template prompt has {encoded['input_ids'].shape[-1]} tokens "
                f"after message-level fitting, above max_prompt_length={max_prompt_length}"
            )
        device = getattr(model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
        completion = _decode_completion(tokenizer, new_tokens)
        out.append(
            {
                **row,
                "rendered_prompt": prompt_text,
                "response": completion,
                "completion_raw": completion,
                "generation": {
                    "model_name_or_path": model_name,
                    "adapter_path": adapter_path,
                    "max_prompt_length": max_prompt_length,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "prompt_format": str(config.get("prompt_format", "legacy")),
                    "top_p": float(config.get("top_p", 1.0)),
                    "top_k": _optional_int(config.get("top_k")),
                    "seed": seed,
                },
            }
        )
    return out


def _normalize_seed(record: Mapping[str, Any], *, arm: str, axis: str) -> dict[str, Any]:
    prompt = _policy_prompt(record)
    row = {
        "id": str(record.get("id") or f"{axis}_{arm}_unknown"),
        "arm": arm,
        "axis": axis,
        "character": str(record.get("character") or record.get("role") or "unknown"),
        "source_work": str(record.get("source_work") or record.get("work") or "unknown"),
        "profile": str(record.get("profile") or record.get("persona") or record.get("character_profile") or ""),
        "conversations": record.get("conversations") or record.get("dialogue") or record.get("history") or [],
        "prompt": prompt,
    }
    messages = _record_messages(record)
    if messages:
        row["messages"] = messages
    # Keep eval-only targets out of the prompt but preserve them for downstream
    # scorers. This is what lets one rollout JSONL feed held-out reward replay
    # and MCQ regression without hand-joining labels later.
    for key in (
        "policy_prompt",
        "question",
        "choices",
        "gold_focus",
        "gold_focus_attr",
        "reference_answer",
        "answer",
        "label",
        "gold",
        "target",
        "correct_answer",
        "split",
        "source",
    ):
        if key in record:
            row[key] = record[key]
    return row


def _record_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    existing = record.get("messages")
    if isinstance(existing, list) and existing:
        return [dict(message) for message in existing if isinstance(message, Mapping)]
    if record.get("conversations") or record.get("dialogue") or record.get("history"):
        return record_to_messages(record)
    return []


def _policy_prompt(record: Mapping[str, Any]) -> str:
    explicit = record.get("policy_prompt")
    if explicit not in (None, ""):
        return str(explicit)
    return render_policy_prompt(record)


def _with_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(config)
    for attr, key in (
        ("model_name_or_path", "model_name_or_path"),
        ("adapter_path", "adapter_path"),
        ("max_records", "max_records"),
        ("max_prompt_length", "max_prompt_length"),
        ("max_new_tokens", "max_new_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("prompt_format", "prompt_format"),
        ("seed", "seed"),
    ):
        value = getattr(args, attr)
        if value is not None:
            out[key] = value
    if args.input_jsonl:
        out["eval_files"] = args.input_jsonl
    if args.axis:
        out["axis"] = args.axis
    return out


def _render_generation_prompt(row: Mapping[str, Any], tokenizer: Any, config: Mapping[str, Any]) -> str:
    prompt_format = str(config.get("prompt_format", "legacy"))
    if prompt_format == "legacy":
        return str(row["prompt"])
    if prompt_format != "chat_template":
        raise ValueError(f"unsupported prompt_format: {prompt_format}")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ChatTemplateError("prompt_format=chat_template requires row.messages")
    max_prompt_length = _optional_int(config.get("max_prompt_length"))
    if max_prompt_length is None:
        return render_messages_with_chat_template(messages, tokenizer, mode="formal")
    return _render_chat_template_with_budget(messages, tokenizer, max_prompt_length=max_prompt_length)


def _render_chat_template_with_budget(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    max_prompt_length: int,
) -> str:
    rendered = render_messages_with_chat_template(messages, tokenizer, mode="formal")
    if _token_length(tokenizer, rendered) <= max_prompt_length:
        return rendered

    system_messages: list[Mapping[str, str]] = []
    dialogue_start = 0
    if messages and str(messages[0].get("role") or "") == "system":
        system_messages = [messages[0]]
        dialogue_start = 1
    dialogue = list(messages[dialogue_start:])

    for keep in range(len(dialogue) - 1, 0, -1):
        candidate_messages = [*system_messages, *dialogue[-keep:]]
        candidate = render_messages_with_chat_template(candidate_messages, tokenizer, mode="formal")
        if _token_length(tokenizer, candidate) <= max_prompt_length:
            return candidate

    raise ChatTemplateError(
        "chat-template prompt exceeds max_prompt_length even after keeping only "
        "system plus the latest dialogue turn"
    )


def _decode_completion(tokenizer: Any, token_ids: Any) -> str:
    return str(tokenizer.decode(token_ids, skip_special_tokens=False))


def _token_length(tokenizer: Any, text: str) -> int:
    if not callable(tokenizer):
        return len(text)
    encoded = tokenizer(text, return_tensors=None, truncation=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_summary(output_jsonl: Path, summary: Mapping[str, Any]) -> None:
    summary_path = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _torch_dtype(torch_module: Any, name: str):
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    if normalized in {"auto", "none"}:
        return "auto"
    raise ValueError(f"unsupported torch dtype: {name}")


if __name__ == "__main__":
    raise SystemExit(main())
