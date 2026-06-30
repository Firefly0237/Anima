"""DPO comparison entrypoint for W4.

This is the conservative, reward-independent comparison arm: it reads W2 DPO
pairs built from the same prompts as GRPO, validates ``chosen``/``rejected``
strictly, and leaves heavyweight TRL imports inside the training path so local
dry-runs remain cheap.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from anima.data.build_reward_records import normalize_text_value
from anima.train.common import (
    PreparedRows,
    load_config,
    load_policy_model_and_tokenizer,
    parse_paths,
    read_records,
    record_to_messages,
    render_policy_prompt,
    render_sft_completion,
    render_sft_completion_content,
    write_json,
)


class DPOPairError(ValueError):
    """Raised when a record cannot be used as a strict DPO pair."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dpo_qwen3b.yaml"))
    parser.add_argument("--dpo-jsonl", action="append", help="Override/add DPO pair JSON/JSONL input path.")
    parser.add_argument("--output-dir", help="Override adapter output directory.")
    parser.add_argument("--sft-adapter", help="Train from an existing SFT LoRA adapter.")
    parser.add_argument("--max-records", type=int, help="Cap records for smoke runs.")
    parser.add_argument(
        "--dpo-schema",
        choices=("legacy", "chat_template"),
        help="Override config.dpo_schema. chat_template uses TRL conversational preference rows.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Normalize pairs and print sample rows only.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dpo_jsonl:
        config["train_files"] = args.dpo_jsonl
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.sft_adapter:
        config["adapter_path"] = args.sft_adapter
    if args.max_records is not None:
        config["max_records"] = args.max_records
    if args.dpo_schema:
        config["dpo_schema"] = args.dpo_schema

    _assert_beta_positive(config)

    train_files = _dpo_input_paths(config)
    if not train_files:
        raise SystemExit("DPO needs at least one pair file via config.train_files or --dpo-jsonl")

    dpo_schema = str(config.get("dpo_schema", "legacy"))
    prepared = build_dpo_rows(
        train_files,
        max_records=_optional_int(config.get("max_records")),
        dpo_schema=dpo_schema,
    )
    if not prepared.rows:
        raise SystemExit(f"No usable DPO pairs found in {[str(path) for path in train_files]}")

    pair_summary = summarize_dpo_rows(prepared.rows)
    summary = {
        "stage": "dpo",
        "dpo_schema": dpo_schema,
        "config": str(args.config),
        "input_files": [str(path) for path in prepared.input_files],
        "rows": len(prepared.rows),
        "skipped": prepared.skipped,
        "output_dir": str(config.get("output_dir")),
        "adapter_path": str(config.get("adapter_path") or ""),
        "beta": float(config.get("beta", 0.1)),
        "loss_type": str(config.get("loss_type", "sigmoid")),
        "reward_independent": True,
        "target_transform": "explicit_character_r1_target_rendering",
        "note": "DPO comparison arm: matched prompts, chosen=reference, rejected=degraded synthetic; never reward-ranked.",
        **pair_summary,
    }

    if args.dry_run:
        print(
            json.dumps(
                {**summary, "sample": prepared.rows[: int(config.get("dry_run_samples", 2))]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    train(config, prepared.rows, resume_from_checkpoint=args.resume_from_checkpoint, summary=summary)
    return 0


def build_dpo_rows(
    paths: Sequence[Path],
    *,
    max_records: int | None = None,
    dpo_schema: str = "legacy",
) -> PreparedRows:
    """Build strict TRL DPO rows from W2 pair JSONL or reward records."""

    if dpo_schema not in {"legacy", "chat_template"}:
        raise ValueError(f"unsupported dpo_schema={dpo_schema!r}; expected legacy or chat_template")
    prepared = read_records(paths, max_records=max_records)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(prepared.rows):
        try:
            rows.append(_normalize_dpo_record(record, index=index, dpo_schema=dpo_schema))
        except DPOPairError as exc:
            record_id = str(record.get("id") or f"record_{index:06d}")
            raise ValueError(f"DPO pair validation failed for {record_id}: {exc}") from exc
    return PreparedRows(rows=rows, skipped=0, input_files=prepared.input_files)


def train(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    resume_from_checkpoint: str | None,
    summary: dict[str, Any],
) -> None:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(str(config.get("output_dir", "/home/featurize/work/anima/models/dpo-3b-w4")))
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = str(config.get("adapter_path") or "").strip() or None
    model, tokenizer, peft_config = load_policy_model_and_tokenizer(
        config,
        padding_side=str(config.get("padding_side", "right")),
        adapter_path=adapter_path,
    )
    if str(config.get("dpo_schema", "legacy")) == "chat_template" and not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("dpo_schema=chat_template requires a tokenizer with a non-empty chat_template")
    dataset = Dataset.from_list(rows)
    training_args = _build_dpo_config(DPOConfig, config, output_dir)

    trainer_kwargs = _dpo_trainer_kwargs(
        DPOTrainer,
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer = DPOTrainer(**trainer_kwargs)
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(train_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    write_json(
        output_dir / "anima_dpo_summary.json",
        {
            **summary,
            "train_metrics": metrics,
            "adapter_path": adapter_path,
            "model_name_or_path": str(config.get("model_name_or_path")),
            "max_length": int(config.get("max_length", 1024)),
            "max_prompt_length": int(config.get("max_prompt_length", 768)),
            "max_target_length": int(config.get("max_target_length", config.get("max_completion_length", 256))),
            "precompute_ref_log_probs": bool(config.get("precompute_ref_log_probs", False)),
        },
    )


def _build_dpo_config(DPOConfig: Any, config: dict[str, Any], output_dir: Path) -> Any:
    requested = {
        "output_dir": str(output_dir),
        "run_name": str(config.get("run_name", output_dir.name)),
        "max_steps": int(config.get("max_steps", 80)),
        "num_train_epochs": float(config.get("num_train_epochs", 1.0)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 4)),
        "learning_rate": float(config.get("learning_rate", 5e-7)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "beta": float(config.get("beta", 0.1)),
        "loss_type": str(config.get("loss_type", "sigmoid")),
        "label_smoothing": float(config.get("label_smoothing", 0.0)),
        "max_length": int(config.get("max_length", 1024)),
        "max_prompt_length": int(config.get("max_prompt_length", 768)),
        "max_target_length": int(config.get("max_target_length", config.get("max_completion_length", 256))),
        "max_completion_length": int(config.get("max_completion_length", config.get("max_target_length", 256))),
        "truncation_mode": str(config.get("truncation_mode", "keep_end")),
        "precompute_ref_log_probs": bool(config.get("precompute_ref_log_probs", False)),
        "generate_during_eval": bool(config.get("generate_during_eval", False)),
        "disable_dropout": bool(config.get("disable_dropout", True)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_steps": int(config.get("save_steps", 40)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "bf16": bool(config.get("bf16", True)),
        "fp16": bool(config.get("fp16", False)),
        "tf32": bool(config.get("tf32", True)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "optim": str(config.get("optim", "paged_adamw_8bit")),
        "report_to": config.get("report_to", []),
        "seed": int(config.get("seed", 42)),
        "remove_unused_columns": bool(config.get("remove_unused_columns", False)),
    }

    supported = _supported_config_keys(DPOConfig)
    if not supported:
        return DPOConfig(**requested)

    dropped = sorted(set(requested) - supported)
    if dropped:
        _fail_on_critical_dropped(
            stage="dpo_config",
            dropped=dropped,
            critical={"beta", "loss_type", "max_length", "max_prompt_length"},
        )
        print(
            json.dumps(
                {
                    "stage": "dpo_config",
                    "dropped_unsupported_keys": dropped,
                    "note": "Installed TRL DPOConfig does not expose these fields; training continues without them.",
                },
                ensure_ascii=False,
            )
        )
    return DPOConfig(**{key: value for key, value in requested.items() if key in supported})


def _fail_on_critical_dropped(*, stage: str, dropped: list[str], critical: set[str]) -> None:
    missing = sorted(set(dropped) & critical)
    if missing:
        raise RuntimeError(
            f"{stage} cannot continue because installed TRL does not support critical keys: {missing}"
        )


def _supported_config_keys(config_cls: Any) -> set[str]:
    fields = getattr(config_cls, "__dataclass_fields__", None)
    if isinstance(fields, dict) and fields:
        return set(fields)

    try:
        import inspect

        signature = inspect.signature(config_cls.__init__)
    except (TypeError, ValueError):
        return set()

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return set()
    return {name for name in params if name != "self"}


def _dpo_trainer_kwargs(
    trainer_cls: Any,
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    tokenizer: Any,
    peft_config: Any,
) -> dict[str, Any]:
    requested = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "peft_config": peft_config,
    }

    try:
        import inspect

        params = inspect.signature(trainer_cls.__init__).parameters
    except (TypeError, ValueError):
        return requested

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return requested

    if "processing_class" not in params and "tokenizer" in params:
        requested["tokenizer"] = requested.pop("processing_class")
    return {key: value for key, value in requested.items() if key in params}


def summarize_dpo_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chosen_lengths = [len(_completion_summary_text(row["chosen"])) for row in rows]
    rejected_lengths = [len(_completion_summary_text(row["rejected"])) for row in rows]
    strategies = Counter(str(row.get("rejected_strategy") or "unknown") for row in rows)
    return {
        "chosen_chars": _length_summary(chosen_lengths),
        "rejected_chars": _length_summary(rejected_lengths),
        "rejected_strategies": dict(sorted(strategies.items())),
    }


def _normalize_dpo_record(record: Mapping[str, Any], *, index: int, dpo_schema: str) -> dict[str, Any]:
    chosen_raw = _first_text(record, ("chosen", "reference_answer"))
    rejected_raw = _first_text(record, ("rejected", "rejected_answer"))
    if not chosen_raw:
        raise DPOPairError("chosen/reference_answer must be non-empty")
    if not rejected_raw:
        raise DPOPairError("rejected/rejected_answer must be non-empty")
    if _canonical_pair_text(chosen_raw) == _canonical_pair_text(rejected_raw):
        raise DPOPairError("chosen and rejected must differ")

    if dpo_schema == "chat_template":
        prompt: str | list[dict[str, str]] = record_to_messages(record)
        prompt_ok = bool(prompt)
        chosen: str | list[dict[str, str]] = [
            {"role": "assistant", "content": _ensure_formatted_completion_content(record, chosen_raw)}
        ]
        rejected: str | list[dict[str, str]] = [
            {"role": "assistant", "content": _ensure_formatted_completion_content(record, rejected_raw)}
        ]
    else:
        prompt = _record_prompt(record)
        prompt_ok = bool(prompt.strip())
        chosen = _ensure_formatted_completion(record, chosen_raw)
        rejected = _ensure_formatted_completion(record, rejected_raw)

    if not prompt_ok:
        raise DPOPairError("prompt must be non-empty")

    row: dict[str, Any] = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "id": str(record.get("id") or f"dpo_{index:06d}"),
        "source_record_id": str(record.get("source_record_id") or record.get("id") or f"dpo_{index:06d}"),
        "character": _first_text(record, ("character", "role", "name", "character_name")) or "unknown",
        "source_work": _first_text(record, ("source_work", "work", "book", "movie")) or "unknown",
    }

    rejected_strategy = _rejected_strategy(record)
    if rejected_strategy:
        row["rejected_strategy"] = rejected_strategy
    return row


def _record_prompt(record: Mapping[str, Any]) -> str:
    prompt = normalize_text_value(record.get("prompt"))
    if prompt:
        return prompt
    return render_policy_prompt(record)


def _ensure_formatted_completion(record: Mapping[str, Any], answer: str) -> str:
    """Keep DPO targets consistent with the Character-R1 output contract."""

    if "<think>" in answer and "\\boxed{" in answer:
        return answer
    return render_sft_completion(record, answer=answer)


def _ensure_formatted_completion_content(record: Mapping[str, Any], answer: str) -> str:
    """Return assistant content for conversational DPO targets."""

    if "<think>" in answer and "\\boxed{" in answer:
        return _strip_known_eos(answer)
    return render_sft_completion_content(record, answer=answer)


def _strip_known_eos(text: str) -> str:
    stripped = text.strip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        if stripped.endswith(token):
            stripped = stripped[: -len(token)].rstrip()
    return stripped


def _completion_summary_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return normalize_text_value(value.get("content")) or str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "\n".join(_completion_summary_text(item) for item in value)
    return str(value)


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key not in record:
            continue
        text = normalize_text_value(record[key])
        if text:
            return text
    return ""


def _rejected_strategy(record: Mapping[str, Any]) -> str:
    text = normalize_text_value(record.get("rejected_strategy"))
    if text:
        return text
    synth_meta = record.get("synth_meta")
    if isinstance(synth_meta, Mapping):
        return normalize_text_value(synth_meta.get("rejected_strategy"))
    return ""


def _canonical_pair_text(text: str) -> str:
    return "".join(text.split())


def _length_summary(lengths: Sequence[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)}


def _dpo_input_paths(config: Mapping[str, Any]) -> list[Path]:
    value = config.get("train_files")
    if value is None:
        value = config.get("dpo_files")
    if value is None:
        value = config.get("dpo_jsonl")
    return parse_paths(value)


def _assert_beta_positive(config: Mapping[str, Any]) -> None:
    beta = float(config.get("beta", 0.1))
    if beta <= 0.0:
        raise SystemExit(f"Refusing DPO with beta={beta}. W4 requires beta>0 for the KL/reference anchor.")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
