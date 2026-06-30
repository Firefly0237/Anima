"""GRPO entrypoint for W3 first real reward-data run.

Default configs keep ``use_vllm=false`` for the first 1x4090 route proof because
the logged Featurize baseline is TRL 0.17.0. Flip vLLM/Unsloth only after the
server-side compatibility smoke is recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anima.rewards.combine import component_rewards, get_reward_funcs, get_reward_weights
from anima.train.common import (
    build_grpo_chat_rows,
    build_grpo_rows,
    load_config,
    load_policy_model_and_tokenizer,
    parse_paths,
    render_sft_completion_content,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grpo_qwen3b.yaml"))
    parser.add_argument("--reward-jsonl", action="append", help="Override/add reward JSONL input path.")
    parser.add_argument("--output-dir", help="Override adapter output directory.")
    parser.add_argument("--sft-adapter", help="Train from an existing SFT LoRA adapter.")
    parser.add_argument("--max-records", type=int, help="Cap records for smoke runs.")
    parser.add_argument(
        "--grpo-schema",
        choices=("legacy", "chat_template"),
        help="Override config.grpo_schema. chat_template uses official conversational prompts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Normalize rows and score a perfect-completion sample only.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.reward_jsonl:
        config["train_files"] = args.reward_jsonl
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.sft_adapter:
        config["adapter_path"] = args.sft_adapter
    if args.max_records is not None:
        config["max_records"] = args.max_records
    if args.grpo_schema:
        config["grpo_schema"] = args.grpo_schema

    _assert_beta_positive(config)
    _assert_batch_divisible(config)

    train_files = parse_paths(config.get("train_files"))
    if not train_files:
        raise SystemExit("GRPO needs at least one reward file via config.train_files or --reward-jsonl")

    grpo_schema = str(config.get("grpo_schema", "legacy"))
    if grpo_schema == "chat_template":
        prepared = build_grpo_chat_rows(
            train_files,
            max_records=_optional_int(config.get("max_records")),
            validate_schema=bool(config.get("validate_schema", True)),
        )
    elif grpo_schema == "legacy":
        prepared = build_grpo_rows(
            train_files,
            max_records=_optional_int(config.get("max_records")),
            validate_schema=bool(config.get("validate_schema", True)),
        )
    else:
        raise SystemExit(f"unsupported grpo_schema={grpo_schema!r}; expected legacy or chat_template")
    if not prepared.rows:
        raise SystemExit(f"No usable GRPO rows found in {[str(path) for path in train_files]}")

    summary = {
        "stage": "grpo",
        "grpo_schema": grpo_schema,
        "config": str(args.config),
        "input_files": [str(path) for path in prepared.input_files],
        "rows": len(prepared.rows),
        "skipped": prepared.skipped,
        "output_dir": str(config.get("output_dir")),
        "beta": float(config.get("beta", 0.02)),
        "reward_weights": get_reward_weights(),
        "reward_note": "Rule-based RLVR only; BaichuanCharRM is eval-only and not used here.",
    }

    if args.dry_run:
        sample = prepared.rows[0]
        perfect = render_sft_completion_content(
            {
                "gold_focus": sample["gold_focus"],
                "gold_focus_attr": sample["gold_focus_attr"],
                "reference_answer": sample["reference_answer"],
                "character": sample.get("character"),
            },
            answer=str(sample["reference_answer"]),
        )
        rewards = component_rewards(
            [perfect],
            gold_focus=[sample["gold_focus"]],
            gold_focus_attr=[sample["gold_focus_attr"]],
            reference_answer=[sample["reference_answer"]],
        )
        print(
            json.dumps(
                {
                    **summary,
                    "sample_prompt": sample["prompt"],
                    "sample_perfect_completion": perfect,
                    "sample_component_rewards": rewards,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    train(config, prepared.rows, resume_from_checkpoint=args.resume_from_checkpoint, summary=summary)
    return 0


def train(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    resume_from_checkpoint: str | None,
    summary: dict[str, Any],
) -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    output_dir = Path(str(config.get("output_dir", "/home/featurize/work/anima/models/grpo-3b-w3-smoke")))
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = str(config.get("adapter_path") or "").strip() or None
    model, tokenizer, peft_config = load_policy_model_and_tokenizer(
        config,
        padding_side="left",
        adapter_path=adapter_path,
    )
    if str(config.get("grpo_schema", "legacy")) == "chat_template" and not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("grpo_schema=chat_template requires a tokenizer with a non-empty chat_template")
    dataset = Dataset.from_list(rows)

    training_args = _build_grpo_config(GRPOConfig, config, output_dir)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=get_reward_funcs(),
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(train_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    write_json(
        output_dir / "anima_grpo_summary.json",
        {
            **summary,
            "train_metrics": metrics,
            "adapter_path": adapter_path,
            "model_name_or_path": str(config.get("model_name_or_path")),
            "max_prompt_length": int(config.get("max_prompt_length", 768)),
            "max_completion_length": int(config.get("max_completion_length", 384)),
            "mask_truncated_completions": bool(config.get("mask_truncated_completions", True)),
            "repetition_penalty": float(config.get("repetition_penalty", 1.05)),
            "use_vllm": bool(config.get("use_vllm", False)),
        },
    )


def _build_grpo_config(GRPOConfig: Any, config: dict[str, Any], output_dir: Path) -> Any:
    requested = {
        "output_dir": str(output_dir),
        "run_name": str(config.get("run_name", output_dir.name)),
        "max_steps": int(config.get("max_steps", 50)),
        "num_train_epochs": float(config.get("num_train_epochs", 1.0)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 4)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
        "num_generations": int(config.get("num_generations", 4)),
        "max_prompt_length": int(config.get("max_prompt_length", 768)),
        "max_completion_length": int(config.get("max_completion_length", 384)),
        "temperature": float(config.get("temperature", 0.9)),
        "top_p": float(config.get("top_p", 1.0)),
        "top_k": _optional_int(config.get("top_k", 50)),
        "beta": float(config.get("beta", 0.02)),
        "learning_rate": float(config.get("learning_rate", 1e-6)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "loss_type": str(config.get("loss_type", "dr_grpo")),
        "reward_weights": get_reward_weights(),
        "scale_rewards": bool(config.get("scale_rewards", True)),
        "log_completions": bool(config.get("log_completions", True)),
        "num_completions_to_print": _optional_int(config.get("num_completions_to_print", 2)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_steps": int(config.get("save_steps", 50)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "bf16": bool(config.get("bf16", True)),
        "fp16": bool(config.get("fp16", False)),
        "tf32": bool(config.get("tf32", True)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "optim": str(config.get("optim", "paged_adamw_8bit")),
        "report_to": config.get("report_to", []),
        "seed": int(config.get("seed", 42)),
        "remove_unused_columns": bool(config.get("remove_unused_columns", False)),
        "use_vllm": bool(config.get("use_vllm", False)),
        "mask_truncated_completions": bool(config.get("mask_truncated_completions", True)),
        "repetition_penalty": float(config.get("repetition_penalty", 1.05)),
    }

    supported = _supported_grpo_config_keys(GRPOConfig)
    if not supported:
        return GRPOConfig(**requested)

    dropped = sorted(set(requested) - supported)
    if dropped:
        _fail_on_critical_dropped(
            stage="grpo_config",
            dropped=dropped,
            critical={"beta", "loss_type", "reward_weights", "max_prompt_length", "max_completion_length"},
        )
        print(
            json.dumps(
                {
                    "stage": "grpo_config",
                    "dropped_unsupported_keys": dropped,
                    "note": "Installed TRL GRPOConfig does not expose these fields; training continues without them.",
                },
                ensure_ascii=False,
            )
        )
    return GRPOConfig(**{key: value for key, value in requested.items() if key in supported})


def _fail_on_critical_dropped(*, stage: str, dropped: list[str], critical: set[str]) -> None:
    missing = sorted(set(dropped) & critical)
    if missing:
        raise RuntimeError(
            f"{stage} cannot continue because installed TRL does not support critical keys: {missing}"
        )


def _supported_grpo_config_keys(GRPOConfig: Any) -> set[str]:
    fields = getattr(GRPOConfig, "__dataclass_fields__", None)
    if isinstance(fields, dict) and fields:
        return set(fields)

    try:
        import inspect

        signature = inspect.signature(GRPOConfig.__init__)
    except (TypeError, ValueError):
        return set()

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return set()
    return {name for name in params if name != "self"}


def _assert_beta_positive(config: dict[str, Any]) -> None:
    beta = float(config.get("beta", 0.02))
    if beta <= 0.0:
        raise SystemExit(f"Refusing GRPO with beta={beta}. W3 requires beta>0 for KL control.")


def _assert_batch_divisible(config: dict[str, Any]) -> None:
    batch = int(config.get("per_device_train_batch_size", 4))
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    generations = int(config.get("num_generations", 4))
    effective = batch * grad_accum
    if generations <= 0:
        raise SystemExit("num_generations must be positive")
    if effective % generations != 0:
        raise SystemExit(
            "TRL GRPO needs the effective train batch size to be divisible by "
            f"num_generations; got per_device={batch}, grad_accum={grad_accum}, "
            f"effective={effective}, num_generations={generations}."
        )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
