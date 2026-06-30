"""Aggregate four-arm eval metrics into Markdown and JSON summaries.

The aggregator is intentionally small and file-format tolerant:
- JSONL: one metric record per line.
- JSON: either a list of records, a single record, or an object with a common
  rows/data/results/records list.

Current CharacterEval output is supported via ``charactereval.score`` and
``charactereval.status``. The CharacterEval 13-metric slots are also wired so a
future scorer can add ``charactereval.metrics.<name>`` without changing this
script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ARMS = ("Base", "SFT", "DPO", "GRPO")
DEFAULT_AXES = ("charactereval", "heldout", "socialbench", "ceval")
DEFAULT_BASELINE_ARM = "SFT"

CHARACTEREVAL_13_METRICS = (
    "fluency",
    "coherency",
    "consistency",
    "knowledge_exposure",
    "knowledge_accuracy",
    "knowledge_hallucination",
    "persona_behavior",
    "persona_utterance",
    "human_likeness",
    "communication_skills",
    "expression_diversity",
    "empathy",
    "mbti_accuracy",
)

DEFAULT_STATUS_FIELDS = (
    "charactereval.status",
    "eval.status",
    "status",
)

DEFAULT_SCORE_FIELDS_BY_AXIS = {
    "charactereval": ("charactereval.score", "score", "mean_score"),
    "heldout": ("heldout.score", "score", "accuracy", "mean_score"),
    "socialbench": ("socialbench.accuracy", "accuracy", "score", "mean_score"),
    "ceval": ("ceval.accuracy", "accuracy", "score", "mean_score"),
}

MISSING = object()


@dataclass(frozen=True)
class AxisSpec:
    name: str
    label: str
    score_fields: tuple[str, ...]
    status_fields: tuple[str, ...]
    metric_fields: dict[str, tuple[str, ...]]


@dataclass
class NumericStats:
    count: int = 0
    total: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

    def to_json(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class AggregateBucket:
    count: int = 0
    score: NumericStats = field(default_factory=NumericStats)
    status_counts: Counter[str] = field(default_factory=Counter)
    metrics: dict[str, NumericStats] = field(default_factory=dict)
    input_files: list[str] = field(default_factory=list)
    file_errors: list[str] = field(default_factory=list)

    def add_file(self, path_text: str) -> None:
        if path_text not in self.input_files:
            self.input_files.append(path_text)

    def add_file_error(self, message: str, status: str) -> None:
        self.file_errors.append(message)
        self.status_counts[status] += 1

    def add_row(self, row: Mapping[str, Any], axis: AxisSpec) -> None:
        self.count += 1
        score_value = first_number(row, axis.score_fields)
        if score_value is not None:
            self.score.add(score_value)

        status = first_text(row, axis.status_fields)
        if status is None:
            status = "ok" if score_value is not None else "missing_status"
        self.status_counts[status] += 1

        for metric_name, fields_for_metric in axis.metric_fields.items():
            value = first_number(row, fields_for_metric)
            if value is None:
                continue
            self.metrics.setdefault(metric_name, NumericStats()).add(value)

    def to_json(self, axis: AxisSpec) -> dict[str, Any]:
        configured_metrics = tuple(axis.metric_fields)
        metric_stats = {
            name: self.metrics.get(name, NumericStats()).to_json()
            for name in configured_metrics
        }
        return {
            "count": self.count,
            "score_count": self.score.count,
            "mean": self.score.mean,
            "min": self.score.min_value,
            "max": self.score.max_value,
            "status_counts": dict(sorted(self.status_counts.items())),
            "metric_slots": {
                "configured": list(configured_metrics),
                "with_values": [
                    name
                    for name in configured_metrics
                    if self.metrics.get(name, NumericStats()).count > 0
                ],
                "stats": metric_stats,
            },
            "input_files": self.input_files,
            "file_errors": self.file_errors,
        }


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as json_exc:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"{path} is not JSON-compatible YAML, and PyYAML is not installed"
            ) from json_exc
        data = yaml.safe_load(text)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: config root must be a mapping") from json_exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: config root must be an object/mapping")
    return data


def iter_metric_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
                yield row
        return

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        for index, row in enumerate(data, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: item {index} must be an object")
            yield row
        return

    if not isinstance(data, dict):
        raise ValueError(f"{path}: JSON root must be an object or list")

    for key in ("rows", "records", "examples", "data", "results"):
        value = data.get(key)
        if isinstance(value, list):
            for index, row in enumerate(value, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{key}[{index}] must be an object")
                yield row
            return

    yield data


def build_arms(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms_value = config.get("arms") or DEFAULT_ARMS
    arms: list[dict[str, Any]] = []
    if isinstance(arms_value, Mapping):
        for name, value in arms_value.items():
            entry = dict(value) if isinstance(value, Mapping) else {}
            entry.setdefault("name", str(name))
            arms.append(entry)
    elif isinstance(arms_value, list | tuple):
        for value in arms_value:
            if isinstance(value, Mapping):
                entry = dict(value)
                if "name" not in entry:
                    raise ValueError("arm entries must include name")
                arms.append(entry)
            else:
                arms.append({"name": str(value)})
    else:
        raise ValueError("config.arms must be a list or mapping")

    names = [str(arm["name"]) for arm in arms]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate arm names: {', '.join(duplicates)}")
    return arms


def build_axes(config: Mapping[str, Any]) -> list[AxisSpec]:
    axes_value = config.get("axes") or DEFAULT_AXES
    raw_axes: list[Any]
    if isinstance(axes_value, Mapping):
        raw_axes = []
        for name, value in axes_value.items():
            entry = dict(value) if isinstance(value, Mapping) else {}
            entry.setdefault("name", str(name))
            raw_axes.append(entry)
    elif isinstance(axes_value, list | tuple):
        raw_axes = list(axes_value)
    else:
        raise ValueError("config.axes must be a list or mapping")

    axes: list[AxisSpec] = []
    for raw_axis in raw_axes:
        if isinstance(raw_axis, Mapping):
            name = str(raw_axis.get("name") or "").strip()
            if not name:
                raise ValueError("axis entries must include name")
            label = str(raw_axis.get("label") or name)
            score_fields = tuple(
                str(field)
                for field in ensure_list(raw_axis.get(
                    "score_fields",
                    DEFAULT_SCORE_FIELDS_BY_AXIS.get(name, ("score", "mean_score")),
                ))
            )
            status_fields = tuple(
                str(field)
                for field in ensure_list(raw_axis.get("status_fields", DEFAULT_STATUS_FIELDS))
            )
            metric_fields = build_metric_fields(name, raw_axis)
        else:
            name = str(raw_axis)
            label = name
            score_fields = DEFAULT_SCORE_FIELDS_BY_AXIS.get(name, ("score", "mean_score"))
            status_fields = DEFAULT_STATUS_FIELDS
            metric_fields = build_metric_fields(name, {})
        axes.append(
            AxisSpec(
                name=name,
                label=label,
                score_fields=tuple(score_fields),
                status_fields=tuple(status_fields),
                metric_fields=metric_fields,
            )
        )

    names = [axis.name for axis in axes]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate axis names: {', '.join(duplicates)}")
    return axes


def build_metric_fields(axis_name: str, axis_config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    explicit = axis_config.get("metric_fields")
    if isinstance(explicit, Mapping):
        return {
            str(metric): tuple(str(field) for field in ensure_list(fields))
            for metric, fields in explicit.items()
        }

    metrics = axis_config.get("future_metrics")
    if metrics is None and axis_name == "charactereval":
        metrics = CHARACTEREVAL_13_METRICS
    if metrics is None:
        metrics = ()

    return {
        str(metric): default_metric_paths(axis_name, str(metric))
        for metric in ensure_list(metrics)
    }


def default_metric_paths(axis_name: str, metric_name: str) -> tuple[str, ...]:
    if axis_name == "charactereval":
        return (
            f"charactereval.metrics.{metric_name}",
            f"charactereval.{metric_name}",
            f"metrics.{metric_name}",
            metric_name,
        )
    return (
        f"{axis_name}.metrics.{metric_name}",
        f"{axis_name}.{metric_name}",
        f"metrics.{metric_name}",
        metric_name,
    )


def collect_input_paths(
    config: Mapping[str, Any],
    arms: list[dict[str, Any]],
    axes: list[AxisSpec],
) -> dict[tuple[str, str], list[str]]:
    paths: dict[tuple[str, str], list[str]] = {
        (str(arm["name"]), axis.name): []
        for arm in arms
        for axis in axes
    }

    axes_config = config.get("axes") or []
    for raw_axis in normalize_config_entries(axes_config):
        axis_name = str(raw_axis.get("name") or "").strip()
        if not axis_name:
            continue
        inputs = raw_axis.get("inputs", {})
        if not isinstance(inputs, Mapping):
            continue
        for arm_name, value in inputs.items():
            paths.setdefault((str(arm_name), axis_name), []).extend(path_texts(value))

    for arm in arms:
        arm_name = str(arm["name"])
        results = arm.get("results", {})
        if not isinstance(results, Mapping):
            continue
        for axis_name, value in results.items():
            paths.setdefault((arm_name, str(axis_name)), []).extend(path_texts(value))

    result_files = config.get("result_files", [])
    for item in ensure_list(result_files):
        if not isinstance(item, Mapping):
            raise ValueError("result_files entries must be mappings")
        arm_name = str(item.get("arm") or "").strip()
        axis_name = str(item.get("axis") or "").strip()
        if not arm_name or not axis_name:
            raise ValueError("result_files entries need arm and axis")
        paths.setdefault((arm_name, axis_name), []).extend(path_texts(item))

    return {key: dedupe_preserving_order(value) for key, value in paths.items()}


def normalize_config_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        entries = []
        for name, item in value.items():
            entry = dict(item) if isinstance(item, Mapping) else {}
            entry.setdefault("name", str(name))
            entries.append(entry)
        return entries
    if isinstance(value, list | tuple):
        entries = []
        for item in value:
            if isinstance(item, Mapping):
                entries.append(dict(item))
        return entries
    return []


def path_texts(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Path):
        return [str(value)]
    if isinstance(value, Mapping):
        if "path" in value:
            return path_texts(value.get("path"))
        if "paths" in value:
            return path_texts(value.get("paths"))
        return []
    if isinstance(value, list | tuple):
        paths: list[str] = []
        for item in value:
            paths.extend(path_texts(item))
        return paths
    raise ValueError(f"unsupported path entry: {value!r}")


def ensure_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_path(path_text: str, config_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() or looks_like_absolute_path(path_text):
        return path
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    return (base_dir / path).resolve()


def looks_like_absolute_path(path_text: str) -> bool:
    return (
        path_text.startswith("/")
        or path_text.startswith("\\")
        or (len(path_text) >= 3 and path_text[1:3] in (":\\", ":/"))
    )


def get_path(record: Mapping[str, Any], path: str) -> Any:
    if path in record:
        return record[path]

    current: Any = record
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return MISSING
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return MISSING
            current = current[index]
        else:
            return MISSING
    return current


def first_number(record: Mapping[str, Any], paths: Iterable[str]) -> float | None:
    for path in paths:
        value = get_path(record, path)
        number = as_number(value)
        if number is not None:
            return number
    return None


def first_text(record: Mapping[str, Any], paths: Iterable[str]) -> str | None:
    for path in paths:
        value = get_path(record, path)
        if value is MISSING or value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def as_number(value: Any) -> float | None:
    if value is MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None

    if math.isnan(number) or math.isinf(number):
        return None
    return number


def aggregate(config_path: Path, strict_missing: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    arms = build_arms(config)
    axes = build_axes(config)
    inputs = collect_input_paths(config, arms, axes)
    allow_missing = bool(config.get("allow_missing_inputs", False)) and not strict_missing

    buckets: dict[tuple[str, str], AggregateBucket] = {
        (str(arm["name"]), axis.name): AggregateBucket()
        for arm in arms
        for axis in axes
    }
    axis_by_name = {axis.name: axis for axis in axes}

    for (arm_name, axis_name), path_list in inputs.items():
        axis = axis_by_name.get(axis_name)
        if axis is None:
            continue
        bucket = buckets.setdefault((arm_name, axis_name), AggregateBucket())
        for path_text in path_list:
            bucket.add_file(path_text)
            resolved = resolve_path(path_text, config_path)
            try:
                rows = list(iter_metric_rows(resolved))
            except FileNotFoundError as exc:
                message = f"{path_text}: file not found"
                if not allow_missing:
                    raise FileNotFoundError(message) from exc
                bucket.add_file_error(message, "missing_file")
                continue
            except OSError as exc:
                message = f"{path_text}: {exc}"
                if not allow_missing:
                    raise OSError(message) from exc
                bucket.add_file_error(message, "file_error")
                continue
            for row in rows:
                bucket.add_row(row, axis)

    results = {
        arm_name: {
            axis.name: buckets[(arm_name, axis.name)].to_json(axis)
            for axis in axes
        }
        for arm_name in (str(arm["name"]) for arm in arms)
    }

    add_deltas(results, axes, "Base")
    baseline_arm = str(config.get("baseline_arm") or DEFAULT_BASELINE_ARM)
    add_deltas(results, axes, baseline_arm)

    return {
        "run_name": config.get("run_name", "four_arm_eval"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "baseline_arm": baseline_arm,
        "arms": arms,
        "axes": [
            {
                "name": axis.name,
                "label": axis.label,
                "score_fields": list(axis.score_fields),
                "status_fields": list(axis.status_fields),
                "metric_fields": {
                    name: list(fields)
                    for name, fields in axis.metric_fields.items()
                },
            }
            for axis in axes
        ],
        "caveats": ensure_list(config.get("caveats")),
        "claim_guardrails": ensure_list(config.get("claim_guardrails")),
        "results": results,
    }


def add_deltas(results: dict[str, dict[str, dict[str, Any]]], axes: list[AxisSpec], baseline_arm: str) -> None:
    suffix = f"delta_vs_{baseline_arm}"
    for axis in axes:
        baseline = results.get(baseline_arm, {}).get(axis.name, {}).get("mean")
        for axis_results in results.values():
            row = axis_results.get(axis.name)
            if row is None:
                continue
            mean = row.get("mean")
            row[suffix] = (
                mean - baseline
                if isinstance(mean, int | float) and isinstance(baseline, int | float)
                else None
            )


def make_markdown(summary: Mapping[str, Any]) -> str:
    lines: list[str] = [f"# {summary['run_name']}"]

    caveats = [str(item) for item in ensure_list(summary.get("caveats")) if str(item).strip()]
    guardrails = [
        str(item)
        for item in ensure_list(summary.get("claim_guardrails"))
        if str(item).strip()
    ]
    for item in caveats + guardrails:
        lines.append(f"> {item}")

    lines.extend(
        [
            "",
            "| arm | axis | n | score_n | mean | delta_vs_SFT | delta_vs_Base | status_counts | metric_slots |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|",
        ]
    )

    axes = [axis["name"] for axis in summary["axes"]]
    for arm in summary["arms"]:
        arm_name = str(arm["name"])
        for axis_name in axes:
            row = summary["results"][arm_name][axis_name]
            metric_slots = row.get("metric_slots", {})
            configured = metric_slots.get("configured", [])
            with_values = metric_slots.get("with_values", [])
            lines.append(
                "| "
                + " | ".join(
                    [
                        arm_name,
                        axis_name,
                        str(row["count"]),
                        str(row["score_count"]),
                        format_number(row.get("mean")),
                        format_number(row.get("delta_vs_SFT")),
                        format_number(row.get("delta_vs_Base")),
                        format_counts(row.get("status_counts", {})),
                        f"{len(with_values)}/{len(configured)}",
                    ]
                )
                + " |"
            )

    return "\n".join(lines) + "\n"


def format_number(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6g}"
    return "NA"


def format_counts(counts: Mapping[str, Any]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def output_path(config: Mapping[str, Any], key: str) -> Path | None:
    outputs = config.get("outputs", {})
    if not isinstance(outputs, Mapping):
        return None
    value = outputs.get(key) or outputs.get(f"{key}_path")
    if not value:
        return None
    return Path(str(value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate four-arm eval result files.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="do not print markdown to stdout")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    summary = aggregate(args.config, strict_missing=args.strict_missing)
    markdown = make_markdown(summary)

    config = load_config(args.config)
    markdown_out = args.markdown_out or output_path(config, "markdown")
    json_out = args.json_out or output_path(config, "json")

    if markdown_out is not None:
        write_text(markdown_out, markdown)
    if json_out is not None:
        write_text(json_out, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if not args.quiet:
        print(markdown, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
