"""Offline DeepSeek synthesis helpers for reward records.

This module is deliberately a batch/offline utility. It must never be imported
from a training loop to score policy samples. Use ``--dry-run`` first to inspect
prompts, then run a small approved batch before any 1k-record synthesis.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request


FOCUS_LABELS = (
    "Knowledge",
    "Style",
    "Worldview",
    "Emotion",
    "Empathetic",
    "Engagement",
    "Human_Like",
    "Extension",
    "Memory",
    "Safety",
)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(path) if row.get("id") is not None}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def conversation_text(conversations: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for turn in conversations:
        role = turn.get("role", "unknown")
        content = str(turn.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def build_synthesis_prompt(seed: dict[str, Any], include_rejected: bool = False) -> str:
    labels = ", ".join(FOCUS_LABELS)
    source_answer = truncate_text(str(seed.get("reference_answer") or "").strip(), max_chars=1200)
    source_answer_section = (
        "\nExisting source answer for grounding. Use it as evidence, but improve "
        "persona fidelity and wording when needed; do not copy it mechanically:\n"
        f"{source_answer}\n"
        if source_answer
        else "\nExisting source answer for grounding: <none provided>\n"
    )
    rejected_field = (
        '\n- "rejected_answer": a deliberately degraded/off-character Chinese answer\n'
        '- "rejected_strategy": one of style_flattening, persona_swap, lore_violation, focus_mismatch\n'
        if include_rejected
        else '\n- "rejected_answer": null\n- "rejected_strategy": null\n'
    )
    return f"""You are preparing verifiable reward data for Chinese role-play RL.

Use ONLY these focus labels: {labels}.
Return one strict JSON object, no markdown, with:
- "gold_focus": a non-empty list using only the allowed labels
- "gold_focus_attr": concise Chinese attribute text explaining the selected focus
- "reference_answer": a plausible in-character Chinese reply
{rejected_field}

Use Chinese for all generated text fields. The reference_answer should sound like the character replying to the user, not like a generic assistant. The gold_focus_attr should describe what the answer must activate, and the final reference_answer must not simply restate gold_focus_attr.

Character: {seed.get("character", "")}
Source work: {seed.get("source_work", "")}
Profile:
{seed.get("profile", "")}

Dialogue:
{conversation_text(seed.get("conversations", []))}
{source_answer_section}
"""


def truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n...[truncated]"


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("DeepSeek response JSON must be an object")
    return value


def call_deepseek(
    prompt: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: int = 120,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        base_url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def call_deepseek_with_retries(
    prompt: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> str:
    last_error: BaseException | None = None
    for attempt in range(1, retries + 2):
        try:
            return call_deepseek(prompt, api_key=api_key, model=model, base_url=base_url)
        except (
            TimeoutError,
            http.client.HTTPException,
            error.URLError,
            error.HTTPError,
        ) as exc:
            last_error = exc
            if attempt > retries:
                break
            time.sleep(retry_sleep_s * attempt)
    raise RuntimeError(f"DeepSeek request failed after {retries + 1} attempt(s): {last_error}") from last_error


def merge_synthesis(seed: dict[str, Any], synth: dict[str, Any], *, model: str) -> dict[str, Any]:
    rejected = synth.get("rejected_answer")
    rejected_strategy = synth.get("rejected_strategy")
    synth_meta = {
        "generator": model,
        "prompt_id": "synth_focus_v1",
        "lore_sources": seed.get("lore_sources", []),
        "rejected_strategy": rejected_strategy,
        "human_reviewed": False,
    }
    return {
        **seed,
        "gold_focus": synth.get("gold_focus"),
        "gold_focus_attr": synth.get("gold_focus_attr"),
        "reference_answer": synth.get("reference_answer"),
        "rejected_answer": rejected,
        "synth_meta": synth_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Seed JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Synthesized record JSONL.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts instead of calling API.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3, help="Retries per DeepSeek request.")
    parser.add_argument("--retry-sleep-s", type=float, default=2.0, help="Base retry sleep in seconds.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing rows and skip ids already present in the output JSONL.",
    )
    args = parser.parse_args()

    seeds = read_jsonl(args.input)
    if args.max_records is not None:
        seeds = seeds[: args.max_records]

    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        raise RuntimeError(f"{args.api_key_env} is required unless --dry-run is set")

    if not args.resume and args.output.exists():
        args.output.unlink()

    skip_ids = existing_ids(args.output) if args.resume else set()
    written_this_run = 0
    skipped_existing = 0

    for seed in seeds:
        seed_id = str(seed.get("id"))
        if seed_id in skip_ids:
            skipped_existing += 1
            continue

        prompt = build_synthesis_prompt(seed, include_rejected=args.include_rejected)
        if args.dry_run:
            row = {"id": seed.get("id"), "prompt": prompt}
            append_jsonl(args.output, row)
            written_this_run += 1
            continue
        raw = call_deepseek_with_retries(
            prompt,
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            retries=args.retries,
            retry_sleep_s=args.retry_sleep_s,
        )
        append_jsonl(args.output, merge_synthesis(seed, extract_json_object(raw), model=args.model))
        written_this_run += 1
        if args.sleep_s:
            time.sleep(args.sleep_s)

    total_records = count_jsonl(args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "records": total_records,
                "written_this_run": written_this_run,
                "skipped_existing": skipped_existing,
                "dry_run": args.dry_run,
                "include_rejected": args.include_rejected,
                "model": args.model,
                "resume": args.resume,
                "retries": args.retries,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
