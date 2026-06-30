"""LOCAL-ONLY March 7th (三月七) seed-dialogue builder.

This turns an operator-curated scene list + a local persona profile into role-play
dialogue *seeds* (each a short multi-turn context ending on a USER turn). The
seeds are then fed to ``anima.data.synth_deepseek`` to fill the verifiable-reward
gold fields. It REUSES the W2 synthesis path (same DeepSeek client) rather than
inventing a new pipeline.

RED LINE: HSR / March 7th is HoYoverse IP. All inputs and outputs are LOCAL-ONLY,
non-redistributed, non-commercial, must stay under ``work/data/hsr/`` (gitignored),
and must never be committed or served by a public demo. This module contains no
HSR data itself (only builder logic); the data lives outside the repo.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from anima.data.synth_deepseek import call_deepseek_with_retries


LOCAL_ONLY_NOTICE = (
    "HSR/March 7th data is local-only, non-redistributed, non-commercial, "
    "and must never be committed or served in the public demo."
)
DEFAULT_CHARACTER = "三月七"
DEFAULT_SOURCE_WORK = "崩坏：星穹铁道"
DEFAULT_LORE_SOURCES = [
    "wiki:zh.wikipedia.org/三月七",
    "wiki:honkai-star-rail.fandom.com/March_7th",
    "wiki:en.wikipedia.org/March_7th_(Honkai:_Star_Rail)",
]


def build_seed_prompt(profile: str, theme: str, facets: list[str], n: int) -> str:
    facet_hint = ", ".join(facets) if facets else "(unspecified)"
    return f"""你在为一个中文角色扮演角色「{DEFAULT_CHARACTER}」准备对话上下文（不是回复）。

角色档案：
{profile}

本批对话的主题/情境：{theme}
（内部参考，不要输出）该主题想触发的认知focus：{facet_hint}

请生成正好 {n} 个**彼此不同**的简短对话上下文。每个上下文是 user 与 {DEFAULT_CHARACTER} 之间的多轮对话，但**必须以 user 的发言结尾**——也就是说**不要写出 {DEFAULT_CHARACTER} 的最后一句回复**（那一句留给下游模型生成）。
要求：
- 每个上下文 1~3 轮；{n} 个之间在措辞、子情境、语气上要有变化，避免雷同。
- 全部用自然中文；user 可以是开拓者或列车同伴。
- 只与已知的星穹列车成员互动；可用 canon 昵称：杨叔(=瓦尔特·杨)、姬子姐姐、列车长(=帕姆/Pom-Pom)、丹恒、开拓者；{DEFAULT_CHARACTER}自称"三月七/三月/小三月"。**不要杜撰这些之外的新人名或专有名词**，拿不准的具体设定就不要写死。
- 保持 {DEFAULT_CHARACTER} 处在《崩坏：星穹铁道》世界观内；不要出现现实世界/游戏外/出戏内容。
- 若上下文含 {DEFAULT_CHARACTER} 的中间发言，可包含 assistant 轮，但整段必须以 user 轮收尾。

只输出一个严格的 JSON 数组，长度为 {n}，每个元素形如：
{{"conversations": [{{"role": "user", "content": "..."}}]}}
其中 conversations 的最后一个元素的 role 必须是 "user"。不要输出 markdown、不要解释。
"""


def extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, list):
        raise ValueError("DeepSeek response must be a JSON array")
    return value


def normalize_conversations(raw: Any) -> list[dict[str, str]] | None:
    """Keep only well-formed turns and require the context to end on a user turn."""
    if isinstance(raw, dict):
        raw = raw.get("conversations")
    if not isinstance(raw, list):
        return None
    turns: list[dict[str, str]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        turns.append({"role": role, "content": content})
    # Trim any trailing assistant turn so the context ends on a user turn.
    while turns and turns[-1]["role"] != "user":
        turns.pop()
    if not turns or turns[-1]["role"] != "user":
        return None
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-file", required=True, type=Path, help="Local persona profile text.")
    parser.add_argument("--scenes-file", required=True, type=Path, help="Operator-curated scenes JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Seed JSONL (local-only).")
    parser.add_argument("--character", default=DEFAULT_CHARACTER)
    parser.add_argument("--source-work", default=DEFAULT_SOURCE_WORK)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--per-scene", type=int, default=None, help="Override each scene's n.")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print one generation prompt per scene; no API calls.")
    parser.add_argument("--resume", action="store_true", help="Skip scenes whose ids already appear in the output.")
    parser.add_argument(
        "--ack-local-only",
        action="store_true",
        help="Required acknowledgement that no generated HSR data will be committed.",
    )
    args = parser.parse_args()

    if not args.ack_local_only:
        raise SystemExit("--ack-local-only is required. " + LOCAL_ONLY_NOTICE)

    import os

    profile = args.profile_file.read_text(encoding="utf-8").strip()
    scenes = json.loads(args.scenes_file.read_text(encoding="utf-8"))
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        raise SystemExit(f"{args.api_key_env} is required unless --dry-run is set")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_scene_ids: set[str] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                sid = str(json.loads(line).get("scene_id", ""))
                if sid:
                    done_scene_ids.add(sid)
    elif args.output.exists() and not args.dry_run:
        args.output.unlink()

    written = 0
    scenes_done = 0
    mode = "a" if (args.resume and args.output.exists()) else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
        for scene in scenes:
            scene_id = str(scene.get("id"))
            if scene_id in done_scene_ids:
                continue
            n = int(args.per_scene or scene.get("n", 8))
            facets = scene.get("facets", [])
            prompt = build_seed_prompt(profile, scene.get("theme", ""), facets, n)

            if args.dry_run:
                print(json.dumps({"scene_id": scene_id, "n": n, "prompt": prompt}, ensure_ascii=False))
                scenes_done += 1
                continue

            raw = call_deepseek_with_retries(prompt, api_key=api_key, model=args.model)
            try:
                items = extract_json_array(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                print(json.dumps({"scene_id": scene_id, "error": f"parse_failed: {exc}"}, ensure_ascii=False))
                continue

            kept = 0
            for k, item in enumerate(items):
                conversations = normalize_conversations(item)
                if conversations is None:
                    continue
                seed = {
                    "id": f"hsr_{scene_id}_{kept:02d}",
                    "scene_id": scene_id,
                    "character": args.character,
                    "source_work": args.source_work,
                    "profile": profile,
                    "conversations": conversations,
                    "facets_hint": facets,
                    "lore_sources": DEFAULT_LORE_SOURCES,
                }
                handle.write(json.dumps(seed, ensure_ascii=False, sort_keys=True) + "\n")
                kept += 1
                written += 1
            scenes_done += 1
            print(json.dumps({"scene_id": scene_id, "requested": n, "kept": kept}, ensure_ascii=False))

    print(json.dumps(
        {"stage": "build_hsr_march7th_seeds", "scenes": scenes_done, "seeds_written": written, "output": str(args.output)},
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
