"""Write auditable source-ledger files for reset runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "anima_source_ledger_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*") if child.is_file()))
        elif path.is_file():
            files.append(path)
    return files


def build_entry(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [Path(value) for value in args.path]
    files = collect_files(input_paths)
    if not files:
        raise FileNotFoundError(f"source ledger needs at least one existing file/path, got {input_paths}")
    if str(args.license_name).strip().lower() in {"", "unknown", "none", "todo"}:
        raise ValueError("source ledger license_name must be explicit; do not use unknown in official reset runs")
    checksums = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "source_id": args.source_id,
        "name": args.name or args.source_id,
        "url": args.url,
        "download_date_utc": args.download_date_utc or utc_now(),
        "snapshot_or_commit": args.snapshot_or_commit,
        "license": {
            "name": args.license_name,
            "url_or_path": args.license_url_or_path,
            "redistribution": args.redistribution,
        },
        "project_use": args.project_use,
        "public_artifact_policy": args.public_artifact_policy,
        "notes": args.notes,
        "input_paths": [str(path) for path in input_paths],
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in checksums),
        "checksums": checksums,
    }


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": SCHEMA_VERSION, "generated_at_utc": utc_now(), "entries": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path} has unsupported schema={data.get('schema')!r}")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path} entries must be a list")
    return data


def upsert_entry(ledger: dict[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(entry["source_id"])
    entries = [item for item in ledger.get("entries", []) if item.get("source_id") != source_id]
    entries.append(dict(entry))
    ledger["entries"] = sorted(entries, key=lambda item: str(item.get("source_id")))
    ledger["generated_at_utc"] = utc_now()
    return ledger


def write_ledger(path: Path, ledger: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--url", required=True)
    parser.add_argument("--snapshot-or-commit", default="unknown")
    parser.add_argument("--download-date-utc", default="")
    parser.add_argument("--license-name", default="unknown")
    parser.add_argument("--license-url-or-path", default="unknown")
    parser.add_argument("--redistribution", default="manifest_only")
    parser.add_argument("--project-use", required=True)
    parser.add_argument("--public-artifact-policy", default="do_not_redistribute_raw")
    parser.add_argument("--notes", default="")
    parser.add_argument("--path", action="append", default=[])
    args = parser.parse_args(argv)

    ledger = load_ledger(args.output)
    entry = build_entry(args)
    write_ledger(args.output, upsert_entry(ledger, entry))
    print(
        json.dumps(
            {
                "stage": "source_ledger",
                "output": str(args.output),
                "source_id": args.source_id,
                "file_count": entry["file_count"],
                "total_bytes": entry["total_bytes"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
