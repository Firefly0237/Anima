"""Write claim-boundary files into run artifact directories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CLAIMS = (
    "This run is part of the official-first reproduction reset.",
    "Do not claim GRPO/DPO/SFT quality superiority without the aggregate eval table.",
    "Do not treat output trimming or post-hoc cleaning as a reported metric fix.",
    "Data source and license status must be read from the source ledger.",
    "Server artifacts are not durable until downloaded locally and SHA256-verified.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def boundary_payload(run_name: str, stage: str, notes: str = "") -> dict[str, Any]:
    return {
        "run_name": run_name,
        "stage": stage,
        "created_at_utc": utc_now(),
        "claim_boundary": list(DEFAULT_CLAIMS),
        "notes": notes,
    }


def write_boundary(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "CLAIM_BOUNDARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"Run: {payload['run_name']}",
        f"Stage: {payload['stage']}",
        f"Created UTC: {payload['created_at_utc']}",
        "",
        "Claim Boundary:",
    ]
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    if payload.get("notes"):
        lines.extend(["", "Notes:", str(payload["notes"])])
    (output_dir / "CLAIM_BOUNDARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    payload = boundary_payload(args.run_name, args.stage, notes=args.notes)
    write_boundary(args.output_dir, payload)
    print(json.dumps({"stage": "claim_boundary", "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
