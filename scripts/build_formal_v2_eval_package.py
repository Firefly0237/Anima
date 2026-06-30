"""Build the formal-v2 eval-first upload zip."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT_FILES = (
    "AGENT.md",
    "DATA.md",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
)

CONFIG_FILES = (
    "configs/formal_v2_eval.json",
)

DOC_FILES = (
    "docs/AGENTS.md",
    "docs/CAREER_DELIVERY_DESIGN.md",
    "docs/FORMAL_V1_ROLEBENCH_RUNBOOK.md",
    "docs/FORMAL_V2_EVAL_FIRST_DESIGN.md",
    "docs/FORMAL_V2_EVAL_RUNBOOK.md",
    "docs/OFFICIAL_REPRO_METHOD_BOUNDARY.md",
    "docs/OFFICIAL_REPRO_RESET_PLAN.md",
    "docs/PLAN_6WEEK_OVERVIEW.md",
    "docs/RESEARCH_DOSSIER.md",
    "docs/SESSION_BOOTSTRAP.md",
    "docs/SKILLS_AND_EXPERIENCE_LOG.md",
    "docs/PROGRESS_LOG.md",
    "docs/report/FORMAL_V1_ROLEBENCH_20260628_SUMMARY.md",
)

INCLUDE_DIRS = ("scripts", "src", "tests")
EXCLUDED_SUFFIXES = (".pyc", ".pyo")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
REQUIRED_FILES = ROOT_FILES + CONFIG_FILES + DOC_FILES + (
    "scripts/formal_v2_eval_server_run.sh",
    "scripts/download_formal_v2_eval_sources.sh",
    "scripts/featurize_setup.sh",
    "src/anima/eval/build_w4_eval_slices.py",
    "src/anima/eval/generate_rollouts.py",
    "src/anima/eval/run_charactereval.py",
    "src/anima/eval/run_socialbench.py",
    "src/anima/eval/run_ceval.py",
    "src/anima/eval/aggregate_results.py",
    "src/anima/data/source_ledger.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_package_files(root: Path) -> list[Path]:
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"required package files missing: {missing}")
    files = [root / rel for rel in ROOT_FILES + CONFIG_FILES + DOC_FILES]
    for dirname in INCLUDE_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel_parts = set(path.relative_to(root).parts)
            if rel_parts & EXCLUDED_PARTS:
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            files.append(path)
    return sorted(set(files), key=lambda item: item.as_posix())


def assert_safe_arcname(name: str) -> None:
    parts = Path(name).parts
    if name.startswith("/") or ".." in parts:
        raise ValueError(f"unsafe zip entry: {name}")
    if parts and parts[0] in {"work", "backups", ".git"}:
        raise ValueError(f"forbidden zip entry: {name}")
    if name.endswith(".safetensors"):
        raise ValueError(f"model weight should not be packaged: {name}")


def build_zip(root: Path, output: Path) -> dict[str, object]:
    files = iter_package_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(root).as_posix()
            assert_safe_arcname(arcname)
            zf.write(path, arcname=arcname)
    digest = sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {
        "output": str(output),
        "sha256": digest,
        "sha256_sidecar": str(sidecar),
        "files": len(files),
        "bytes": output.stat().st_size,
        "entries": [path.relative_to(root).as_posix() for path in files],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("formal_v2_eval_package.zip"))
    parser.add_argument("--manifest-json", type=Path, default=Path("formal_v2_eval_package.manifest.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    summary = build_zip(root, args.output.resolve())
    if args.manifest_json:
        args.manifest_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: summary[key] for key in ("output", "sha256", "files", "bytes")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
