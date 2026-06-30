import json

from anima.data.source_ledger import build_entry, load_ledger, upsert_entry, write_ledger


class Args:
    source_id = "RoleBench"
    name = "RoleBench"
    url = "https://huggingface.co/datasets/ZenMoore/RoleBench"
    download_date_utc = "2026-06-28T00:00:00Z"
    snapshot_or_commit = "test"
    license_name = "Apache-2.0"
    license_url_or_path = "dataset card"
    redistribution = "manifest_only"
    project_use = "SFT smoke"
    public_artifact_policy = "do_not_redistribute_raw"
    notes = "unit test"

    def __init__(self, path):
        self.path = [str(path)]


def test_source_ledger_writes_checksums(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"id":"x"}\n', encoding="utf-8")
    entry = build_entry(Args(raw))

    assert entry["source_id"] == "RoleBench"
    assert entry["file_count"] == 1
    assert entry["total_bytes"] == raw.stat().st_size
    assert len(entry["checksums"][0]["sha256"]) == 64

    path = tmp_path / "ledger.json"
    write_ledger(path, upsert_entry(load_ledger(path), entry))
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema"] == "anima_source_ledger_v1"
    assert data["entries"][0]["source_id"] == "RoleBench"
