import argparse
import json

import pytest

from anima.data.build_formal_rolebench import build_rows, partition_by_key, split_key
from anima.data.source_ledger import build_entry, load_ledger, upsert_entry, write_ledger


def _write_rolebench_like(path):
    rows = [
        {
            "id": "a1",
            "character": "甲",
            "source_work": "作品A",
            "profile": "甲的角色卡。",
            "question": "你是谁？",
            "generated": ["我是甲。"],
        },
        {
            "id": "a2",
            "character": "甲",
            "source_work": "作品A",
            "profile": "甲的角色卡。",
            "question": "你怎么看朋友？",
            "generated": ["朋友当以诚相待。"],
        },
        {
            "id": "b1",
            "character": "乙",
            "source_work": "作品B",
            "profile": "乙的角色卡。",
            "question": "今天心情如何？",
            "answer": "尚可。",
        },
        {
            "id": "c1",
            "character": "丙",
            "source_work": "作品C",
            "profile": "丙的角色卡。",
            "question": "为何出发？",
            "answer": "为寻旧约。",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_formal_rolebench_builds_valid_rows_and_disjoint_keys(tmp_path):
    raw = tmp_path / "rolebench.jsonl"
    _write_rolebench_like(raw)

    rows, skipped = build_rows([raw], source="RoleBench", id_prefix="formal")
    assert skipped == {}
    assert len(rows) == 4
    assert rows[0]["reference_answer"] == "我是甲。"
    assert rows[0]["label_source"] == "deterministic_bootstrap_not_human_gold"

    train_keys, heldout_keys = partition_by_key(rows, heldout_fraction=0.34)
    assert train_keys
    assert heldout_keys
    assert train_keys.isdisjoint(heldout_keys)
    assert {split_key(row) for row in rows} == train_keys | heldout_keys


def test_formal_rolebench_requires_explicit_source_ledger_license(tmp_path):
    raw = tmp_path / "rolebench.jsonl"
    _write_rolebench_like(raw)
    ledger = tmp_path / "source_ledger.json"
    args = argparse.Namespace(
        path=[str(raw)],
        source_id="RoleBench",
        name="RoleBench",
        url="https://huggingface.co/datasets/ZenMoore/RoleBench",
        download_date_utc="2026-06-28T00:00:00Z",
        snapshot_or_commit="test",
        license_name="unknown",
        license_url_or_path="unknown",
        redistribution="manifest_only",
        project_use="formal reproduction",
        public_artifact_policy="do_not_redistribute_raw_rows",
        notes="unit test",
    )

    with pytest.raises(ValueError):
        entry = build_entry(args)
        write_ledger(ledger, upsert_entry(load_ledger(ledger), entry))
