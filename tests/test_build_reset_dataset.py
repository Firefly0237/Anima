import json

from anima.data.build_reset_dataset import build_records


def test_build_reset_dataset_from_rolebench_like_row(tmp_path):
    raw = tmp_path / "rolebench.jsonl"
    raw.write_text(
        json.dumps(
            {
                "id": "rb1",
                "character": "李白",
                "source_work": "唐诗",
                "profile": "唐代诗人。",
                "question": "今日为何饮酒？",
                "answer": "举杯邀明月。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows, skipped = build_records([raw], source="RoleBench", id_prefix="reset", max_records=4)

    assert skipped == {}
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "RoleBench"
    assert row["label_source"] == "deterministic_bootstrap_not_human_gold"
    assert row["gold_focus"] == ["Style", "Engagement"]
    assert row["reference_answer"] == "举杯邀明月。"
    assert row["conversations"] == [{"role": "user", "content": "今日为何饮酒？"}]
