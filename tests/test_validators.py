from __future__ import annotations

import json
from pathlib import Path

from anima.data.make_character_split import (
    SplitRatios,
    build_character_split,
    check_character_split,
)
from anima.data import schemas
from anima.data.validators import (
    FOCUS_LABELS,
    check_split_invariants,
    dedupe_records,
    find_cross_split_near_duplicates,
    is_near_duplicate,
    make_perfect_completion,
    normalize_text,
    parse_completion,
    validate_record,
    validate_records,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reward_toy_public.jsonl"
_DEFAULT_SYNTH_META = object()


def _record(
    record_id: str = "cb_0001",
    *,
    character: str = "李白",
    source_work: str = "唐诗",
    split: str = "reward",
    source: str = "DeepSeek-synth",
    gold_focus: list[str] | None = None,
    reference_answer: str = "哈哈，且来共饮一杯，再把月色写进诗里！",
    rejected_answer: str | None = "我是一个普通助手，无法用李白的口吻回答。",
    synth_meta: dict | None | object = _DEFAULT_SYNTH_META,
) -> dict:
    if synth_meta is _DEFAULT_SYNTH_META:
        synth_meta = {
            "generator": "deepseek-v4-flash",
            "prompt_id": "synth_focus_v1",
            "lore_sources": ["fixture:poem"],
            "rejected_strategy": "style_flattening" if rejected_answer else None,
            "human_reviewed": False,
        }
    return {
        "id": record_id,
        "character": character,
        "source_work": source_work,
        "character_cluster": 0,
        "profile": f"{character}角色卡：中文角色，重视原作设定与说话风格。",
        "conversations": [
            {"role": "user", "content": "今日月色很好，你想说些什么？"},
            {"role": "assistant", "content": "月下正宜高歌。"},
            {"role": "user", "content": "那就用你的风格回答我。"},
        ],
        "gold_focus": gold_focus or ["Knowledge", "Style"],
        "gold_focus_attr": "豪放洒脱，好酒，常以月与诗入答",
        "reference_answer": reference_answer,
        "rejected_answer": rejected_answer,
        "source": source,
        "split": split,
        "synth_meta": synth_meta,
    }


def _split_config() -> dict:
    return {
        "sft": ["林黛玉"],
        "reward": ["李白"],
        "eval_heldout": ["鲁智深", "三月七"],
        "_split_unit": "source_work",
        "_source_work_by_character": {
            "林黛玉": "红楼梦",
            "李白": "唐诗",
            "鲁智深": "水浒传",
            "三月七": "HSR(local-only)",
        },
    }


def _codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_closed_focus_label_set_matches_guide():
    assert FOCUS_LABELS == {
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
    }
    assert tuple(sorted(FOCUS_LABELS)) == tuple(sorted(schemas.FOCUS_LABELS))
    assert schemas.VALID_SPLITS == ("sft", "reward", "eval_heldout")
    assert schemas.VALID_SOURCES == (
        "CharacterBench",
        "RoleBench",
        "DeepSeek-synth",
        "HSR-canon",
        "HSR-synth",
    )


def test_schema_jsonl_loader_reads_toy_public_fixtures():
    records = schemas.load_jsonl(FIXTURE_PATH)

    assert len(records) == 4
    assert {record.id for record in records} == {
        "toy_public_0001",
        "toy_public_0002",
        "toy_public_0003",
        "toy_public_0004",
    }
    assert all(record.gold_focus for record in records)
    assert all(record.source in schemas.VALID_SOURCES for record in records)


def test_schema_rejects_invalid_focus_and_bad_dpo_metadata():
    payload = schemas.load_jsonl(FIXTURE_PATH)[0].to_dict()
    payload["gold_focus"] = ["Knowledge", "Invented_Label"]

    errors = schemas.validate_record(payload)
    assert any("illegal labels" in error for error in errors)

    dpo_payload = schemas.load_jsonl(FIXTURE_PATH)[2].to_dict()
    dpo_payload["synth_meta"]["rejected_strategy"] = None
    errors = schemas.validate_record(dpo_payload)
    assert any("rejected_strategy" in error for error in errors)


def test_schema_jsonl_round_trip_preserves_fixture_records(tmp_path: Path):
    records = schemas.load_jsonl(FIXTURE_PATH)
    destination = tmp_path / "round_trip.jsonl"

    schemas.write_jsonl(destination, records)
    loaded = schemas.load_jsonl(destination)

    assert [record.to_dict() for record in loaded] == [record.to_dict() for record in records]


def test_schema_jsonl_reports_line_number_for_invalid_record(tmp_path: Path):
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"id": "bad", "gold_focus": ["Knowledge"]}\n', encoding="utf-8")

    try:
        schemas.load_jsonl(bad_path)
    except schemas.SchemaValidationError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected SchemaValidationError")

    assert "line 1" in message
    assert "character" in message


def test_valid_record_passes_schema_parseability_provenance_and_dpo_checks():
    result = validate_record(_record(), split_config=_split_config())

    assert result.ok, result.issues

    completion = make_perfect_completion(_record())
    parsed = parse_completion(completion)
    assert parsed.well_formed
    assert set(parsed.focus) == {"Knowledge", "Style"}
    assert parsed.focus_attr == "豪放洒脱，好酒，常以月与诗入答"
    assert parsed.answer == "哈哈，且来共饮一杯，再把月色写进诗里！"


def test_schema_missing_required_field_is_rejected():
    record = _record()
    del record["reference_answer"]

    result = validate_record(record)

    assert "schema.missing" in _codes(result)
    assert not result.ok


def test_illegal_or_duplicate_focus_labels_are_rejected():
    result = validate_record(_record(gold_focus=["Knowledge", "Mood", "Knowledge"]))

    assert {"label.illegal", "label.duplicate"} <= _codes(result)


def test_reference_sanity_checks_cjk_and_gold_field_copying():
    result = validate_record(
        _record(
            reference_answer="豪放洒脱，好酒，常以月与诗入答",
            rejected_answer="平淡的中文拒绝答案。",
        )
    )

    assert "reference.equals_gold_attr" in _codes(result)

    no_cjk = validate_record(
        _record(
            record_id="cb_ascii",
            reference_answer="generic answer",
            rejected_answer="another generic answer",
        )
    )
    assert "reference.not_cjk" in _codes(no_cjk)


def test_dpo_chosen_and_rejected_must_differ_after_normalization():
    result = validate_record(
        _record(
            reference_answer="哈哈，且来共饮一杯！",
            rejected_answer="哈哈 且来 共饮 一杯",
        )
    )

    assert "dpo.chosen_equals_rejected" in _codes(result)


def test_synthetic_provenance_and_rejected_strategy_are_required():
    missing_meta = validate_record(_record(synth_meta=None))
    assert "provenance.synth_meta_missing" in _codes(missing_meta)
    assert "provenance.rejected_strategy_missing" in _codes(missing_meta)

    rejected_from_seed = validate_record(_record(source="CharacterBench", synth_meta=None))
    assert "provenance.rejected_strategy_missing" in _codes(rejected_from_seed)

    bad_meta = validate_record(
        _record(
            synth_meta={
                "generator": "deepseek-v4-flash",
                "prompt_id": "synth_focus_v1",
                "lore_sources": [],
                "human_reviewed": False,
            }
        )
    )
    assert "provenance.rejected_strategy_missing" in _codes(bad_meta)


def test_split_membership_and_hsr_eval_only_are_enforced():
    mismatch = validate_record(
        _record(character="鲁智深", source_work="水浒传", split="reward"),
        split_config=_split_config(),
    )
    assert "split.mismatch" in _codes(mismatch)

    hsr_bad = validate_record(
        _record(
            record_id="hsr_0001",
            character="三月七",
            source_work="HSR(local-only)",
            split="reward",
            source="HSR-synth",
        )
    )
    assert "leakage.hsr_not_eval_only" in _codes(hsr_bad)


def test_batch_validator_detects_character_and_source_work_leakage():
    reward = _record(record_id="cb_reward", character="角色甲", source_work="同一作品", split="reward")
    eval_record = _record(
        record_id="cb_eval",
        character="角色乙",
        source_work="同一作品",
        split="eval_heldout",
        reference_answer="我在评测集中回答一个完全不同的中文问题。",
        rejected_answer="我在评测集中给出平淡中文回答。",
    )

    report = validate_records([reward, eval_record], check_cross_split_duplicates=False)

    assert not report.ok
    assert any(issue.code == "leakage.source_work_record_split_overlap" for issue in report.batch_issues)


def test_exact_and_normalized_near_duplicate_helpers():
    assert normalize_text("  你好，World！ ") == "你好world"
    assert is_near_duplicate("你好，今天一起去喝酒！", "你好 今天 一起 去 喝酒")
    assert is_near_duplicate(
        "我们在月下饮酒作诗，谈笑之间仍要保持李白的豪放口吻。",
        "我们在月下饮酒作诗，谈笑之间要保持李白的豪放口吻。",
        threshold=0.6,
    )


def test_dedupe_records_keeps_first_duplicate_cluster_member():
    first = _record(record_id="cb_1")
    second = _record(record_id="cb_2")
    unique = _record(
        record_id="cb_3",
        character="鲁智深",
        source_work="水浒传",
        split="eval_heldout",
        reference_answer="洒家只问一句：这事可还讲个义字？",
        rejected_answer="这是一个普通中文回答。",
    )

    kept, duplicates = dedupe_records([first, second, unique])

    assert [record["id"] for record in kept] == ["cb_1", "cb_3"]
    assert [(pair.left_id, pair.right_id) for pair in duplicates] == [("cb_1", "cb_2")]


def test_cross_split_near_duplicates_report_reward_eval_collision():
    reward = _record(record_id="reward_dup", split="reward")
    eval_record = _record(
        record_id="eval_dup",
        character="鲁智深",
        source_work="水浒传",
        split="eval_heldout",
        reference_answer="哈哈 且来 共饮 一杯 再把 月色 写进 诗里",
        rejected_answer="这是一个普通中文回答。",
    )

    pairs = find_cross_split_near_duplicates([reward, eval_record], threshold=0.7)

    assert pairs
    assert pairs[0].left_id == "reward_dup"
    assert pairs[0].right_id == "eval_dup"


def test_split_invariant_checker_uses_source_work_metadata_without_records():
    split = _split_config()
    split["eval_heldout"].append("杜甫")
    split["_source_work_by_character"]["杜甫"] = "唐诗"

    issues = check_split_invariants(split)

    assert any(issue.code == "leakage.source_work_split_overlap" for issue in issues)


def test_character_split_builder_respects_existing_splits_and_checks_invariants():
    records = [
        _record(record_id="sft_1", character="林黛玉", source_work="红楼梦", split="sft"),
        _record(record_id="reward_1", character="李白", source_work="唐诗", split="reward"),
        _record(
            record_id="eval_1",
            character="鲁智深",
            source_work="水浒传",
            split="eval_heldout",
            reference_answer="洒家看这月色，也要问一句义气何在。",
            rejected_answer="普通中文回答。",
        ),
        _record(
            record_id="hsr_1",
            character="三月七",
            source_work="HSR(local-only)",
            split="eval_heldout",
            source="HSR-synth",
            reference_answer="哎呀，这么好的月色当然要拍下来啦！",
            rejected_answer="这是一个普通中文回答。",
        ),
    ]

    split = build_character_split(records)

    assert split["sft"] == ["林黛玉"]
    assert split["reward"] == ["李白"]
    assert split["eval_heldout"] == ["三月七", "鲁智深"]
    assert check_character_split(split, records) == []


def test_character_split_builder_auto_assigns_whole_source_works_deterministically():
    records = [
        _record(record_id="a1", character="甲", source_work="作品A", split="reward"),
        _record(record_id="a2", character="乙", source_work="作品A", split="reward"),
        _record(record_id="b1", character="丙", source_work="作品B", split="reward"),
        _record(record_id="c1", character="丁", source_work="作品C", split="reward"),
    ]

    split_a = build_character_split(
        records,
        ratios=SplitRatios(sft=1, reward=1, eval_heldout=1),
        respect_existing_split=False,
    )
    split_b = build_character_split(
        list(reversed(records)),
        ratios=SplitRatios(sft=1, reward=1, eval_heldout=1),
        respect_existing_split=False,
    )

    assert split_a == split_b
    work_to_splits = {}
    for split_name in ("sft", "reward", "eval_heldout"):
        for character in split_a[split_name]:
            source_work = split_a["_source_work_by_character"][character]
            work_to_splits.setdefault(source_work, set()).add(split_name)
    assert all(len(splits) == 1 for splits in work_to_splits.values())


def test_committed_example_character_split_is_small_and_valid():
    path = Path("src/anima/data/character_split.json")
    split = json.loads(path.read_text(encoding="utf-8"))

    assert set(split) >= {"sft", "reward", "eval_heldout", "_split_unit"}
    assert split["_split_unit"] == "source_work"
    assert "三月七" in split["eval_heldout"]
    assert check_split_invariants(split) == []
