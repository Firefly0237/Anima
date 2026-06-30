import pytest

from anima.rewards.combine import (
    PINNED_REWARD_METRICS,
    combined_reward,
    component_rewards,
    get_reward_funcs,
    get_reward_weights,
)
from anima.rewards.focus_overlap import focus_overlap_score
from anima.rewards.format_reward import format_reward, format_score
from anima.rewards.reference_bleu import REFERENCE_METRIC_NAME, reference_reward, reference_score


def _completion(answer: str, *, focus: str = "Knowledge, Style", attr: str = "豪放洒脱，好酒") -> str:
    return (
        f"<think><focus>{focus}</focus>"
        f"<focus_attr>{attr}</focus_attr></think>"
        f"\\boxed{{{answer}}}"
    )


def test_reference_identical_chinese_boxed_answer_scores_one():
    gold = "哈哈，且来共饮一杯！"

    assert "chrf" in REFERENCE_METRIC_NAME
    assert reference_score(_completion(gold), gold) == 1.0


def test_reference_chinese_near_match_beats_too_short_answer():
    gold = "哈哈，且来共饮一杯！"
    near = reference_score(_completion("哈哈，且来共饮一杯吧！"), gold)
    too_short = reference_score(_completion("好"), gold)

    assert near > too_short
    assert near > 0.55
    assert too_short < 0.25


def test_reference_malformed_missing_box_scores_zero_and_format_zero():
    malformed = "<think><focus>Knowledge</focus><focus_attr>豪放</focus_attr></think>普通回答"

    assert reference_score(malformed, "哈哈，且来共饮一杯！") == 0.0
    assert format_score(malformed) == 0.0


def test_format_reward_scores_well_formed_output_only():
    good = _completion("哈哈，且来共饮一杯！")
    illegal = _completion("哈哈！", focus="Knowledge, MadeUp")

    assert format_score(good) == 1.0
    assert format_score(illegal) == 0.0
    assert format_reward([good, illegal]) == [1.0, 0.0]


def test_format_reward_rejects_repeated_structure_after_box():
    repeated = (
        _completion("第一答")
        + "\n<think><focus>Engagement</focus><focus_attr>继续</focus_attr></think>"
        + "\\boxed{第二答}"
    )

    assert format_score(repeated) == 0.0


def test_format_reward_rejects_garbage_tail_after_box_but_allows_eos():
    good_with_eos = _completion("哈哈，且来共饮一杯！") + "<|im_end|>"
    bad_tail = _completion("哈哈，且来共饮一杯！") + "ledged"
    bad_endoftext = _completion("哈哈，且来共饮一杯！") + "<|endoftext|>"

    assert format_score(good_with_eos) == 1.0
    assert format_score(bad_tail) == 0.0
    assert format_score(bad_endoftext) == 0.0


def test_round_trip_perfect_completion_scores_focus_and_format_one():
    gold_focus = ["Knowledge", "Style"]
    gold_attr = "豪放洒脱，好酒"
    reference = "哈哈，且来共饮一杯！"
    completion = _completion(reference, attr=gold_attr)

    assert focus_overlap_score(completion, gold_focus) == 1.0
    assert format_score(completion) == 1.0
    assert reference_score(completion, reference) == 1.0


def test_combine_exposes_trl_reward_funcs_and_weights():
    funcs = get_reward_funcs()
    weights = get_reward_weights()

    assert len(funcs) == 4
    assert weights == [0.4, 0.2, 0.2, 0.2]
    assert sum(weights) == pytest.approx(1.0)
    assert PINNED_REWARD_METRICS["focus"] == "f1"
    assert "chrf" in PINNED_REWARD_METRICS["attribute"]
    assert "chrf" in PINNED_REWARD_METRICS["reference"]


def test_combined_reward_matches_weighted_components_and_trl_kwargs():
    completion = _completion("哈哈，且来共饮一杯！")
    kwargs = {
        "gold_focus": [["Knowledge", "Style"]],
        "gold_focus_attr": ["豪放洒脱，好酒"],
        "reference_answer": ["哈哈，且来共饮一杯！"],
    }

    components = component_rewards([completion], **kwargs)
    score = combined_reward([completion], **kwargs)[0]
    func_scores = [func([completion], **kwargs)[0] for func in get_reward_funcs()]

    assert components == {
        "focus": [1.0],
        "attribute": [1.0],
        "reference": [1.0],
        "format": [1.0],
    }
    assert func_scores == [1.0, 1.0, 1.0, 1.0]
    assert score == 1.0


def test_malformed_completion_zeroes_all_reward_paths():
    malformed = _completion("哈哈，且来共饮一杯！") + "ledged"
    kwargs = {
        "gold_focus": [["Knowledge", "Style"]],
        "gold_focus_attr": ["豪放洒脱，好酒"],
        "reference_answer": ["哈哈，且来共饮一杯！"],
    }

    components = component_rewards([malformed], **kwargs)
    score = combined_reward([malformed], **kwargs)
    func_scores = [func([malformed], **kwargs)[0] for func in get_reward_funcs()]

    assert format_score(malformed) == 0.0
    assert components == {
        "focus": [0.0],
        "attribute": [0.0],
        "reference": [0.0],
        "format": [0.0],
    }
    assert score == [0.0]
    assert func_scores == [0.0, 0.0, 0.0, 0.0]


def test_reference_reward_is_trl_batch_compatible():
    completions = [_completion("哈哈，且来共饮一杯！"), _completion("好")]

    scores = reference_reward(completions, reference_answer=["哈哈，且来共饮一杯！", "哈哈，且来共饮一杯！"])

    assert scores[0] == 1.0
    assert scores[1] < scores[0]
