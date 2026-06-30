import pytest

from anima.rewards.focus_overlap import (
    FOCUS_OVERLAP_METRIC,
    focus_overlap_reward,
    focus_overlap_score,
    focus_strict_em,
    score_focus_overlap,
)


def _completion(focus: str) -> str:
    return (
        f"<think><focus>{focus}</focus>"
        "<focus_attr>豪放洒脱，好酒</focus_attr></think>"
        "\\boxed{哈哈，且来共饮一杯！}"
    )


def test_focus_perfect_multilabel_scores_one_and_strict_em_one():
    completion = _completion("Knowledge, Style")

    result = score_focus_overlap(completion, ["Style", "Knowledge"])

    assert FOCUS_OVERLAP_METRIC == "f1"
    assert result.reward == 1.0
    assert result.strict_em == 1.0
    assert focus_overlap_score(completion, ["Knowledge", "Style"]) == 1.0
    assert focus_strict_em(completion, ["Knowledge", "Style"]) == 1.0


def test_focus_partial_overlap_uses_graded_f1_not_strict_em():
    completion = _completion("Knowledge, Style")

    result = score_focus_overlap(completion, ["Knowledge", "Emotion"])

    assert result.reward == pytest.approx(0.5)
    assert result.strict_em == 0.0


def test_focus_empty_or_missing_scores_zero():
    assert focus_overlap_score(_completion(""), ["Knowledge"]) == 0.0
    assert focus_overlap_score("普通回答，没有结构", ["Knowledge"]) == 0.0


def test_focus_illegal_label_does_not_match_and_breaks_strict_em():
    result = score_focus_overlap(_completion("Knowledge, LoreHack"), ["Knowledge"])

    assert result.reward == pytest.approx(2 / 3)
    assert result.strict_em == 0.0
    assert result.illegal_predicted == ("LoreHack",)


def test_focus_reward_is_trl_batch_compatible():
    completions = [_completion("Knowledge"), _completion("Style, Emotion")]
    scores = focus_overlap_reward(
        completions,
        gold_focus=[["Knowledge"], ["Style", "Safety"]],
    )

    assert scores[0] == 1.0
    assert scores[1] == pytest.approx(0.5)
