import pytest

from anima.rewards.attribute_bleu import CHRF_METRIC_NAME, attribute_reward, attribute_score


def _completion(attr: str) -> str:
    return (
        "<think><focus>Knowledge, Style</focus>"
        f"<focus_attr>{attr}</focus_attr></think>"
        "\\boxed{哈哈，且来共饮一杯！}"
    )


def test_attribute_identical_chinese_chrf_scores_one():
    gold = "豪放洒脱，好酒，长于七言"

    assert "chrf" in CHRF_METRIC_NAME
    assert attribute_score(_completion(gold), gold) == 1.0


def test_attribute_chinese_near_match_beats_unrelated_text():
    gold = "豪放洒脱，好酒，长于七言"
    near = attribute_score(_completion("豪放洒脱，爱饮酒，擅长七言"), gold)
    unrelated = attribute_score(_completion("冷静克制，机械回答，避免闲谈"), gold)

    assert near > unrelated
    assert near > 0.30
    assert unrelated < 0.25


def test_attribute_missing_focus_attr_scores_zero():
    completion = "<think><focus>Knowledge</focus></think>\\boxed{好啊！}"

    assert attribute_score(completion, "豪放洒脱") == 0.0


def test_attribute_reward_is_trl_batch_compatible():
    completions = [_completion("豪放洒脱，好酒"), _completion("温柔安慰用户")]
    scores = attribute_reward(completions, gold_focus_attr=["豪放洒脱，好酒", "冷淡拒绝"])

    assert scores[0] == 1.0
    assert scores[1] == pytest.approx(attribute_score(completions[1], "冷淡拒绝"))
    assert scores[1] < scores[0]
