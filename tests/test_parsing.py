from anima.rewards.parsing import FOCUS_LABELS, output_contract_issues, parse_completion


def test_parse_cjk_completion_extracts_all_required_segments():
    completion = (
        "<think>先判断身份与语气。"
        "<focus>Knowledge, Style</focus>"
        "<focus_attr>豪放洒脱，好酒，长于七言</focus_attr>"
        "</think>\n"
        "\\boxed{哈哈，且来共饮一杯！}"
    )

    parsed = parse_completion(completion)

    assert "Knowledge" in FOCUS_LABELS
    assert parsed.think is not None
    assert parsed.focus_labels == ("Knowledge", "Style")
    assert parsed.illegal_focus_labels == ()
    assert parsed.focus_attr == "豪放洒脱，好酒，长于七言"
    assert parsed.boxed_answer == "哈哈，且来共饮一杯！"
    assert parsed.is_well_formed is True


def test_parse_handles_multiple_focus_tags_and_nested_box_braces():
    completion = (
        "<think><focus>Knowledge</focus><focus>Emotion</focus>"
        "<focus_attr>记得旧事，也带一点兴奋</focus_attr></think>"
        "\\boxed{且看{这一杯}如何}"
    )

    parsed = parse_completion(completion)

    assert parsed.focus_labels == ("Knowledge", "Emotion")
    assert parsed.boxed_answer == "且看{这一杯}如何"
    assert parsed.is_well_formed is False
    assert "multi_focus" in output_contract_issues(parsed)


def test_malformed_missing_box_is_not_well_formed():
    parsed = parse_completion(
        "<think><focus>Knowledge</focus><focus_attr>豪放</focus_attr></think>"
    )

    assert parsed.boxed_answer is None
    assert parsed.is_well_formed is False


def test_illegal_focus_label_is_flagged_and_not_counted_as_focus():
    parsed = parse_completion(
        "<think><focus>Knowledge, LoreHack</focus>"
        "<focus_attr>豪放</focus_attr></think>\\boxed{好啊！}"
    )

    assert parsed.focus_labels == ("Knowledge",)
    assert parsed.illegal_focus_labels == ("LoreHack",)
    assert parsed.is_well_formed is False


def test_parse_accepts_trl_chat_completion_shape():
    parsed = parse_completion(
        [
            {
                "role": "assistant",
                "content": (
                    "<think><focus>Safety</focus><focus_attr>温和拒绝危险请求</focus_attr>"
                    "</think>\\boxed{这个我不能帮你做。}"
                ),
            }
        ]
    )

    assert parsed.focus_labels == ("Safety",)
    assert parsed.boxed_answer == "这个我不能帮你做。"
    assert parsed.is_well_formed is True


def test_prompt_leakage_and_prefix_are_output_contract_issues():
    parsed = parse_completion(
        "Human:\n<think><focus>Style</focus><focus_attr>保持语气</focus_attr>"
        "</think>\\boxed{好。}"
    )

    issues = output_contract_issues(parsed)

    assert "prefix_before_think" in issues
    assert "prompt_leakage" in issues
    assert parsed.is_well_formed is False
