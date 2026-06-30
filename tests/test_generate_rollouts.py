import pytest

from anima.eval.generate_rollouts import _decode_completion, _normalize_seed, _render_generation_prompt
from anima.train.common import ChatTemplateError


class FakeTokenizer:
    chat_template = "qwen-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert isinstance(messages, list)
        return "<QWEN_CHAT_PROMPT>"

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "  <think></think>\\boxed{答}ledged  "


class BudgetTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        parts = [
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
            for message in messages
        ]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def __call__(self, text, *, return_tensors=None, truncation=False, max_length=None):
        input_ids = text.split()
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids}


def _record():
    return {
        "id": "rollout_test",
        "character": "李白",
        "source_work": "唐诗",
        "profile": "唐代诗人，豪放洒脱。",
        "conversations": [{"role": "user", "content": "今日为何饮酒？"}],
        "gold_focus": ["Knowledge"],
        "gold_focus_attr": "豪放",
        "reference_answer": "举杯。",
    }


def test_normalize_seed_keeps_gold_fields_out_of_messages_but_preserves_targets():
    row = _normalize_seed(_record(), arm="SFT", axis="heldout")
    joined_messages = "\n".join(message["content"] for message in row["messages"])

    assert row["gold_focus"] == ["Knowledge"]
    assert row["gold_focus_attr"] == "豪放"
    assert row["reference_answer"] == "举杯。"
    assert "Knowledge" in joined_messages  # legal label list is part of the schema instruction
    assert "举杯。" not in joined_messages


def test_render_generation_prompt_uses_chat_template_when_requested():
    row = _normalize_seed(_record(), arm="SFT", axis="heldout")

    prompt = _render_generation_prompt(row, FakeTokenizer(), {"prompt_format": "chat_template"})

    assert prompt == "<QWEN_CHAT_PROMPT>"


def test_render_generation_prompt_fits_chat_template_by_dropping_old_messages():
    row = {
        "messages": [
            {"role": "system", "content": "schema contract"},
            {"role": "user", "content": "old question " * 80},
            {"role": "assistant", "content": "old answer " * 80},
            {"role": "user", "content": "latest question"},
        ],
        "prompt": "legacy",
    }

    prompt = _render_generation_prompt(
        row,
        BudgetTokenizer(),
        {"prompt_format": "chat_template", "max_prompt_length": 20},
    )

    assert "schema contract" in prompt
    assert "latest question" in prompt
    assert "old question" not in prompt
    assert "old answer" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_render_generation_prompt_fails_closed_without_messages():
    with pytest.raises(ChatTemplateError):
        _render_generation_prompt(
            {"prompt": "legacy only"},
            FakeTokenizer(),
            {"prompt_format": "chat_template"},
        )


def test_normalize_seed_keeps_messages_for_structured_policy_prompt_records():
    record = dict(_record(), policy_prompt="legacy prompt should not block messages")

    row = _normalize_seed(record, arm="SFT", axis="heldout")

    assert row["prompt"] == "legacy prompt should not block messages"
    assert row["messages"]
    assert "今日为何饮酒？" in "\n".join(message["content"] for message in row["messages"])


def test_normalize_seed_does_not_invent_messages_for_plain_policy_prompt_records():
    row = _normalize_seed({"id": "plain", "policy_prompt": "A. B. C."}, arm="Base", axis="ceval")

    assert row["prompt"] == "A. B. C."
    assert "messages" not in row


def test_decode_completion_preserves_raw_text_without_strip():
    decoded = _decode_completion(FakeTokenizer(), [1, 2, 3])

    assert decoded == "  <think></think>\\boxed{答}ledged  "
