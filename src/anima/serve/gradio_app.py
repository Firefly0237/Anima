"""HF ZeroGPU Space app — Chinese role-play NPC (PUBLIC-DOMAIN roles only). [W5]

Serves Qwen2.5-3B-Instruct + a formal-v1 LoRA adapter (SFT or GRPO) as a
structured role-play NPC: the model emits the trained
``<think><focus>...</focus><focus_attr>...</focus_attr></think> \\boxed{reply}``
format; the UI shows the in-character reply (the boxed answer) and, optionally,
the cognitive ``<focus>`` tags the reward optimized.

RED LINES:
- PUBLIC-DOMAIN roles ONLY. No HSR / 三月七 / HoYoverse characters here, ever.
- Loads a LoRA adapter on top of the base model; no merged weights required.

Runs on HF ZeroGPU (uses ``spaces.GPU`` when available) and locally (falls back
to a no-op decorator). Config via env:
- ``BASE_MODEL``     (default ``Qwen/Qwen2.5-3B-Instruct``)
- ``ADAPTER_PATH``   local dir or HF repo id of the formal-v1 adapter (optional;
                     if unset, serves the base model)
- ``ADAPTER_LABEL``  display label, e.g. ``SFT`` / ``GRPO`` (default ``SFT``)
"""

from __future__ import annotations

import os
import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # HF ZeroGPU; harmless no-op locally
    import spaces

    GPU = spaces.GPU
except Exception:  # pragma: no cover - local fallback

    def GPU(func=None, **_kwargs):
        return func if func else (lambda f: f)


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "").strip()
ADAPTER_LABEL = os.environ.get("ADAPTER_LABEL", "SFT")

# Public-domain / classic roles only (no copyrighted or HoYoverse IP).
ROLE_CARDS: dict[str, str] = {
    "夏洛克·福尔摩斯": (
        "夏洛克·福尔摩斯——亚瑟·柯南·道尔笔下的咨询侦探（公共领域）。"
        "极度理性、观察入微、以演绎推理见长；言谈犀利、略带傲慢，对平庸缺乏耐心，"
        "酷爱有挑战的谜题。会从细节推断对方身份与处境。"
    ),
    "孙悟空": (
        "孙悟空——《西游记》中的齐天大圣（公共领域）。"
        "桀骜不驯、机敏好斗、重情义；自称'俺老孙'，说话豪爽带火气，"
        "有七十二变与火眼金睛，最恨妖魔与虚伪，护短而忠于取经队伍。"
    ),
    "苏格拉底": (
        "苏格拉底——古希腊哲学家（公共领域）。"
        "以'诘问法'闻名，从不直接给答案，而是层层反问引导对方自省；"
        "谦逊地自称'我只知道我一无所知'，温和而执着地追问定义与本质。"
    ),
    "林黛玉": (
        "林黛玉——《红楼梦》中的人物（公共领域）。"
        "才情出众、敏感细腻、多愁善感；说话含蓄婉转、好用诗词，"
        "心思缜密、自尊心强，对真心相待者极为珍重。"
    ),
}

FORMAT_INSTRUCTION = (
    "请用中文扮演指定角色，保持角色身份、语气、价值观和对话上下文。"
    "输出必须是一个 assistant 回复，结构为："
    "<think>简短说明本次角色回复关注点。"
    "<focus>从 Knowledge, Style, Worldview, Emotion, Empathetic, Engagement, "
    "Human_Like, Extension, Memory, Safety 中选择一个或多个英文标签，用逗号分隔</focus>"
    "<focus_attr>用中文写出本次回复体现的角色属性</focus_attr></think> "
    "\\boxed{你的角色回复}"
)

_BOXED = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
_FOCUS = re.compile(r"<focus>\s*(.*?)\s*</focus>", re.DOTALL | re.IGNORECASE)

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # Load on CPU; the @GPU function moves it to CUDA per call. ZeroGPU reclaims
    # the GPU between requests, so a global model placed on CUDA at import goes
    # stale — load on CPU and .to(device) inside the GPU-allocated function.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if ADAPTER_PATH:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    _model = model


def _build_messages(role: str, history: list[dict], user_msg: str) -> list[dict]:
    system = f"{ROLE_CARDS[role]}\n\n{FORMAT_INSTRUCTION}"
    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})
    return messages


@GPU(duration=60)
def _generate(role: str, history: list[dict], user_msg: str, show_focus: bool) -> str:
    _load()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(dev)
    messages = _build_messages(role, history, user_msg)
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(dev)
    with torch.inference_mode():
        out = _model.generate(
            **inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9
        )
    text = _tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    boxed = _BOXED.search(text)
    reply = boxed.group(1).strip() if boxed else text.strip()
    if show_focus:
        focus = _FOCUS.search(text)
        if focus:
            reply = f"{reply}\n\n_（关注点 focus: {focus.group(1).strip()}）_"
    return reply


def chat_fn(user_msg, history, role, show_focus):
    reply = _generate(role, history or [], user_msg, show_focus)
    return reply


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Anima · 中文角色扮演 NPC (RLVR)") as demo:
        gr.Markdown(
            "# Anima — 中文角色扮演 NPC（可验证奖励 GRPO）\n"
            f"Qwen2.5-3B-Instruct + LoRA（**{ADAPTER_LABEL}** arm）。"
            "模型以训练得到的 `<think><focus>…</focus>…</think> \\boxed{回复}` 结构输出；"
            "下方展示括号内的角色回复，可勾选显示其优化的认知 `focus` 标签。\n\n"
            "*仅公共领域角色；本演示不含任何受版权保护或 HoYoverse 的角色。*"
        )
        with gr.Row():
            role = gr.Dropdown(choices=list(ROLE_CARDS), value=list(ROLE_CARDS)[0], label="选择角色")
            show_focus = gr.Checkbox(value=True, label="显示 focus 标签")
        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[role, show_focus],
            type="messages",
            examples=[["你是谁？", "夏洛克·福尔摩斯", True], ["你怕死吗？", "苏格拉底", True]],
        )
    return demo


if __name__ == "__main__":
    build_demo().queue().launch()
