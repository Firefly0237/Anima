# Anima — Verifiable-Reward GRPO for Chinese Role-Play NPCs

Reproduce a **verifiable-reward (RLVR) GRPO** recipe for Chinese role-play on a single RTX 4090:
**Qwen2.5-3B-Instruct + LoRA/QLoRA**, trained with a **self-implemented rule-based reward**, and
evaluated four arms (**Base / SFT / DPO / GRPO**) on a **leakage-controlled, multi-benchmark** suite.

## Highlights

- **RLVR core** — GRPO with a 100% rule-based, verifiable multi-reward:
  `R = 0.4·focus-overlap(graded) + 0.2·attribute-chrF/BLEU + 0.2·reference-chrF/BLEU + 0.2·format`.
  No neural reward model and **no API judge inside the RL loop** → bounded cost, nothing to hack.
- **Official-first stack** — Qwen/HF chat template + TRL `SFTTrainer` / `DPOTrainer` / `GRPOTrainer`,
  PEFT LoRA, 4-bit QLoRA. Project code stays in the reward, data, and eval layers.
- **Four-arm ablation** on the *same* objective eval, reporting **deltas, not absolutes**.
- **Leakage-controlled evaluation** — train characters never appear in any reported eval slice
  (split by source-work, with cross-split near-duplicate checks); the training reward is never an eval metric.
- **Honest results** — GRPO **matches SFT within noise** here; reported as such, with the mechanism explained
  rather than overclaimed.

## Results (four arms, public benchmarks)

Eval-first multi-axis comparison of the four arms (role-play rule-replay continuity, an external Chinese
role-play quality proxy, an objective social/role MCQ axis, and a general-capability regression canary):

| axis | Base | SFT | DPO | GRPO |
|---|---:|---:|---:|---:|
| role-play heldout (rule replay) | 0.00 | **0.82** | 0.82 | 0.82 |
| quality proxy (4-bit scalar) | 0.30 | **0.41** | 0.41 | 0.40 |
| social/role MCQ | 0.19 | 0.23 | **0.24** | **0.24** |
| general-capability canary | 0.46 | **0.53** | 0.51 | **0.53** |
| output-contract pass | 0 / 89 | 89 / 89 | 89 / 89 | 89 / 89 |

**Reading:** SFT delivers the bulk of the gain over Base and teaches the structured output contract;
DPO and GRPO match SFT within noise; the general-capability canary shows no regression. The near-tie is
the expected outcome for a strong, reward-aligned SFT warm-start under a short, KL-anchored RL run.

## Demo

A Gradio app (`src/anima/serve/gradio_app.py`) serves the model on **public-domain roles only**
(e.g. Sherlock Holmes, Sun Wukong, Socrates). It shows the in-character reply parsed from the trained
`<think><focus>…</focus>…</think> \boxed{reply}` format, optionally with the cognitive `focus` tags the
reward optimized. The same pipeline also extends to new custom personas (a local-only transfer study,
kept local out of respect for third-party IP).

## Repo structure

```
src/anima/
  data/      reward-record schema, validators, dedup/leakage, offline synthesis
  rewards/   focus-overlap, attribute/reference chrF·BLEU, format, parsing, weighted combine
  train/     SFT / DPO / GRPO loops (TRL + PEFT)
  eval/      rollout generation, rule scoring, CharacterEval/SocialBench/C-Eval harnesses, aggregation
  serve/     Gradio app (public roles)
  utils/     logging, backup, claim-boundary
configs/     SFT / DPO / GRPO / eval configs
scripts/     reproducible server run scripts
tests/       reward + parsing + validator unit tests (incl. Chinese fixtures)
```

## Reproduce (outline)

1. Build verifiable-reward records and the DPO preference pairs; commit the anti-leakage split.
2. `anima.train.sft` → `anima.train.grpo` (warm-started from SFT); DPO as a matched, reward-independent arm.
3. `anima.eval.*` to roll out and score all arms on the held-out + cross-benchmark axes; aggregate the table.

Two isolated envs: a training env (`transformers>=4.51`, recent torch, TRL, PEFT, vLLM) and an isolated
4-bit scorer env for the external quality proxy.

## Data & licenses

- **RoleBench** (Apache-2.0) for SFT / held-out role-play.
- **CharacterEval / BaichuanCharRM, SocialBench, C-Eval** used **eval-only** (research use; no raw-row
  redistribution — manifests + checksums only).
- Code is **Apache-2.0**. No third-party datasets or model weights are vendored in this repo.

## Honest scope

Single 4090, 3B + LoRA. The reward is rule-based (verifiable), not human-gold. This project does **not**
claim GRPO superiority — it ships a rigorous, leakage-controlled, reproducible comparison and reports the
deltas as measured.
