#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/featurize/work/anima}"
CODE="${CODE:-/home/featurize/work/Anima}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT/envs/anima-train-v2}"
CONDA_MAIN_CHANNEL="${CONDA_MAIN_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
CONDA_PYTORCH_CHANNEL="${CONDA_PYTORCH_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch}"
CONDA_NVIDIA_CHANNEL="${CONDA_NVIDIA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia}"

mkdir -p "$ROOT"/{envs,models,data,runs,eval_out,backups,logs,hf_cache}

cat > "$ROOT/env.sh" <<EOF
export HF_HOME="$ROOT/hf_cache"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="$ROOT/models/transformers_cache"
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export PYTHONPATH="$CODE/src:\${PYTHONPATH:-}"
EOF

source "$ROOT/env.sh"

if [ ! -x "$TRAIN_ENV/bin/python" ]; then
  conda create -y -p "$TRAIN_ENV" python=3.11 -c "$CONDA_MAIN_CHANNEL" --override-channels
fi

PY="$TRAIN_ENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

if ! "$PY" - <<'PY'
import torch
assert torch.__version__.startswith("2.5.1")
assert torch.cuda.is_available()
PY
then
  set +e
  conda install -y -p "$TRAIN_ENV" pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 \
    -c "$CONDA_PYTORCH_CHANNEL" -c "$CONDA_NVIDIA_CHANNEL" -c "$CONDA_MAIN_CHANNEL" --override-channels
  CONDA_TORCH_STATUS=$?
  set -e
  if [ "$CONDA_TORCH_STATUS" -ne 0 ]; then
    "$PY" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  fi
fi

# PyTorch 2.5.x can fail to import with the MKL/OpenMP 2025 stack
# (`undefined symbol: iJIT_NotifyEvent`). Pin the mature 2024 runtime.
conda install -y -p "$TRAIN_ENV" "mkl<2024.1" "intel-openmp<2024.1" \
  -c "$CONDA_MAIN_CHANNEL" --override-channels

"$PY" -m pip install \
  transformers==4.51.3 \
  trl==0.17.0 \
  peft==0.15.2 \
  datasets==3.5.0 \
  accelerate==1.6.0 \
  bitsandbytes==0.45.5 \
  sacrebleu==2.5.1 \
  pydantic==2.11.3 \
  pytest==9.1.1 \
  sentencepiece \
  protobuf \
  pyarrow \
  huggingface_hub

"$PY" - <<'PY'
import json
import torch, transformers, trl, peft, datasets
from trl import SFTTrainer, DPOTrainer, GRPOTrainer
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "peft": peft.__version__,
    "datasets": datasets.__version__,
    "trl_trainers": ["SFTTrainer", "DPOTrainer", "GRPOTrainer"],
}, ensure_ascii=False, sort_keys=True))
PY

mkdir -p "$ROOT/envs/locks"
conda env export -p "$TRAIN_ENV" > "$ROOT/envs/locks/anima-train-v2-conda-env.yaml" || true
"$PY" -m pip freeze > "$ROOT/envs/locks/anima-train-v2-pip-freeze.txt"
