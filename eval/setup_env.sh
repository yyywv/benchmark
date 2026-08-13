#!/usr/bin/env bash
# 建立 RoboChrono 评测专用 conda 环境。
#
# 说明：
#  - torch 2.6.0 的 PyPI 默认 wheel 自带 CUDA 12.4，与本机驱动 550.127.08 匹配，
#    不需要额外的 download.pytorch.org 源。
#  - 走清华 PyPI 镜像：本机 NO_PROXY 已包含 .tuna.tsinghua.edu.cn，直连不过代理。
#  - 不改动任何已有 env。
set -euo pipefail

ENV_NAME="${ENV_NAME:-robochrono}"
CONDA_BASE="$(conda info --base)"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> 创建 conda 环境: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.11

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> 安装 PyTorch (CUDA 12.4)"
pip install -i "${PIP_INDEX}" torch==2.6.0 torchvision==0.21.0

echo "==> 安装推理与数据依赖"
pip install -i "${PIP_INDEX}" \
    "transformers==4.51.3" \
    "accelerate==1.6.0" \
    "einops==0.8.1" \
    "timm==1.0.15" \
    "sentencepiece==0.2.0" \
    "decord==0.6.0" \
    "opencv-python-headless==4.11.0.86" \
    "pillow==11.1.0" \
    "numpy==1.26.4" \
    "pandas==2.2.3" \
    "requests==2.32.3" \
    "huggingface_hub==0.30.2" \
    "pyyaml==6.0.2" \
    "tqdm==4.67.1"

echo
echo "==> 自检"
python - <<'PY'
import torch, transformers, cv2, decord, PIL, numpy, pandas, requests, yaml
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
print(f"  transformers {transformers.__version__}")
print(f"  opencv       {cv2.__version__}")
print(f"  decord       ok")
print(f"  numpy        {numpy.__version__}")
PY

echo
echo "完成。使用: conda activate ${ENV_NAME}"
