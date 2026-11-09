#!/usr/bin/env bash
# Stable-Makeup for Cosmeta's Model switch, once: its code, its own environment (StableMakeup/smenv, about 3 GB of
# Python packages) and its models (about 8 GB; STABLE_MAKEUP_DIR says where, default F:\AIModels\StableMakeup).
set -e
cd "$(dirname "$0")"
if [ ! -f StableMakeup/pipeline_sd15.py ]; then
    echo "Getting Stable-Makeup's code..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/spaces/sky24h/Stable-Makeup-unofficial StableMakeup
    rm -rf StableMakeup/checkpoints
fi
PY=StableMakeup/smenv/bin/python
[ -d StableMakeup/smenv/Scripts ] && PY=StableMakeup/smenv/Scripts/python.exe
if [ ! -x "$PY" ]; then
    python3.11 -m venv StableMakeup/smenv
    [ -d StableMakeup/smenv/Scripts ] && PY=StableMakeup/smenv/Scripts/python.exe
    "$PY" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    "$PY" -m pip install transformers==4.42.4 accelerate==0.33.0 huggingface-hub==0.25.2 spiga batch_face opencv-python "numpy<2" pillow safetensors gdown
fi
python3 -m cosmetics.stable_makeup_models
ROOT="${STABLE_MAKEUP_DIR:-F:/AIModels/StableMakeup}"
[ -e StableMakeup/checkpoints ] || ln -s "$ROOT" StableMakeup/checkpoints
echo ready > StableMakeup/ready
echo "Stable-Makeup is ready: pick it under Copy makeup, Model."
