# Stable-Makeup for Cosmeta's Model switch, once: its code, its own environment (StableMakeup\smenv, about 3 GB of
# Python packages) and its models (about 8 GB, into F:\AIModels\StableMakeup when F: is there; STABLE_MAKEUP_DIR overrides).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path "StableMakeup\pipeline_sd15.py")) {
    Write-Host "Getting Stable-Makeup's code..."
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    git clone --depth 1 https://huggingface.co/spaces/sky24h/Stable-Makeup-unofficial StableMakeup
    Remove-Item -Recurse -Force "StableMakeup\checkpoints" -ErrorAction SilentlyContinue
}
$Py = "StableMakeup\smenv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    py -3.11 -m venv StableMakeup\smenv
    & $Py -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    & $Py -m pip install transformers==4.42.4 accelerate==0.33.0 huggingface-hub==0.25.2 spiga batch_face opencv-python "numpy<2" pillow safetensors gdown
}
python -m cosmetics.stable_makeup_models
$Root = if ($env:STABLE_MAKEUP_DIR) { $env:STABLE_MAKEUP_DIR } else { "F:\AIModels\StableMakeup" }
if (-not (Test-Path "StableMakeup\checkpoints")) { New-Item -ItemType Junction -Path "StableMakeup\checkpoints" -Target $Root | Out-Null }
Set-Content "StableMakeupeady" "ready"
Write-Host "Stable-Makeup is ready: pick it under Copy makeup, Model."
