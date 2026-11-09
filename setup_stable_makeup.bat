@echo off
setlocal
cd /d "%~dp0"
rem Stable-Makeup for Cosmeta's Model switch, once: its code (the Space sky24h/Stable-Makeup-unofficial, without its
rem weights), its own environment (StableMakeup\smenv: about 3 GB of Python packages) and its models (about 8 GB, into
rem F:\AIModels\StableMakeup when F: is there; STABLE_MAKEUP_DIR overrides).
if not exist "StableMakeup\pipeline_sd15.py" (
    echo Getting Stable-Makeup's code...
    set GIT_LFS_SKIP_SMUDGE=1
    git clone --depth 1 https://huggingface.co/spaces/sky24h/Stable-Makeup-unofficial StableMakeup || goto :error
    rmdir /s /q "StableMakeup\checkpoints"
)
set "PY=StableMakeup\smenv\Scripts\python.exe"
if not exist "%PY%" (
    py -3.11 -m venv StableMakeup\smenv || goto :error
    "%PY%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 || goto :error
    "%PY%" -m pip install transformers==4.42.4 accelerate==0.33.0 huggingface-hub==0.25.2 spiga batch_face opencv-python "numpy<2" pillow safetensors gdown || goto :error
)
python -m cosmetics.stable_makeup_models || goto :error
if not defined STABLE_MAKEUP_DIR set "STABLE_MAKEUP_DIR=F:\AIModels\StableMakeup"
if not exist "StableMakeup\checkpoints" mklink /J "StableMakeup\checkpoints" "%STABLE_MAKEUP_DIR%" >nul
echo Stable-Makeup is ready: pick it under Copy makeup, Model.
pause
exit /b 0
:error
echo Setup failed, see the messages above.
pause
exit /b 1
