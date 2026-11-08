@echo off
rem run.bat - Changeling Cosmetics. Sets up .\venv the first time (and again when requirements.txt changes),
rem fetches the face parser once, then opens the window. run.bat photo.jpg opens with that photo.
setlocal
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Creating venv ...
    py -3.11 -m venv venv || goto :fail
)
fc /b requirements.txt venv\requirements.installed >nul 2>&1
if errorlevel 1 (
    echo Installing requirements ...
    "%PY%" -m pip install -r requirements.txt || goto :fail
    copy /y requirements.txt venv\requirements.installed >nul
)
if not exist "models\face_parsing_int8.onnx" (
    echo Downloading the face parser, about 90 MB, once ...
    if not exist models mkdir models
    curl -L --fail -o "models\face_parsing_int8.onnx.part" "https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model_quantized.onnx" || goto :fail
    move /y "models\face_parsing_int8.onnx.part" "models\face_parsing_int8.onnx" >nul
)
if not exist "models\dmt.onnx" (
    echo Downloading the makeup transfer model, about 48 MB, once ...
    curl -L --fail -o "models\dmt.onnx.part" "https://github.com/CursedPrograms/Changeling-Cosmetics/releases/download/models/dmt.onnx" && move /y "models\dmt.onnx.part" "models\dmt.onnx" >nul
)
"%PY%" app.py %*
if errorlevel 1 pause
exit /b
:fail
echo Setup failed: Changeling Cosmetics needs Python 3.11.
pause
exit /b 1
