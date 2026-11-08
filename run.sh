#!/usr/bin/env bash
# run.sh - Changeling Cosmetics. Sets up ./venv the first time (and again when requirements.txt changes),
# fetches the face parser once, then opens the window. ./run.sh photo.jpg opens with that photo.
set -e
cd "$(dirname "$0")"
if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=venv/bin/python; fi
if [ ! -x "$PY" ]; then
    echo "Creating venv ..."
    if command -v py >/dev/null 2>&1; then py -3.11 -m venv venv; else python3.11 -m venv venv; fi
    if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=venv/bin/python; fi
fi
if ! cmp -s requirements.txt venv/requirements.installed; then
    echo "Installing requirements ..."
    "$PY" -m pip install -r requirements.txt
    cp requirements.txt venv/requirements.installed
fi
if [ ! -f models/face_parsing_int8.onnx ]; then
    echo "Downloading the face parser, about 90 MB, once ..."
    mkdir -p models
    curl -L --fail -o models/face_parsing_int8.onnx.part "https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model_quantized.onnx"
    mv models/face_parsing_int8.onnx.part models/face_parsing_int8.onnx
fi
if [ ! -f models/dmt.onnx ]; then
    echo "Downloading the makeup transfer model, about 48 MB, once ..."
    curl -L --fail -o models/dmt.onnx.part "https://github.com/CursedPrograms/Changeling-Cosmetics/releases/download/models/dmt.onnx" && mv models/dmt.onnx.part models/dmt.onnx || true
fi
exec "$PY" app.py "$@"
