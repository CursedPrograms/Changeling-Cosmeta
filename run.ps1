# run.ps1 - Changeling Cosmeta. Sets up .\venv the first time (and again when requirements.txt changes), fetches the
# models once, then starts app.py. Options pass through: .\run.ps1 --camera 1 ...
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Py = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
function Fail($msg) { Write-Host "Setup failed: $msg"; Read-Host 'Press Enter'; exit 1 }
if (-not (Test-Path $Py)) {
    Write-Host 'Creating venv ...'
    & py -3.11 -m venv venv
    if ($LASTEXITCODE) { Fail 'Changeling Cosmeta needs Python 3.11.' }
}
$stamp = 'venv\requirements.installed'
if (-not (Test-Path $stamp) -or (Get-FileHash requirements.txt).Hash -ne (Get-FileHash $stamp).Hash) {
    Write-Host 'Installing requirements ...'
    & $Py -m pip install -r requirements.txt
    if ($LASTEXITCODE) { Fail 'pip could not install the requirements.' }
    Copy-Item requirements.txt $stamp -Force
}
function Fetch($file, $url, $what) {
    if (-not (Test-Path "models/$file")) {
        Write-Host "Downloading $what, once ..."
        New-Item -ItemType Directory -Force models | Out-Null
        & curl.exe -L --fail -o "models/$file.part" $url
        if ($LASTEXITCODE) { Fail "could not download $file." }
        Move-Item "models/$file.part" "models/$file" -Force
    }
}
Fetch 'face_parsing_int8.onnx' 'https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model_quantized.onnx' 'the face parser, about 90 MB'
& $Py app.py @args
if ($LASTEXITCODE) { Read-Host 'Press Enter' }
