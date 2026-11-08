@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Hype Vision - Windows baslatici. Python 3.10-3.12 gerekli (py launcher ile).

if not exist venv (
  echo Python ortami kuruluyor...
  py -3.12 -m venv venv 2>nul || py -3.11 -m venv venv 2>nul || py -3.10 -m venv venv
  venv\Scripts\python -m pip install -q --upgrade pip
  nvidia-smi >nul 2>&1
  if errorlevel 1 (
    echo GPU bulunamadi, CPU paketleri kuruluyor...
    venv\Scripts\python -m pip install -r requirements-cpu.txt
  ) else (
    echo NVIDIA GPU bulundu, GPU paketleri kuruluyor ^(~2 GB^)...
    venv\Scripts\python -m pip install -r requirements.txt
  )
)
if not exist models mkdir models
if not exist models\auraface\glintr100.onnx (
  echo AuraFace modeli indiriliyor ^(~260 MB^)...
  venv\Scripts\python -c "from huggingface_hub import hf_hub_download as d; [d('fal/AuraFace-v1',f,local_dir='models/auraface') for f in ['glintr100.onnx','LICENSE.md']]"
)
if not exist models\yunet.onnx (
  echo YuNet modeli indiriliyor...
  curl -sL -o models\yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
)
start "" http://localhost:8000
venv\Scripts\python app.py
pause
