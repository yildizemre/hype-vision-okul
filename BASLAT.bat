@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv (
  echo Python ortami kuruluyor...
  py -3.12 -m venv venv
  venv\Scripts\python -m pip install -r requirements.txt
)
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
