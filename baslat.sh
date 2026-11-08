#!/usr/bin/env bash
# Hype Vision - Ubuntu baslatici
#   chmod +x baslat.sh && ./baslat.sh
# Ilk calistirmada: sistem paketleri, sanal ortam, Python paketleri ve modeller kurulur.
set -e
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if ! $PY -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)' 2>/dev/null; then
  for v in python3.12 python3.11 python3.10; do command -v $v >/dev/null && PY=$v && break; done
fi

if [ ! -d venv ]; then
  echo ">> Sistem paketleri (ffmpeg, venv) kontrol ediliyor..."
  if command -v apt-get >/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-venv python3-pip ffmpeg curl >/dev/null
  fi
  echo ">> Python ortami kuruluyor ($PY)..."
  $PY -m venv venv
  ./venv/bin/pip install -q --upgrade pip
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    echo ">> NVIDIA GPU bulundu, GPU paketleri kuruluyor (~2 GB)..."
    ./venv/bin/pip install -r requirements.txt
  else
    echo ">> GPU yok, CPU paketleri kuruluyor..."
    ./venv/bin/pip install -r requirements-cpu.txt
  fi
fi

mkdir -p models
if [ ! -f models/auraface/glintr100.onnx ]; then
  echo ">> AuraFace modeli indiriliyor (~260 MB)..."
  ./venv/bin/python -c "from huggingface_hub import hf_hub_download as d; [d('fal/AuraFace-v1',f,local_dir='models/auraface') for f in ['glintr100.onnx','LICENSE.md']]"
fi
if [ ! -f models/yunet.onnx ]; then
  echo ">> YuNet modeli indiriliyor..."
  curl -sL -o models/yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
fi

echo ">> Sunucu: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT:-8000}"
exec ./venv/bin/python app.py
