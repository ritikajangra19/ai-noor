#!/bin/bash

set -euo pipefail

echo "[watch] Step 1/10: Activating conda environment 'MuseTalk'..."
conda activate MuseTalk

echo "[watch] Step 2/10: Installing PyTorch stack with pip..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo "[watch] Step 3/10: Installing PyTorch stack with conda..."
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

echo "[watch] Step 4/10: Installing Python requirements..."
pip install -r requirements.txt

echo "[watch] Step 5/10: Installing openmim..."
pip install --no-cache-dir -U openmim

echo "[watch] Step 6/10: Installing mmengine..."
mim install mmengine

echo "[watch] Step 7/10: Installing mmcv==2.0.1..."
mim install "mmcv==2.0.1"

echo "[watch] Step 8/10: Installing mmdet==3.1.0..."
mim install "mmdet==3.1.0"

echo "[watch] Step 9/10: Installing mmpose==1.1.0..."
mim install "mmpose==1.1.0"

echo "[watch] Step 10/10: Updating package lists and installing ffmpeg..."
apt update
sudo apt-get install ffmpeg
ffmpeg -version

echo "[watch] Installation script completed successfully."

