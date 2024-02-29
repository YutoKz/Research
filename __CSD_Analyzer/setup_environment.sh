#!/bin/bash

# ドライバ の認識確認
echo "----- Check Driver -----"
nvidia-smi
echo ""

# CUDA の確認
echo "----- Check CUDA -----"
nvcc -V
echo ""

# apt パッケージのアップデート
apt update

# python, pip
apt install -y python3-pip 

# Opencv に必要なパッケージ
apt install -y libgl1-mesa-glx libglib2.0-0 

# Python パッケージ
pip3 install -r requirements.txt