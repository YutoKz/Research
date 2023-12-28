#!/bin/bash

# ドライバ の認識確認
nvidia-smi

# CUDA の確認
nvcc -V

# apt パッケージのアップデート
apt update

# python, pip
apt install -y python3-pip 

# Python パッケージ
pip3 install -r requirements.txt

# Opencv に必要なパッケージ
apt install -y libgl1-mesa-glx libglib2.0-0 