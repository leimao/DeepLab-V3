#!/bin/bash
# Python dependencis
pip install tensorflow-gpu opencv-python tqdm pillow joblib requests

# May be required by OpenCV
apt update && apt install -y libsm6 libxext6 libxrender-dev

