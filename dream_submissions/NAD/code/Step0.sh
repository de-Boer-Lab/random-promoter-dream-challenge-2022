#!/bin/bash
# Install pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# Install requirements
pip install -r requirements.txt
