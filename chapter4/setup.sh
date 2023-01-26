#!/bin/bash

python -m venv .venv
source .venv/bin/activate && pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
source .venv/bin/activate && pip install -r requirements.txt
