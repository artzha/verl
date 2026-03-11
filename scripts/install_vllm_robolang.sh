#!/bin/bash

uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --torch-backend=cu128

uv pip install -U "torch==2.9.*" vllm=="0.16.0" \
    --torch-backend=cu128 \
    --extra-index-url https://pypi.org/simple/vllm

uv pip install flash_attn==2.8.3 --no-build-isolation --no-cache-dir