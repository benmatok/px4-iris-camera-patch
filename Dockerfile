FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Basic Build Tools and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gnupg lsb-release sudo openssh-client software-properties-common \
    build-essential cmake \
    libopencv-dev \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
        opencv-python opencv-contrib-python numpy==1.26.4 \
        "grpcio>=1.71.0" protobuf==5.29.0 \
        aiohttp

# PyTorch
RUN pip3 install --no-cache-dir \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# RL / WarpDrive
RUN pip3 install --no-cache-dir rl-warp-drive==2.7.1

# Remaining RL deps
RUN pip3 install --no-cache-dir \
    pytorch-lightning>=2.0 pyyaml tqdm tensorboard \
    jupyterlab matplotlib pandas numba

# Workspace
WORKDIR /workspace
ENV PYTHONPATH="/workspace:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES=0

CMD ["/bin/bash"]
