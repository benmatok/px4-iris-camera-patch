FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gnupg lsb-release sudo openssh-client software-properties-common \
    build-essential cmake \
    python3-pip python3-dev \
    libopencv-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python Upgrade (Standardize on system python or upgrade if needed)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch (Match previous version or latest stable compatible with CUDA 12.1/12.2)
RUN pip3 install --no-cache-dir \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Project Dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Additional dependencies if not in requirements.txt (just in case)
RUN pip3 install --no-cache-dir \
    pytorch-lightning>=2.0 jupyterlab

# Workspace Setup
WORKDIR /workspace
ENV PYTHONPATH="/workspace:${PYTHONPATH:-}"

CMD ["/bin/bash"]
