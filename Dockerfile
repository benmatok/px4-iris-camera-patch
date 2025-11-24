# syntax=docker/dockerfile:1

FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install core system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    gnupg \
    lsb-release \
    sudo \
    openssh-client \
    software-properties-common \
    && apt-get remove -y modemmanager \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
    

# Install development dependencies (including python3-pip, no python3-numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    openjdk-21-jdk \
    openjdk-21-jre \
    libopencv-dev \
    libhidapi-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    libhidapi-hidraw0 \
    libhidapi-libusb0 \
    python3-pip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip 
#RUN python3 -m pip install --no-cache-dir torch

# Install GStreamer dependencies (without system python3-gi)
RUN apt-get update && apt-get install -y \
    gir1.2-gstreamer-1.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install dev packages for building PyGObject
RUN apt-get update && apt-get install -y \
    libgirepository1.0-dev \
    gobject-introspection \
    libcairo2-dev \
    pkg-config \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Add ROS 2 Repository and Install ROS 2 Humble Packages
# ---------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
            http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ros-humble-ros-core \
        ros-dev-tools \
        ros-humble-xacro \
        ros-humble-urdf \
        ros-humble-cv-bridge \
        ros-humble-gazebo-ros \
	ros-humble-gazebo-plugins \
        ros-humble-robot-state-publisher && \
    rm -rf /var/lib/apt/lists/*


# install Python packages required for ROS
RUN pip3 install --no-cache-dir empy==3.3.4 pyros-genmsg setuptools colcon-common-extensions

# Install Python packages via pip for Python 3.10 (system default)
RUN python3 -m pip install --no-cache-dir \
    opencv-python \
    opencv-contrib-python \
    numpy==1.26.4 \
    mavsdk \
    "grpcio>=1.71.0" \
    pydualsense \
    pycairo \
    PyGObject \
    protobuf==5.29.0


RUN apt-get install sed

RUN apt-get update && apt-get install -y gazebo libgazebo11 libgazebo-dev && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y gedit && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    libjpeg-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    pygame \
    PyOpenGL

# Create a non-root user
RUN useradd -m px4user && \
    echo "px4user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/px4user && \
    usermod -a -G dialout px4user


USER px4user
WORKDIR /home/px4user

# Clone PX4-Autopilot repository
RUN git clone https://github.com/PX4/PX4-Autopilot.git -b v1.14.0 --recursive

# Run PX4 setup script and configure environment
WORKDIR /home/px4user/PX4-Autopilot/Tools/setup
RUN sed -i 's/matplotlib>=3.0.*/matplotlib>=3.0.0/' requirements.txt
RUN bash ubuntu.sh --no-sim-tools

# Download QGroundControl AppImage from GitHub releases
#RUN wget https://github.com/mavlink/qgroundcontrol/releases/download/v4.4.0/QGroundControl.AppImage -O QGroundControl.AppImage || \
#    { echo "QGroundControl download failed. Please manually download the AppImage and mount it into the container."; exit 1; } && \
#    chmod +x QGroundControl.AppImage


ARG CACHE_BREAKER=1

RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

WORKDIR /src/
RUN git clone https://github.com/benmatok/px4-iris-camera-patch.git
WORKDIR /src/px4-iris-camera-patch	
RUN chmod +x /src/px4-iris-camera-patch/apply_patch.sh
RUN sh /src/px4-iris-camera-patch/apply_patch.sh
WORKDIR /home/px4user/

# Set up environment
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# ===================================================================
# ADD FULL SALESFORCE WARPDRIVE + PYTORCH CUDA STACK
# (Works with FROM ubuntu:22.04 + NVIDIA runtime)
# ===================================================================

USER root

# Install CUDA toolkit components that match the driver on your host
# (These are the minimal runtime libraries + headers needed for PyTorch/WarpDrive)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-key del 7fa2af80 && \
    . /etc/os-release && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
        -o cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && \
    rm cuda-keyring.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-2 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch + WarpDrive + everything needed for training & live testing
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
        torch==2.3.0+cu121 \
        torchvision==0.18.0+cu121 \
        torchaudio==2.3.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir \
        pytorch-lightning>=2.0 \
        warp-drive>=2.1.0 \
        pymavlink \
        pyyaml \
        tqdm \
        tensorboard \
        jupyterlab \
        matplotlib \
        pandas

# Create workspace for your WarpDrive code and models
RUN mkdir -p /workspace/models && chown px4user:px4user /workspace /workspace/models

# Switch back to your normal user
USER px4user
WORKDIR /workspace

# Final environment
ENV PYTHONPATH="/workspace:${PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
