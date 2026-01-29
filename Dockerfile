FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
# Full CUDA + nvcc + cuda.h = pycuda builds perfectly

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# ONE apt-get update for everything system-level
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gnupg lsb-release sudo openssh-client software-properties-common \
    build-essential cmake openjdk-21-jdk openjdk-21-jre \
    libopencv-dev libhidapi-dev libusb-1.0-0-dev libudev-dev \
    python3-pip python3-dev \
    # GStreamer + PyGObject + SDL + Gazebo
    gir1.2-gstreamer-1.0 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgirepository1.0-dev gobject-introspection libcairo2-dev pkg-config \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libfreetype6-dev libportmidi-dev libjpeg-dev gedit sed \
    gazebo libgazebo11 libgazebo-dev \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
          http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-core ros-dev-tools \
        ros-humble-xacro ros-humble-urdf ros-humble-cv-bridge \
        ros-humble-gazebo-ros ros-humble-gazebo-plugins \
        ros-humble-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*

# Python deps (your list)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
        empy==3.3.4 pyros-genmsg colcon-common-extensions \
        opencv-python opencv-contrib-python numpy==1.26.4 \
        mavsdk "grpcio>=1.71.0" pydualsense pycairo PyGObject protobuf==5.29.0 \
        pygame PyOpenGL

# Non-root user + PX4 + iris-camera-patch (your exact workflow, shallow clone)
RUN useradd -m px4user && \
    echo "px4user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/px4user && \
    usermod -a -G dialout px4user
USER px4user
WORKDIR /home/px4user

RUN git clone --depth 1 --branch v1.14.0 --single-branch --recursive \
    https://github.com/PX4/PX4-Autopilot.git /home/px4user/PX4-Autopilot

WORKDIR /home/px4user/PX4-Autopilot/Tools/setup
RUN sed -i 's/matplotlib>=3.0.*/matplotlib>=3.0.0/' requirements.txt && \
    bash ubuntu.sh --no-sim-tools

RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

WORKDIR /home/px4user
RUN git clone https://github.com/benmatok/px4-iris-camera-patch.git && \
    cd px4-iris-camera-patch && chmod +x apply_patch.sh && ./apply_patch.sh

WORKDIR /home/px4user
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PATH=$PATH:$JAVA_HOME/bin

# ===================================================================
# FULL WARPDRIVE V2.7.1 + PYTORCH + FULL PYCUDA (compiled!)
# ===================================================================
USER root

# PyTorch pre-compiled wheel
RUN pip3 install --no-cache-dir \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir rl-warp-drive==2.7.1

# Remaining RL deps
RUN pip3 install --no-cache-dir \
    pytorch-lightning>=2.0 pymavlink pyyaml tqdm tensorboard \
    jupyterlab matplotlib pandas numba

# Workspace
RUN mkdir -p /workspace/models && chown px4user:px4user /workspace /workspace/models
USER px4user
WORKDIR /workspace

ENV PYTHONPATH="/workspace:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES=0

CMD ["/bin/bash"]
