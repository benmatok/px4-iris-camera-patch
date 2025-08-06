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
RUN python3 -m pip install --no-cache-dir torch

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
    numpy==1.26.4 \
    mavsdk \
    "grpcio>=1.71.0" \
    pydualsense \
    pycairo \
    PyGObject \
    protobuf==5.29.0


RUN apt-get install sed

RUN apt-get update && apt-get install -y gazebo libgazebo11 libgazebo-dev && rm -rf /var/lib/apt/lists/*

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
RUN wget https://github.com/mavlink/qgroundcontrol/releases/download/v4.4.0/QGroundControl.AppImage -O QGroundControl.AppImage || \
    { echo "QGroundControl download failed. Please manually download the AppImage and mount it into the container."; exit 1; } && \
    chmod +x QGroundControl.AppImage
    
ARG CACHE_BREAKER=1

WORKDIR /src/
RUN git clone https://github.com/benmatok/px4-iris-camera-patch.git
WORKDIR /src/px4-iris-camera-patch	
RUN chmod +x /src/px4-iris-camera-patch/apply_patch.sh
RUN sh /src/px4-iris-camera-patch/apply_patch.sh
WORKDIR /home/px4user/

# Set up environment
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
