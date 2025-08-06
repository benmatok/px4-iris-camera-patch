

This repo provides a patch to add a forward-looking RGB camera pitched up at 30 degrees relative to the Iris body in PX4 Gazebo Classic simulations (compatible with v1.14.0).

## How to Use in Docker Build
- Clone this repo
- build using the dockerfile
- run using the command below
```
git clone https://github.com/benmatok/px4-iris-camera-patch.git
cd px4-iris-camera-patch
sudo docker build --build-arg CACHE_BREAKER=$RANDOM -t px4-gazebo-setup .
xhost +local:
sudo docker run -it --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/dri:/dev/dri -v /dev/shm:/dev/shm --network host -v $HOME/.ssh:/home/px4user/.ssh:rw -v /dev:/dev px4-gazebo-setup
```

## inside the docker
```
source /opt/ros/humble/setup.bash
cd ~/PX4-Autopilot
make px4_sitl gazebo-classic_iris__baylands
```
## in a second bash
```
sudo docker ps  # Note the CONTAINER ID of your running px4 container
sudo docker exec -it <CONTAINER_ID> bash
```

## inside second docker
```
source /opt/ros/humble/setup.bash
cd /src/px4-iris-camera-patch
python3 main.py
```
