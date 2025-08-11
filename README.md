<img width="1475" height="850" alt="Screenshot from 2025-08-10 12-48-40" src="https://github.com/user-attachments/assets/47b1f314-c5a3-4acc-ab10-66812e4b3dc6" />

This repo provides a patch to add a forward-looking RGB camera pitched up at 30 degrees relative to the Iris body in PX4 Gazebo Classic simulations (compatible with v1.14.0).
### Prequesits 
- nvidia drivers
- nvidia docker
- internet connection


## How to Use in Docker Build
- Clone this repo
- build using the dockerfile
- run using the command below
```
git clone https://github.com/benmatok/px4-iris-camera-patch.git
cd px4-iris-camera-patch
sudo docker build --build-arg CACHE_BREAKER=$RANDOM -t px4-gazebo-setup .
xhost +local:
sudo docker run -it --rm --privileged --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/dri:/dev/dri -v /dev/shm:/dev/shm --network host -v $HOME/.ssh:/home/px4user/.ssh:rw -v /dev:/dev px4-gazebo-setup
inside-container> cd ~/PX4-Autopilot
inside-container> HEADLESS=1 make px4_sitl gazebo-classic_iris__baylands
inside-container> HEADLESS=1 make px4_sitl gazebo-classic_iris__ksql_airport
```
## in a second bash
```
sudo docker ps  # Note the CONTAINER ID of your running px4 container
sudo docker exec -it <CONTAINER_ID> bash
inside-container> python3 /src/px4-iris-camera-patch/main.py
```
