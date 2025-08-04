# PX4 Iris Camera Patch for Gazebo Classic

This repo provides a patch to add a forward-looking RGB camera pitched up at 30 degrees relative to the Iris body in PX4 Gazebo Classic simulations (compatible with v1.14.0).

## How to Use in Docker Build
- Clone this repo and build using the dockerfile
```
sudo docker build --build-arg CACHE_BREAKER=$RANDOM -t px4-gazebo-setup .
```
