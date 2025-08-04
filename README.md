# PX4 Iris Camera Patch for Gazebo Classic

This repo provides a patch to add a forward-looking RGB camera pitched up at 30 degrees relative to the Iris body in PX4 Gazebo Classic simulations (compatible with v1.14.0).

## How to Use in Docker Build
- Clone this repo after cloning PX4-Autopilot in your Dockerfile.
- Run `./apply_patch.sh` to overwrite the standard iris.sdf with the modified version.

Example Dockerfile snippet (add after PX4 clone):
