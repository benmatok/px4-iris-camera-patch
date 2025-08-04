#!/bin/bash
set -e

# Path to PX4's Iris SDF file
PX4_SDF_PATH="/home/px4user/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf"

# Overwrite with modified SDF
cp iris.sdf "$PX4_SDF_PATH"

echo "Iris SDF patched with forward camera."
