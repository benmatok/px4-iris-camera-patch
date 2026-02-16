# Drone Pursuit Simulation & Web Demo

This repository contains a Python-based simulation of a quadrotor drone performing autonomous visual pursuit. It features a custom 6-DOF physics engine, a visual servoing flight controller, and a web-based interface for real-time visualization and control.

## Overview

The project demonstrates a robust control strategy for tracking and intercepting a moving target using synthetic vision. The system is designed to handle "Blind Mode" scenarios where velocity sensors are unavailable, relying on internal state estimation and optical flow.

## Installation

### Prerequisites
- Python 3.8+

### Setup
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Demo

To start the simulation server and web interface:

```bash
python3 theshow.py
```

Once the server is running, open your browser and navigate to:
**http://localhost:8080**

The web interface allows you to:
- View the 3D simulation of the drone and target.
- Monitor real-time telemetry (Altitude, Pitch, Speed).
- Visualize "Ghost Paths" (predicted trajectories).
- Reset scenarios with different parameters.

## Methodology

### 1. Simulation Engine (`sim_interface.py`)
The simulation is built on a pure Python implementation of a 6-DOF Quadrotor model (`PyGhostModel`). It simulates:
- **Dynamics**: Thrust, drag, gravity, and rotational inertia.
- **Environment**: Dynamic wind vectors.
- **Sensors**: Synthetic camera feed generation for the visual tracker.

### 2. Flight Controller (`flight_controller.py`)
The `DPCFlightController` implements a heuristic-based control logic designed for robustness:
- **Visual Servoing**: Uses a PID-based approach with Adaptive Pitch Biasing to keep the target centered and maintain an optimal glide slope.
- **Blind Mode**: When velocity sensors (`vz`, `vx`, `vy`) are unavailable, the controller estimates state by integrating acceleration commands, allowing it to function without GPS or flow sensors.
- **Ghost Paths**: Generates forward-predicted trajectories based on the current estimated state. These "ghosts" are visualized in the web app to show where the drone "thinks" it is going.
- **Final Mode**: specialized logic triggered when close to the target to ensure precise docking or collision avoidance.

### 3. Vision System
- **Visual Tracker** (`visual_tracker.py`): Detects the target (red blob) in the synthetic camera image and provides `(u, v)` coordinates and size.
- **Flow Velocity Estimator** (`vision/flow_estimator.py`): Estimates the Focus of Expansion (FOE) from optical flow to aid in velocity estimation when other sensors fail.

## Validation

To verify the simulation logic and controller performance, you can run the scenario validation script:

```bash
python3 tests/validate_scenarios.py
```

This script runs multiple dive scenarios (varying altitude and distance) and checks if the drone successfully intercepts the target within a specified distance.
