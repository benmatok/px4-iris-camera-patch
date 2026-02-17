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
The `DPCFlightController` implements a robust, multi-stage control strategy for visual homing:

1. **Basic Tracking (Visual Servoing)**:
   Uses PID control on the target's image coordinates `(u, v)` to steer the drone. It incorporates an **Adaptive Pitch Bias** ("Tent Function") that adjusts the glide slope target based on current pitch, preventing overshoot in steep dives while maintaining altitude in shallow approaches.

2. **RER (Rapid Exponential Rendezvous)**:
   Regulates closure speed using the **Relative Expansion Rate** (time-to-contact estimate) of the target's bounding box.
   - **Thrust Modulation**: Reduces thrust (brakes) if the target expands too quickly, ensuring a controlled collision.
   - **Flare Logic**: Triggers a pitch-up maneuver if the expansion rate exceeds safety thresholds to prevent high-speed impact.

3. **Final Mode (Docking)**:
   Activated when the target is in close proximity (large vertical error in frame).
   - **Recovery**: If undershooting (target high), the drone levels off and boosts thrust.
   - **Sink**: If overshooting (target low), the drone pitches down and reduces thrust to sink onto the target.

Additionally, the controller features a **Blind Mode Estimator** that integrates acceleration commands to estimate velocity when sensors are unavailable, and a **Ghost Path** generator for visualizing predicted trajectories.

### 3. Vision System
- **Visual Tracker** (`visual_tracker.py`): Detects the target (red blob) in the synthetic camera image and provides `(u, v)` coordinates and size.
- **Flow Velocity Estimator** (`vision/flow_estimator.py`): Estimates the Focus of Expansion (FOE) from optical flow to aid in velocity estimation when other sensors fail.

## Validation

To verify the simulation logic and controller performance, you can run the scenario validation script:

```bash
python3 tests/validate_scenarios.py
```

This script runs multiple dive scenarios (varying altitude and distance) and checks if the drone successfully intercepts the target within a specified distance.
