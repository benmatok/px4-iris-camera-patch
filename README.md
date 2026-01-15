# Jules: High-Performance Drone Pursuit Policy

This repository contains the training and validation infrastructure for "Jules", a neural network policy trained to pilot a drone and intercept a moving target using visual tracking inputs.

## Project Status
The project is currently focused on **Supervised Learning** where a Student Policy learns to mimic an Oracle (`LinearPlanner`) that has access to ground truth state information.

**Artifacts:**
- `final_jules.pth`: The fully trained model after 5000 iterations.
- `latest_jules.pth`: Checkpoint saved every 50 iterations during training.

## Installation

### Prerequisites
- **Python 3.8+**
- **C++ Compiler**: `g++` or `clang++` with OpenMP support.
- **Python Packages**: Listed in `requirements.txt`.

### Quick Start
To set up the environment, compile the optimized Cython physics engine, and start training immediately:

```bash
./run_training.sh
```

### Manual Installation
If you prefer to run steps manually:

```bash
# 1. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Compile Cython extensions (Physics Engine)
python3 setup.py build_ext --inplace
```

## Training

The core training script is `train_drone.py`. It runs a simulation loop where a `LinearPlanner` (Oracle) computes optimal actions to intercept the target, and a `DronePolicy` (Student) is trained via Supervised Learning (MSE Loss) to clone this behavior.

### Usage
```bash
python3 train_drone.py --num_agents 200 --iterations 5000
```

### Arguments
- `--num_agents`: Number of parallel drones to simulate (default: 200).
- `--iterations`: Number of training iterations (default: 5000).
- `--episode_length`: Duration of each simulation episode in steps (default: 400).
- `--load`: Path to a checkpoint (e.g., `latest_jules.pth`) to resume training.
- `--viz_freq`: Frequency (in iterations) to generate visualization GIFs (default: 100).
- `--debug`: Enable debug mode (NaN checks).

### The Oracle (LinearPlanner)
The teacher is a rule-based planner that uses Inverse Dynamics:
1.  **Elevation Control**: If the drone is too low (relative to target), it commands a climb to achieve a 15-degree glide slope.
2.  **Intercept**: Uses Proportional Navigation logic to compute the necessary acceleration for a constant-velocity intercept.

### Outputs
- **Logs**: Training loss and validation distance are printed to stdout.
- **Visualizations**:
    - `visualizations/training_loss.png`: Plot of MSE loss over time.
    - `visualizations/reward_plot.png`: Plot of validation performance (negative distance).
    - `visualizations/traj_{itr}.gif`: Animated top-down and POV view of the drone.

## Validation

To test the robustness of the trained policy against sensor noise and degradation, use `run_validation.py`. This script runs the policy in a closed loop (without Oracle assistance) under various scenarios.

### Usage
```bash
python3 run_validation.py --checkpoint final_jules.pth --agents 200
```

### Scenarios
The script automatically evaluates the following conditions:
1.  **Baseline**: Standard environment with ideal sensors.
2.  **Input Noise**: Adds 1% relative Gaussian noise to all observation inputs.
3.  **Tracking Robustness**: Simulates a lower-quality tracker by:
    - **Decimation**: Rounding pixel coordinates to integer values (VGA resolution).
    - **Holding**: If the target leaves the Field of View, the "last known" valid coordinate is held constant.
4.  **Tracking Noise**: Adds 3-pixel standard deviation Gaussian noise to the tracker coordinates.

Results (metrics and GIFs) are saved to `validation_results/`.

## Environment & Physics

The simulation is built on a custom **Cython-optimized** engine (`drone_env/`) for high throughput.

- **Dynamics**: 6-DOF Quadrotor dynamics with drag and gravity.
- **Environment**:
    - **Wind**: Dynamic wind vectors.
    - **Delays**: Variable communication/actuation delays (0-500ms).
    - **Target**: A virtual target moving in a trajectory (default: circular/wavy).
- **Observation Space (304 dimensions)**:
    - **0-299**: 30-step history of state variables (Position, Velocity, Attitude).
    - **300-303**: Tracker Data:
        - `u, v`: Normalized pixel coordinates of the target in the camera frame ([-1.732, 1.732] for 120° FOV).
        - `size`: Relative size of the target.
        - `conf`: Confidence score.
- **Coordinate System**:
    - **Camera**: Pitched up 30 degrees relative to the body.
    - **Frame**: NED (North-East-Down) aligned.

## Architecture

### Policy Network
- **Input**: 304 dimensions (State History + Tracker).
- **Hidden**: 256 units.
- **Output**: 4 dimensions (Thrust, Roll Rate, Pitch Rate, Yaw Rate).
- **Type**: Deterministic Neural Network (for Supervised Learning).

### Backend
- **Cython**: The default backend (`drone_env/drone_cython.pyx`) uses OpenMP to parallelize physics updates across CPU cores, achieving >10x speedup over pure Python/NumPy.
