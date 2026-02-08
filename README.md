# High-Performance Drone Pursuit Policy

This repository contains the training and validation infrastructure for a neural network policy, as well as a real-time web-based demonstration of the Ghost-DPC control algorithm.

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
- **Observation Space (302 dimensions)**:
    - **0-299**: 30-step history of relevant observables (10 features per step).
        - **0-3**: Control Actions (Thrust, Roll Rate, Pitch Rate, Yaw Rate).
        - **4-6**: Attitude (Yaw, Pitch, Roll).
        - **7**: Altitude (Z).
        - **8-9**: Visual Tracking (u, v).
    - **300-301**: Auxiliary Tracker Data:
        - `size`: Relative size of the target.
        - `conf`: Confidence score.
- **Coordinate System**:
    - **Camera**: Pitched up 30 degrees relative to the body.
    - **Frame**: NED (North-East-Down) aligned.

## Web-Based Homing Demo (`theshow.py`)

The `theshow.py` script runs a real-time simulation of the Ghost-DPC control algorithm performing a full homing mission (Takeoff -> Scan -> Homing -> Land). It provides a web interface to visualize the drone's behavior, internal state, and ghost trajectories.

### Running the Server

To start the simulation server:

```bash
uvicorn theshow:app --host 0.0.0.0 --port 8080
```

Or simply:

```bash
python3 theshow.py
```

Then open your browser to `http://localhost:8080` (or the appropriate IP address).

### Architecture

The system uses a modular architecture decoupled from the web server logic:

*   **Frontend**: HTML5/Three.js interface (in `web/`) for visualizing the drone, target, and "Ghost" predictions.
*   **Backend**: FastAPI with a background task loop (`control_loop`) that advances the simulation.
*   **Simulation**: `SimDroneInterface` wraps the high-performance `DroneEnv` (Cython) to provide realistic physics, wind, and drag.
*   **Control**: `DPCFlightController` implements the Ghost-DPC logic using `PyDPCSolver` (Pure Python).
*   **Perception**: `VisualTracker` provides synthetic vision inputs (blob detection and localization).
*   **Mission**: `MissionManager` handles high-level state transitions (TAKEOFF -> SCAN -> HOMING).

### Current Optimization & Performance

The `ghost_dpc` controller has been optimized for robustness in extreme conditions ("Blind Dive" and "Wind Gusts").

#### 1. Pure Python Gradient-Based MPC
The `PyDPCSolver` is now a pure Python implementation, utilizing analytic gradients for a 12-dimensional state vector (Position, Velocity, Attitude, Angular Rates). This allows for easier debugging and deployment while maintaining real-time performance (30 solver iterations per 50ms step).

#### 2. Robust Cost Function
*   **High Position Gain (`k_pos = 100.0`)**: Provides strong gradient drive to overcome drag and initialization bias, ensuring the drone commits to intercepts.
*   **Scale-Less TTC Barrier**: A safety barrier based on Time-to-Collision ($\tau$) prevents ground impact without creating artificial "floors", allowing for deep dives (gain `200.0`).
*   **Velocity Damping (`k_damp = 0.5`)**: Stabilizes high-speed approaches and reduces overshoot.
*   **Validity Gating**: Visual costs (Gaze and Flow) are strictly gated by Field-of-View constraints to prevent gradient explosions.

#### 3. Adaptive Estimation
The estimator uses a hybrid approach with tuned learning rates to rapidly adapt to unmodeled dynamics:
*   **Wind**: Learning rate `0.5` for immediate reaction to gusts.
*   **Mass & Drag**: Learning rates `0.01` for fast convergence during payload changes.
*   **Velocity Estimation**: Fuses model predictions with optical flow derivatives to estimate velocity without GPS.

#### 4. Performance Benchmarks
*   **Blind Dive**: Intercepts target from 100m altitude with < 7m error.
*   **Wind Gusts**: Holds position with ~0.22m error under strong wind.
*   **Heavy Configuration**: Estimates mass with < 0.01kg error.
