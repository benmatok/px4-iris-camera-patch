# High-Performance Drone Pursuit Policy

This repository contains a real-time web-based demonstration of the Ghost-DPC control algorithm.

## Overview

Ghost-DPC is a model-based control architecture that handles extreme physical uncertainty (e.g., wind shear, payload drops, battery failure) by maintaining a "Fan of Beliefs". Instead of trying to learn one perfect policy, it runs multiple internal physics simulations ("Ghosts") in parallel—each representing a different hypothesis (e.g., "Heavy Drone", "Crosswind Left", "Normal Flight").

## Installation

### Prerequisites
- **Python 3.8+**
- **C++ Compiler**: `g++` or `clang++` with OpenMP support.
- **Python Packages**: Listed in `requirements.txt`.

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

## Web-Based Homing Demo (`theshow.py`)

The `theshow.py` script runs a real-time simulation of the Ghost-DPC control algorithm performing a full homing mission (Takeoff -> Scan -> Homing -> Land). It provides a web interface to visualize the drone's behavior, internal state, and ghost trajectories.

### Running the Server

To start the simulation server, first ensure your environment is set up and dependencies are installed:

```bash
# 1. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

Then, run the server:

```bash
uvicorn theshow:app --host 0.0.0.0 --port 8080
```

Or simply:

```bash
python3 theshow.py
```

Finally, open your browser to `http://localhost:8080` (or the appropriate IP address).

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
