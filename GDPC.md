# Ghost-DPC Adaptive Homing System (GDPC)

**Date:** 2023-10-27
**Module:** `ghost_dpc`
**Status:** Validated / Prototype

---

## 1. General Concept

**Ghost-DPC** is a robust, adaptive control kernel designed for high-speed homing and tracking in uncertain physical environments. It solves the "Black Box" control problem by explicitly modeling uncertainty using a **"Fan of Beliefs"**:

1.  **Multiple Model Adaptive Estimation (MMAE):** A bank of parallel "Ghost" physics models runs in real-time, each representing a different physical hypothesis (e.g., Heavy Payload, Broken Motor, Crosswind). By comparing predicted acceleration against real IMU data, the system identifies the true physical state within milliseconds.
2.  **Differentiable Predictive Control (DPC):** Instead of a single optimal trajectory, the controller optimizes a single action sequence that minimizes cost across *multiple* weighted future scenarios (Optimistic, Headwind, Crosswind). This produces a "Robust Average" control that is safe even if the exact wind condition is unknown.
3.  **Analytical Gradients:** The entire pipeline uses analytically derived gradients (Jacobians) for the physics dynamics, avoiding the overhead and "black box" nature of Autograd frameworks (PyTorch/TensorFlow). This ensures determinism and microsecond-scale performance.

---

## 2. Implementation Details

The system is implemented as a standalone C++ kernel with Python bindings via Cython.

*   **`ghost_model.hpp` (Physics Engine):**
    *   Implements Symplectic Euler integration.
    *   **Analytical Gradients:** Computes the Jacobian $\frac{\partial \mathbf{S}_{t+1}}{\partial \mathbf{u}_t}$ (9x4 matrix) and $\frac{\partial \mathbf{S}_{t+1}}{\partial m}$ (9x1 vector) manually. This is the foundation of the gradient-based solver.
    *   **State:** Position (3), Velocity (3), Attitude (Roll, Pitch, Yaw).
    *   **Action:** Thrust (0-1), Roll Rate, Pitch Rate, Yaw Rate.

*   **`ghost_estimator.hpp` (MMAE Estimator):**
    *   Maintains a bank of `GhostModel` instances (e.g., Nominal, Heavy, Light).
    *   **Bayesian Update:** Updates the probability $P(M_i | z_t)$ based on the prediction error $||\mathbf{a}_{measured} - \mathbf{a}_{predicted}||^2$.
    *   Converges to the correct physical model in < 20 steps (approx 1s at 20Hz).

*   **`dpc_solver.hpp` (Gradient-Based MPC):**
    *   **Horizon:** 10 steps (approx 0.5s lookahead).
    *   **Optimization:** Performs 10 iterations of Gradient Descent per control cycle.
    *   **Cost Function:** Weighted sum of Distance to Target, Altitude Safety (Target Z + 2m), and Descent Rate constraints.
    *   Optimizes against a weighted ensemble of wind hypotheses (Nominal, +Crosswind, -Crosswind).

*   **`ghost_dpc.pyx`:** Cython wrapper exposing the C++ classes to Python as `PyGhostModel`, `PyGhostEstimator`, and `PyDPCSolver`.

---

## 3. Validation

We validate the system using a "Gauntlet" of three automated scenarios in `run_ghost_validation.py`.

### A. Payload Drop (Step Response)
*   **Scenario:** Drone mass drops from 1.0kg to 0.5kg instantly at t=2.0s.
*   **Result:** Estimator probability for "Light Model" shoots to >99% within 0.5s. Controller adapts thrust to prevent ballooning.

### B. Dying Battery (Ramp Response)
*   **Scenario:** Thrust coefficient decays linearly from 1.0 to 0.7 over 10s.
*   **Result:** Controller automatically increases throttle command to maintain altitude as the motor efficiency drops.

### C. Blind Dive (Wind Compensation)
*   **Scenario:** Target is deep below/forward. Unmodeled 10m/s crosswind.
*   **Result:** DPC Solver detects drift via the physics mismatch (indirectly via estimator or robust constraints) and angles the drone (Crab) to maintain the ground track.

---

## 4. Example Usage

```python
from ghost_dpc.ghost_dpc import PyGhostEstimator, PyDPCSolver

# 1. Initialize Estimator with Hypotheses
models = [
    {'mass': 1.0, 'drag': 0.1, 'thrust_coeff': 1.0}, # Nominal
    {'mass': 1.5, 'drag': 0.1, 'thrust_coeff': 1.0}, # Heavy
]
estimator = PyGhostEstimator(models)
solver = PyDPCSolver()

# Control Loop
while True:
    # Update Estimator with IMU Data
    # measured_accel = [ax, ay, az] from IMU
    estimator.update(state, last_action, measured_accel, dt=0.05)

    # Get Robust Beliefs
    weighted_model = estimator.get_weighted_model()

    # Fork Hypotheses for Solver (e.g. Add Wind Uncertainty)
    solver_models = [weighted_model, weighted_model_with_wind]
    weights = [0.8, 0.2]

    # Solve for Optimal Action
    action = solver.solve(state, target_pos, last_action, solver_models, weights, dt=0.05)

    # Apply action
    drone.apply(action)
```

---

## 5. Gazebo Integration Plan

To deploy this on a ROS2/Gazebo drone (e.g., PX4/ArduPilot):

1.  **Bridge Node:** Create a C++ ROS2 node `ghost_dpc_node`.
2.  **Subscriptions:**
    *   `/mavros/local_position/odom` -> Maps to `px, py, pz, vx, vy, vz, orientation`.
    *   `/mavros/imu/data` -> Maps to `measured_accel`.
3.  **Control Loop (Timer @ 50Hz):**
    *   Convert ROS Msg -> `GhostState`.
    *   Run `estimator.update()`.
    *   Run `solver.solve()`.
    *   **Output:** Publish to `/mavros/setpoint_raw/attitude` (Target Roll/Pitch/YawRate/Thrust).
4.  **Watchdog:** If solver diverges or computation exceeds 15ms, fallback to a safe hover mode.

---

## 6. Horizon and Compute

*   **Current Horizon:** 10 steps (~0.5s at dt=0.05).
*   **Compute Load:** Very Low. < 1ms per solve on CPU (Single Core).
*   **Ways to Increase Horizon without High Compute:**
    *   **Variable Time Steps:** Use `dt=0.05` for first 5 steps, then `dt=0.2` for next 5. This extends prediction to ~1.25s with same compute.
    *   **Control Splines:** Optimize 3 control points of a spline instead of 10 independent actions.
    *   **Terminal Cost:** Train a lightweight Neural Network (Value Function) to approximate the cost-to-go beyond the horizon, effectively infinite horizon.

---

## 7. Next Steps (Robustness & Performance)

1.  **Noise Modeling:** The current estimator assumes perfect state observation. Add Gaussian noise handling to the Bayesian update (Kalman Filter-like covariance update).
2.  **SIMD Optimization:** The current C++ implementation is scalar. Porting the Jacobian computation to AVX2 (processing 8 hypotheses at once) would provide 4-6x speedup.
3.  **Wind Field Estimation:** Instead of just discrete wind hypotheses, add `wind_x`, `wind_y` as continuous state variables in the solver state vector and estimate them via the Jacobian.

---

## 8. High-Level Control (The "Brain")

Ghost-DPC is a "Reflex" controller (low-level). To build a full autonomy stack:

1.  **Finite State Machine (FSM):**
    *   **IDLE:** Motors off.
    *   **TAKEOFF:** Target Z = Current Z + 5m. Wait for altitude.
    *   **SEARCH:** Spiral pattern target generation.
    *   **TRACK:** Feed detection coordinates to DPC Solver target.
    *   **LAND:** DPC Target = Ground. Descent rate constraint active.
2.  **Visual Servoing:** Map pixel coordinates $(u, v)$ directly to the cost function. If the target moves in the image, the gradient pulls the drone to center it.
3.  **Safe Corridors:** Add inequality constraints (barriers) to the cost function to prevent entering "No-Fly Zones" (e.g., `Cost += exp(-distance_to_obstacle)`).

---

## 9. Fixed-Wing Adaptation

To use this for a Plane/UAV:

1.  **Physics Model (`GhostModel`):**
    *   Replace Quadrotor dynamics with Fixed-Wing dynamics.
    *   Lift $L = C_L \frac{1}{2} \rho v^2 S$, Drag $D = C_D \frac{1}{2} \rho v^2 S$.
    *   State must include **Airspeed** and **Angle of Attack**.
2.  **Action Space:**
    *   Replace (Thrust, Rates) with (Throttle, Aileron, Elevator, Rudder).
3.  **Constraints:**
    *   **Stall Prevention:** Add a massive cost penalty if $v < v_{stall}$.
    *   **Bank Limit:** Constrain roll angle to $\pm 45^\circ$ for coordinated turns.
4.  **Solver:** The Gradient Descent logic remains exactly the same; only the `step()` and `get_gradients()` functions change to reflect the new equations of motion.

---

## 10. Sprint: Red Object Tracking Integration

**Goal:** Implement an automatic scenario where the drone detects and tracks a red object on the ground using the Ghost-DPC controller.

### Gap Analysis & Missing Components

The current Python scripts (`imu_test.py`, `keyboard_control.py`, `motor_test.py`, `video_viewer.py`) provide basic building blocks but lack the integration required for autonomous tracking.

*   **`video_viewer.py` (To be upgraded to `controller_node.py`):**
    *   **Missing: Object Detection:** Needs OpenCV logic to identify red pixels, compute centroid $(u, v)$, and bounding box.
    *   **Missing: Coordinate Projection:** Needs a camera model (pinhole) to project the 2D pixel $(u, v)$ to a 3D ray and intersect it with the ground plane ($z=0$) to generate a 3D target position for the solver.
    *   **Missing: State Estimation:** Needs to subscribe to MAVSDK telemetry (Orientation, Velocity, Acceleration) and feed it into `PyGhostEstimator`.
    *   **Missing: Control Loop:** Needs to instantiate `PyDPCSolver`, update it with the estimated state and projected target, and publish the resulting `Attitude` setpoints.

*   **`imu_test.py`:**
    *   **Status:** Functional. Useful for verifying IMU data integrity before feeding the estimator.

*   **`keyboard_control.py`:**
    *   **Status:** Functional. Useful for manual override and safety pilot during testing.

*   **`motor_test.py`:**
    *   **Status:** Functional. verifies actuation/mixing.

### Integration Plan

#### Phase 1: Perception & State
1.  **Red Object Detector:** Implement HSV color thresholding in `video_viewer.py`.
    *   Input: `cv::Mat` (BGR).
    *   Output: `target_u`, `target_v`, `confidence`.
2.  **Raycasting:** Implement `screen_to_world(u, v, altitude, attitude)` function.
    *   Assumptions: Flat ground at $z=0$, Camera pointing forward/down.
3.  **State Monitor:** Create an async class to buffer the latest MAVSDK telemetry for the synchronous control loop.

#### Phase 2: Control Integration
1.  **Ghost-DPC Bindings:** Import `ghost_dpc` in the Python node.
2.  **Estimator Setup:** Initialize `PyGhostEstimator` with nominal, heavy, and draggy hypotheses.
3.  **Solver Loop:**
    *   Frequency: 20Hz.
    *   Step 1: Update Estimator with IMU data.
    *   Step 2: Project visual target to 3D world coordinates.
    *   Step 3: Run `solver.solve()` targeting the 3D position.
    *   Step 4: Send `Attitude` command to PX4.

#### Phase 3: Simulation Scenario
1.  **Gazebo World:** Ensure a red box or sphere is spawned at $(x=10, y=0)$ in the simulation world.
2.  **Automated Start:** The script should arm, takeoff to 5m, and switch to OFFBOARD mode automatically.
