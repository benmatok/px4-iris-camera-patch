# Ghost-DPC: Differentiable Predictive Control with "Ghost" Models

## Overview
Ghost-DPC is a model-based control architecture that handles **extreme physical uncertainty** (e.g., wind shear, payload drops, battery failure) by maintaining a "Fan of Beliefs". Instead of trying to learn one perfect policy, it runs multiple internal physics simulations ("Ghosts") in parallel—each representing a different hypothesis (e.g., "Heavy Drone", "Crosswind Left", "Normal Flight").

A differentiable solver optimizes a single control sequence that minimizes the **expected cost** across these weighted ghost models. The weights are updated in real-time by an **MMAE (Multiple Model Adaptive Estimation)** block based on prediction errors.

---

## Core Components
### 1. The Ghost Estimator (MMAE)
*   **Role:** The "Brain's Belief System".
*   **Logic:** Compares the real drone's acceleration against each Ghost Model's prediction.
*   **Math:**
    $$ w_i(t+1) = \frac{w_i(t) \cdot p(y|x, \theta_i)}{\sum w_j \cdot p(y|x, \theta_j)} $$
    where $p(y|...)$ is the likelihood of the observed acceleration given model $\theta_i$.

### 2. The Ghost Solver (Gradient-Based MPC)
*   **Role:** The "Action Optimizer".
*   **Logic:** Finds a thrust/attitude sequence that survives the *worst-case* plausible scenario (or weighted average).
*   **Math:** Backpropagates gradients through the physics engine to minimize:
    $$ J = \sum_{t=0}^H \sum_{i=1}^N w_i \cdot \text{Cost}(\text{GhostState}_i(t), \text{Target}) $$

### 3. The Differentiable Physics Engine
*   **Role:** The "Imagination".
*   **Logic:** A lightweight, analytical derivatives-enabled simulator (written in C++/Cython) that predicts the next state given a state and action.

---

## Sprint: Red Object Tracking Integration

This sprint focuses on integrating the Ghost-DPC core with a vision system to track a red object in Gazebo. The work is divided into 4 independent, verifiable phases.

### Phase 1: The "Brain" on the Bench (Core Kernel)
**Goal:** Validate the C++ math and physics models in isolation before touching any simulation or robot.

*   **Step 1.1: Gradient Check Unit Test** [x]
    *   **File:** `tests/test_ghost_gradients.py`
    *   **Logic:**
        *   Instantiate `PyGhostModel`.
        *   Compute analytical gradients using `get_gradients(state, action)`.
        *   Compute numerical gradients using finite differences: `(step(x+h) - step(x-h)) / 2h`.
    *   **Critical Validation:**
        *   **Command:** `pytest tests/test_ghost_gradients.py -v`
        *   **Pass Criteria:** Max relative error between analytical and numerical gradients must be **< 1e-5**.
        *   **Sanity Check:** Ensure gradients are *non-zero* for Thrust and Pitch (if pitch != 0).
        *   **Failure Mode:** If error > 1e-2, check `ghost_model.hpp` for missing terms in the Jacobian chain rule (especially angular rate couplings).

*   **Step 1.2: Estimator Convergence Test** [x]
    *   **File:** `tests/test_estimator.py`
    *   **Logic:**
        *   Create `PyGhostEstimator` with "Light" (1.0kg) and "Heavy" (2.0kg) models.
        *   Generate synthetic trajectory using "Heavy" physics.
        *   Update estimator for 20 steps.
    *   **Critical Validation:**
        *   **Command:** `python3 tests/test_estimator.py`
        *   **Pass Criteria:**
            1.  Step 0: Weights are `[0.5, 0.5]`.
            2.  Step 20: "Heavy" weight is **> 0.95**.
            3.  Plot shows monotonic increase in "Heavy" probability.
        *   **Failure Mode:** If weights oscillate or stay 0.5, check the `likelihood` function variance (sigma). It might be too large (ignoring errors) or too small (numerical underflow).

*   **Step 1.3: Solver Latency Benchmark** [x]
    *   **File:** `tests/benchmark_solver.py`
    *   **Logic:**
        *   Run `PyDPCSolver.solve()` 1,000 times in a loop.
    *   **Critical Validation:**
        *   **Command:** `python3 tests/benchmark_solver.py`
        *   **Pass Criteria:** Mean execution time **< 1.5ms** on development CPU.
        *   **Hard Limit:** Max time must never exceed **20ms** (system tick is 50Hz).
        *   **Failure Mode:** If > 2ms, verify `ghost_dpc.pyx` is compiled with `-O3` and `-march=native`.

### Phase 2: The "Nervous System" (Simulation Bridge)
**Goal:** Ensure data flows correctly between Gazebo/PX4 and your Python script.

*   **Step 2.1: Telemetry Sanity Check** [x]
    *   **File:** `tools/check_telemetry.py`
    *   **Logic:**
        *   Subscribe to MAVSDK telemetry.
        *   Log `Position (NED)`, `Attitude (Euler)`, `IMU (Accel)`.
    *   **Critical Validation:**
        *   **Command:** `python3 tools/check_telemetry.py` (while Simulator is running)
        *   **Pass Criteria:**
            1.  **Stationary on Ground:** `Alt` ≈ 0.0m, `Vel` ≈ 0.0 m/s.
            2.  **Gravity Sign:** `IMU.z` must be approximately **-9.8 m/s²** (assuming NED frame) or **+9.8 m/s²** (Body frame).
                *   *Action:* Lift the drone manually in Gazebo. `Alt` must increase.
                *   *Action:* Tilt drone Nose Down. `Pitch` must be positive (if using standard aero convention).
        *   **Failure Mode:** If `IMU.z` is near 0, physics is paused. If `Pitch` sign is flipped, control will invert (crash).

*   **Step 2.2: The "Wiggle" Test (Actuation)** [x]
    *   **File:** `tools/test_actuation.py`
    *   **Logic:**
        *   Arm -> Offboard.
        *   Command `Roll = +0.1 rad` (5s) -> `Roll = -0.1 rad` (5s).
    *   **Critical Validation:**
        *   **Command:** `python3 tools/test_actuation.py`
        *   **Pass Criteria:**
            1.  Drone visually tilts **Right**, holds, then tilts **Left**.
            2.  Must not gain altitude > 1m (Thrust should be idle/hover).
        *   **Failure Mode:** If drone spins (Yaw) instead of Rolling, check `Offboard` message mapping (RPY vs Quaternion).

### Phase 3: The "Eyes" (Perception Layer)
**Goal:** Build and verify the vision system.

*   **Step 3.1: Red Blob Detector** [x]
    *   **File:** `vision/detector.py`
    *   **Logic:** HSV Thresholding + Contour extraction.
    *   **Critical Validation:**
        *   **Command:** `python3 vision/detector.py --test-image assets/red_ball.jpg`
        *   **Pass Criteria:**
            1.  Output image saved to `validation_results/detection_debug.jpg`.
            2.  Green bounding box tightly encompasses the red ball.
            3.  Console prints `Found object: Center=(320, 240), Area=1500`.
        *   **Failure Mode:** False positives on shadows? Adjust HSV lower bound `V` (Value) > 50.

*   **Step 3.2: Ray-Casting Math** [x]
    *   **File:** `vision/projection.py`
    *   **Logic:** Pinhole Camera Model + Homography to $z=0$ plane.
    *   **Critical Validation:**
        *   **Command:** `pytest tests/test_projection.py`
        *   **Pass Criteria:**
            1.  **Center Case:** Drone @ (0,0,10m), Pixel (Center) -> World (0,0). Error < 0.01m.
            2.  **Corner Case:** Drone @ (0,0,10m), Pixel (Top-Left) -> World (-X, +Y) (Verify signs based on FOV).
            3.  **Rotation Case:** Drone Pitch +45 deg, Pixel (Center) -> World (+X, 0).
        *   **Failure Mode:** If World coordinates are huge (>1000m) or negative when they shouldn't be, check for `tan(0)` or division by zero in the projection ray.

### Phase 4: Closed-Loop Integration (The "Gauntlet")
**Goal:** Autonomous flight.

*   **Step 4.1: The "Z-Hold" Test** [x]
    *   **File:** `video_viewer.py` (Controller Node)
    *   **Logic:** Target = `[CurX, CurY, 5.0]`. Vision Disabled.
    *   **Critical Validation:**
        *   **Command:** `python3 video_viewer.py --mode hover`
        *   **Pass Criteria:**
            1.  Drone climbs to 5.0m.
            2.  **Stability:** Altitude error stays within **+/- 0.2m** for 60 seconds.
            3.  **Drift:** XY Position drift < 1.0m (without GPS, this relies on Optical Flow/Vision, or Simulator Truth).
        *   **Failure Mode:** Oscillations > 1Hz indicate `D-gain` is too low. Slow wander indicates `I-gain` needed or `P-gain` too low.

*   **Step 4.2: The "Statue" Test (Stationary Tracking)** [x]
    *   **File:** `video_viewer.py`
    *   **Logic:** Vision Enabled. Target Red Object @ (10, 10).
    *   **Critical Validation:**
        *   **Command:** `python3 video_viewer.py --mode track`
        *   **Pass Criteria:**
            1.  Drone moves from (0,0) to approx (10,10).
            2.  **Visual Lock:** The red object remains in the center 20% of the camera frame.
            3.  **Convergence:** Reaches target within 10 seconds.
        *   **Failure Mode:** If drone circles the target, verify Camera-to-Body rotation matrix.

*   **Step 4.3: The "Bullfight" (Dynamic Tracking)** [x]
    *   **File:** `video_viewer.py`
    *   **Logic:** Move target manually.
    *   **Critical Validation:**
        *   **Command:** `python3 video_viewer.py --mode track`
        *   **Pass Criteria:**
            1.  **Response Time:** Drone reacts to target movement within < 0.5s.
            2.  **No Overshoot:** When target stops, drone stops without passing it by > 1.0m.
            3.  **Robustness:** Tracking holds even if target moves at 2 m/s.
        *   **Failure Mode:** If drone loses target, check "Lost & Found" logic (should hover or search, not crash).
