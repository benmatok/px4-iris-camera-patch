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

*   **Step 1.1: Gradient Check Unit Test**
    *   **File:** `tests/test_ghost_gradients.py`
    *   **Logic:**
        *   Instantiate `PyGhostModel` (from `ghost_dpc`).
        *   Compute analytical gradients using `get_gradients(state, action)`.
        *   Compute numerical gradients using finite differences: `(step(x+h) - step(x-h)) / 2h`.
    *   **Validation:** Assert `norm(analytical - numerical) < 1e-5`. If this fails, the controller will crash or diverge.

*   **Step 1.2: Estimator Convergence Test**
    *   **File:** `tests/test_estimator.py`
    *   **Logic:**
        *   Create `PyGhostEstimator` initialized with two models: "Light" (mass=1.0) and "Heavy" (mass=2.0).
        *   Simulate a "Heavy" drone trajectory (ground truth) using `PyGhostModel`.
        *   Feed the state/action/acceleration history into the estimator using `estimator.update()`.
    *   **Validation:** Assert that the probability weight for the "Heavy" model crosses **0.9** within 20 simulation steps.

*   **Step 1.3: Solver Latency Benchmark**
    *   **File:** `tests/benchmark_solver.py`
    *   **Logic:**
        *   Initialize `PyDPCSolver`.
        *   Run `solve()` 1,000 times in a loop with random valid initial states and targets.
    *   **Validation:** Assert average execution time is **< 1ms** (or acceptable bounds for 50Hz control).

### Phase 2: The "Nervous System" (Simulation Bridge)
**Goal:** Ensure data flows correctly between Gazebo/PX4 and your Python script. Ignore control logic for now.

*   **Step 2.1: Telemetry Sanity Check**
    *   **File:** `tools/check_telemetry.py`
    *   **Logic:**
        *   Connect to MAVSDK (`udp://:14540`).
        *   Subscribe to `telemetry.position()`, `telemetry.attitude_euler()`, and `telemetry.imu()`.
        *   Print these values to stdout at ~10Hz.
    *   **Validation:**
        *   Altitude should be ~0m when on the ground.
        *   **Crucial:** Check `IMU Z-Accel`. Is it +9.8 or -9.8? This determines if gravity needs sign flipping in the C++ kernel.

*   **Step 2.2: The "Wiggle" Test (Actuation)**
    *   **File:** `tools/test_actuation.py`
    *   **Logic:**
        *   Connect via MAVSDK, Arm, and switch to Offboard mode.
        *   Send a `set_attitude` command with `roll_deg=5.0` (approx 0.1 rad) for 2 seconds.
        *   Send `roll_deg=0.0` to level out.
    *   **Validation:** Visual confirmation in Gazebo. The drone should twitch to the right and then level out. This confirms the command pipeline works.

### Phase 3: The "Eyes" (Perception Layer)
**Goal:** Build the vision system using the Gazebo camera feed.

*   **Step 3.1: Red Blob Detector**
    *   **File:** `vision/detector.py`
    *   **Logic:**
        *   Implement `detect_red_object(cv2_image) -> (u, v, bbox_area)`.
        *   Convert BGR to HSV.
        *   Apply `cv2.inRange` for Red (handling the hue wrap-around: 0-10 and 160-180).
        *   Find largest contour using `cv2.findContours`.
        *   Return center centroid `(cx, cy)` of the bounding rect.
    *   **Validation:** `tests/test_vision_static.py`. Load a synthetic image with a red circle. Assert the function returns the correct pixel coordinates.

*   **Step 3.2: Ray-Casting Math**
    *   **File:** `vision/projection.py`
    *   **Logic:**
        *   Implement `screen_to_world(u, v, drone_state, camera_params) -> (world_x, world_y)`.
        *   Model the camera as a pinhole.
        *   Apply rotation matrix (from drone attitude) to the camera vector.
        *   Intersect the vector with the ground plane ($z=0$).
    *   **Validation:** `tests/test_projection.py`.
        *   Mock State: Drone at $(0,0,10)$, looking straight down.
        *   Mock Input: Center pixel $(u=W/2, v=H/2)$.
        *   Assert: Output is approx $(0,0)$.
        *   Mock Input: Offset pixel. Assert output shifts in the correct direction.

### Phase 4: Closed-Loop Integration (The "Gauntlet")
**Goal:** Connect Phase 1, 2, and 3. This is the first time the drone flies autonomously using the full stack.

*   **Step 4.1: The "Z-Hold" Test**
    *   **File:** `video_viewer.py` (evolving into `controller_node.py`)
    *   **Logic:**
        *   Integrate `PyGhostEstimator` and `PyDPCSolver`.
        *   Set a static target: `[Current_X, Current_Y, 5.0]`.
        *   Feed telemetry to Estimator -> Solver -> Control.
        *   Ignore vision input for this step.
    *   **Validation:** Drone takes off and holds 5m altitude stably. If it oscillates, tune `D` gains or Estimator noise parameters.

*   **Step 4.2: The "Statue" Test (Stationary Tracking)**
    *   **File:** `video_viewer.py`
    *   **Logic:**
        *   Hover at 5m.
        *   Spawn red object at `x=10` in Gazebo.
        *   Enable Vision: Feed `vision/detector.py` output -> `vision/projection.py` -> Target for Solver.
    *   **Validation:** Drone flies to `x=10` and hovers above the object.

*   **Step 4.3: The "Bullfight" (Dynamic Tracking)**
    *   **File:** `video_viewer.py`
    *   **Logic:**
        *   Same as 4.2, but manually move the red object in Gazebo using the "Move" tool.
    *   **Validation:** Drone chases the object. The "Future Horizon" lines (debug visualization) should point toward the moving target.
