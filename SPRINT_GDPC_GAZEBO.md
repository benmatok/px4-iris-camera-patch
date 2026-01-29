# Sprint Plan: GDPC Integration with Gazebo

## Executive Summary
This 4-week sprint aims to migrate the "Ghost" Differentiable Predictive Control (GDPC) system from the custom `DroneEnv` simulator to a high-fidelity Gazebo/PX4 environment. The goal is to validate the control logic against realistic aerodynamics and sensor noise.

## Prerequisites
- **Environment:** Docker container defined in `Dockerfile` (Ubuntu 22.04, ROS2 Humble, PX4 v1.14.0).
- **Physics:** Modified Iris airframe (patched via `apply_patch.sh` to match "Chimera CX10" dynamics).
- **Controller:** GDPC C++ core wrapped via `ghost_dpc/ghost_dpc.pyx`.

---

## Week 1: Infrastructure & Bridge
**Goal:** Establish a closed-loop control loop where GDPC drives the simulated drone in Gazebo.

### Tasks
1.  **MAVSDK Wrapper (`gazebo_env.py`)**
    -   Create a Python class that mimics the interface of `DroneEnv` but wraps `mavsdk.System`.
    -   Implement `reset()`: Arm, takeoff, and hover using MAVSDK Offboard mode.
    -   Implement `step(action)`: Convert GDPC `GhostAction` (Thrust, Rates) to MAVSDK `AttitudeRate` commands.
2.  **State Mapping**
    -   Subscribe to MAVSDK telemetry (`telemetry.position`, `telemetry.velocity_ned`, `telemetry.attitude_euler`).
    -   Map these streams to the `state_dict` format required by `PyGhostEstimator.update()` (px, py, pz, vx, vy, vz, roll, pitch, yaw).
3.  **Camera Integration**
    -   Verify the ROS2 bridge (`ros-humble-cv-bridge`) connects to the camera defined in `apply_patch.sh`.
    -   Ensure image data is accessible in the Python control loop (for future visual tracking).
4.  **"Hello World" Flight**
    -   Script a simple hover test where GDPC maintains altitude using the mapped telemetry.

**Deliverable:** A script `test_bridge.py` that successfully hovers the drone in Gazebo using GDPC logic.

---

## Week 2: "Ghost" Tuning & State Estimation
**Goal:** Calibrate the Multiple Model Adaptive Estimator (MMAE) to function correctly with Gazebo's physics engine.

### Tasks
1.  **Model Parameter Tuning**
    -   Update `PyGhostEstimator` initialization in the control script.
    -   Set "Nominal" model parameters to match `apply_patch.sh`:
        -   **Mass:** 0.826 kg (Base) + Payload (if attached).
        -   **Inertia:** Scale appropriately based on the patch values.
        -   **Thrust Coefficient:** Calibrate based on the `motorConstant` (5.84e-05) and prop size.
2.  **Estimator Validation**
    -   Run "Open Loop" tests: Fly random trajectories in Gazebo.
    -   Feed telemetry to `PyGhostEstimator`.
    -   Compare `estimator.get_weighted_model()` against known Gazebo parameters.
    -   **Success Criteria:** Estimator converges to the correct mass/drag model within 2 seconds of a change.
3.  **Noise Robustness**
    -   Analyze Gazebo IMU noise levels.
    -   Adjust `PyGhostEstimator` process noise covariance (if exposed) or smoothing factors to prevent jitter.

**Deliverable:** Validated `GhostEstimator` configuration that correctly identifies the drone's physical parameters in simulation.

---

## Week 3: Scenario Porting
**Goal:** Recreate the specific validation scenarios from `run_dpc_validation.py` in the Gazebo environment.

### Tasks
1.  **Scenario A: Wind Shear**
    -   **Implementation:** Use MAVSDK `Param` interface or Gazebo Wind Plugin.
    -   **Test:** Inject 5-10 m/s lateral wind gusts.
    -   **Verify:** GDPC "Fan of Beliefs" should identify the wind and crab the drone to compensate.
2.  **Scenario B: Stall / Motor Failure**
    -   **Implementation:** Use MAVSDK `Failure` plugin (if available) or manually limit thrust commands in the bridge to simulate power loss.
    -   **Test:** Trigger failure at altitude.
    -   **Verify:** GDPC should prioritize recovery (pitch down) over tracking.
3.  **Scenario C: Lost & Found (Visual)**
    -   **Implementation:** Teleport the visual target in Gazebo out of the camera FOV.
    -   **Verify:** Drone should execute the search pattern (spiral/scan) defined in the policy.

**Deliverable:** Python scripts `run_gazebo_wind.py`, `run_gazebo_stall.py`, etc.

---

## Week 4: Validation & Benchmarking
**Goal:** Comprehensive performance analysis and comparison.

### Tasks
1.  **Automated Test Suite**
    -   Create a runner that executes all Week 3 scenarios sequentially.
    -   Log `pos_error`, `survival_rate`, and `computational_time`.
2.  **Comparison Analysis**
    -   Compare Gazebo results vs. `validation_results/` (Custom Sim).
    -   Identify discrepancies (e.g., is Gazebo drag higher? Is latency an issue?).
3.  **Final Report**
    -   Generate plots similar to `validation_ghost_wars.png` but using Gazebo data.
    -   Document tuning parameters and lessons learned.

**Deliverable:** `GAZEBO_VALIDATION_REPORT.md` and associated plots.
