# Memory Usage Assessment

## Current Memory Usage (Per Agent)

Based on the configuration:
- **Agents:** 1024 (1 agent per env, 1024 envs)
- **Episode Length:** 100 steps
- **History Length:** 300 steps (3 seconds at 100Hz)

### Environment Memory (GPU/CPU State)
| Variable | Dimensions | Size (Floats) | Size (Bytes) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `imu_history` | 300 * 6 | 1,800 | 7,200 | **Redundant** (copied to obs) |
| `observations` | 1804 | 1,804 | 7,216 | Contains history + targets |
| `pos_history` | 100 * 3 | 300 | 1,200 | For visualization |
| `step_counts`, `rewards`, etc. | ~20 | ~20 | ~80 | Negligible |
| **Total Env** | | **~3,924** | **~15.7 KB** | |

### Model Memory (Training)
- **Parameters:** ~100k floats (~400 KB shared).
- **Activations:** ~16,000 floats (~64 KB per agent) due to large Conv1D layers and dense layers processing the 1800-dim history.

### Total per Agent
- **~80 KB** per agent.
- For 1024 agents: ~80 MB.
- For 1,000,000 agents: ~80 GB.

## Diagnosis & Recommendations

1.  **Remove `imu_history` Redundancy:**
    - The `imu_history` array tracks the sliding window of IMU data, which is then copied entirely into `observations` at every step.
    - **Optimization:** Perform the sliding window update directly within the `observations` array.
    - **Savings:** Saves 7.2 KB per agent (approx 50% of environment state memory).

2.  **Reduce History Length:**
    - Currently 300 steps (3 seconds).
    - If reduced to 100 steps (1 second):
        - `observations` size reduces from 1804 to 604.
        - Model activations reduce significantly (approx 3x).
    - **Savings:** ~4.8 KB per agent in Env, plus ~40 KB per agent in Model Activations.

3.  **Optimize `pos_history`:**
    - Currently stores full episode trajectory in GPU memory.
    - If visualization is infrequent, we can remove this from GPU state and only log current position, accumulating it on CPU (though this slows down data transfer if done every step).
    - **Verdict:** Keep it for now as it's small (1.2 KB).
