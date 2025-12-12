# Memory Usage Assessment & Capacity Analysis

## Memory Usage Breakdown (Per Agent)

Based on the configuration:
- **Episode Length:** 100 steps
- **Observation Size:** 1804 floats (History + Targets)

### 1. Environment State (Persistent)
| Variable | Dimensions | Size (Floats) | Size (Bytes) |
| :--- | :--- | :--- | :--- |
| `observations` | 1804 | 1,804 | 7.2 KB |
| `pos_history` | 100 * 3 | 300 | 1.2 KB |
| Misc (`vel`, `rot`, `targets`) | ~20 | ~20 | ~0.1 KB |
| **Total Env State** | | **~2,124** | **~8.5 KB** |

### 2. Training Rollout Buffer (Per Episode)
PPO requires storing data for every step of the episode to compute advantages and update the policy.
| Buffer | Dimensions per Step | Total (100 steps) | Size (Bytes) |
| :--- | :--- | :--- | :--- |
| Observations | 1804 | 180,400 | ~722 KB |
| Actions | 4 | 400 | 1.6 KB |
| Rewards | 1 | 100 | 0.4 KB |
| Values | 1 | 100 | 0.4 KB |
| Log Probs | 1 | 100 | 0.4 KB |
| **Total Rollout** | | **~181,100** | **~725 KB** |

### 3. Model Activations (During Update)
During the backward pass, intermediate activations must be stored.
- **Encoder/Decoder & Policy:** Approx. 16,500 floats per agent.
- **Size:** ~66 KB per agent.

### **Total Memory Per Agent**
- **Conservative Estimate:** Env State + Rollout Buffer + Activations
- **Total:** 8.5 KB + 725 KB + 66 KB ≈ **800 KB per agent**

---

## Actual Memory Benchmark
Measured on CPU with full training buffers allocated (PPO Rollout + Model + Env).

| Agents | Memory (MB) | Est. per Agent (MB) |
|---|---|---|
| 100 | 701.42 | ~7.01 (High overhead) |
| 1,000 | 1,330.06 | ~1.33 |
| 5,000 | 4,121.23 | ~0.82 |
| 10,000 | 7,577.60 | ~0.76 |
| **20,000** | **OOM (Failed to alloc 14GB)** | |

**Conclusion:** The practical limit on a 16GB system is between 10k and 20k agents.
Scaling to 1M agents would require ~760 GB of RAM/VRAM.

---

## Maximum Capacity Estimates (Theoretical)
Assuming strictly linear scaling of 0.8 MB/agent.

| Hardware | VRAM | Max Agents (Est.) |
| :--- | :--- | :--- |
| **Consumer GPU (8 GB)** | 8 GB | ~10,000 |
| **High-End Consumer (24 GB)** | 24 GB | ~30,000 |
| **Data Center (80 GB)** | 80 GB | ~100,000 |

### Recommendations to Scale Further
1.  **Reduce Observation Size:** 1804 floats is very large. Compressing the 300-step history (e.g., storing only every 5th step or using a lower precision format like float16) would linearly reduce the rollout buffer size.
2.  **Float16 / Mixed Precision:** Switching to `float16` for buffers would halve the memory usage.

## Optimization Implemented
1.  **Removed `imu_history` Redundancy:** Saves 7.2 KB per agent (Env State). Small but helpful.
2.  **Optimizer Separation:** Separated RL and AE optimizers to fix stability bugs and ensure correct loss handling.
