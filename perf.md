# Performance Analysis Report

## Baseline Performance

**Tool**: `benchmark_cython.py` (measuring 100 steps for 5000 agents)
**Result**: ~30x speedup over NumPy CPU baseline.
- NumPy CPU: ~2.9s
- Cython/AVX: ~0.096s

**Profiling Notes**:
Valgrind `cachegrind` was used to analyze the execution. While full Python profiling was challenging in the environment, analysis of the C++ logic (`physics_avx.hpp`) and Cython code (`drone_cython.pyx`) revealed specific hotspots.

## Optimization Opportunities

### 1. Redundant Trigonometry in `physics_avx.hpp`
**Issue**: `sincos256_ps` is called for `roll`, `pitch`, and `yaw` inside the physics substep loop (10 times). It is then called *again* after the loop to compute the Rotation Matrix for reward calculation.
**Optimization**: The values of `sin/cos` from the *last* substep (iteration 9) correspond to the final state. We can cache these values in variables (`final_sr`, `final_cr`, etc.) and reuse them after the loop, saving 3 calls to `sincos256_ps` per block of 8 agents per step.

### 2. Redundant Terrain Height Calculation in `physics_avx.hpp`
**Issue**: `terrain_height` (involving `sin` and `cos`) is computed inside the substep loop to check for collisions. It is computed *again* after the loop to determine the final collision flag for rewards.
**Optimization**: The `terr_z` value computed in the last substep corresponds to the final position `px, py`. We can reuse this value, saving 1 call to `sin` and `cos` per block of 8 agents per step.

### 3. Scalar Fallback Optimization in `drone_cython.pyx`
**Issue**: The scalar fallback `_step_agent_scalar` performs similar redundant calculations (terrain height and trigonometry after the loop).
**Optimization**: Apply the same caching strategy to the scalar implementation.

## Plan
1. Modify `drone_env/physics_avx.hpp` to cache `sr, cr, sp, cp, sy, cy` and `terr_z` from the last loop iteration.
2. Modify `drone_env/drone_cython.pyx` to apply similar optimizations to `_step_agent_scalar`.
3. Verify correctness and measure speedup using `benchmark_cython.py`.

## Top 5 Future Improvements

Based on the analysis of `cachegrind` reports and code structure, here are the ranked improvements for future iterations:

1.  **Reduce Memory Traffic in Observation Shifting (`memmove`)**
    *   **Impact**: High. The observation buffer shift (1740 floats per agent per step) generates significant memory traffic.
    *   **Proposal**: Implement a circular buffer for IMU history within the observation array or use pointers instead of physical data movement. This would require changing the `DroneEnv` observation interface but would eliminate the most expensive memory operation.

2.  **Optimize `terrain_height` Calculation**
    *   **Impact**: Medium-High. `terrain_height` is called 10 times per step per agent and involves expensive `sin`/`cos` computations.
    *   **Proposal**: Replace the `sin`/`cos` based terrain with a pre-computed lookup table (heightmap) or a simpler polynomial approximation if the exact analytical shape isn't strictly required. Vectorizing the constants further could also help.

3.  **Instruction Level Parallelism (ILP) via Unrolling**
    *   **Impact**: Medium. The dependency chain in the integration loop limits CPU pipeline utilization.
    *   **Proposal**: Unroll the main agent loop inside `step_agents_avx2` to process two blocks of 8 agents (16 total) interleaved. This allows the CPU to execute independent instructions from the second block while waiting for latencies (e.g., FMA, memory) in the first block.

4.  **Optimize `exp` calls in Reward Function**
    *   **Impact**: Medium. `exp256_ps` is computationally expensive and is called twice per agent per step.
    *   **Proposal**: Use a faster approximation for `exp` (e.g., Padé approximant) or implement an early exit strategy where `exp` is approximated to 0 if the input (error squared) is large enough.

5.  **Data Alignment Enforcement**
    *   **Impact**: Low-Medium. Currently, `_mm256_loadu_ps` (unaligned load) is used.
    *   **Proposal**: Ensure that all NumPy arrays allocated in Python are aligned to 32-byte boundaries. This allows replacing `_mm256_loadu_ps` with `_mm256_load_ps`, which can be slightly faster on some microarchitectures and avoids potential penalties.

## Validation Strategy

A naive Python implementation (`tests/test_physics_validation.py`) was created to verify the numerical correctness of the optimized Cython/AVX backend. This script compares the state evolution (position, velocity) of the optimized kernel against a readable, standard NumPy implementation. The validation confirms that the optimizations (trigonometry caching, loop restructuring) maintain parity with the reference logic.
