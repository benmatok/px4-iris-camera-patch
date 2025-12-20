# Performance Optimization Report

## Baseline Profiling (Before Optimization)

Profiling was performed using Valgrind (Cachegrind) on `train_ae.py` with 100 agents for 1 episode.

*   **Total Instructions (Ir):** ~13,827,076,437
*   **Observations:** The physics loop involves heavy trigonometric computations (`sin`, `cos`) for updating agent orientation (roll, pitch, yaw) at every substep.

## Optimization: Combined Sine/Cosine (sincos)

The core optimization targeted the redundant trigonometric function calls in the physics engine.

### Changes Implemented:

1.  **AVX Backend (`drone_env/physics_avx.hpp` & `drone_env/avx_mathfun.h`):**
    *   Implemented `sincos256_ps` in the AVX math library.
    *   This function computes sine and cosine for an AVX vector simultaneously.
    *   Since `sin(x)` and `cos(x)` share significant computational steps (range reduction, polynomial evaluation structures), computing them together is much faster than calling them separately.
    *   Updated `step_agents_avx2` to use `sincos256_ps` for roll, pitch, and yaw updates.

2.  **Scalar/Cython Backend (`drone_env/drone_cython.pyx`):**
    *   Imported the `sincosf` function from the C standard library (via `math.h` extern).
    *   Replaced separate `sin()` and `cos()` calls with `sincosf()` in the scalar fallback loop.

### Expected Impact:

*   **Reduction in Trig Calls:** For each agent, in each substep, we compute `sin/cos` for 3 angles. Previously this required 6 calls. Now it requires 3 `sincos` calls.
*   **Instruction Count:** This should lead to a measurable reduction in total instruction count for the physics step, as trigonometric evaluation is computationally expensive (polynomial approximation).

## Optimization: Texture Engine Block Processing (2D Tiling) & SIMD

To enable real-time analysis of high-resolution texture features, we implemented a **2D Tiling + Explicit SIMD** strategy for the Texture Engine (`drone_env/texture_features.pyx`).

### Problem
The initial naive implementation computed gradients and features for the entire image at once, causing L2 cache thrashing for large images (e.g., 2048x2048). Furthermore, the scalar arithmetic operations for 3x3 window accumulations were instruction-heavy.

### Solution 1: 2D Tiling
Refactored the pipeline to process the image in **$32 \times 32$ Tiles**.
1.  **Thread-Local Buffers**: Each thread allocates a tiny scratchpad (~4KB) to hold gradient data for a single tile. This guarantees the working set fits entirely within the **L1 Cache**.
2.  **Flattened Parallelism**: We flatten the tile iteration space (`num_strips_r * num_strips_c`) and parallelize over all tiles using `prange` with dynamic scheduling, ensuring better load balancing than row-strip parallelization.

### Solution 2: Explicit AVX2 Intrinsics
We replaced the auto-vectorized loops with hand-tuned AVX2 intrinsics (`immintrin.h`) for the inner 3x3 convolution and feature computation:
1.  **Vectorized Accumulation**: Structure Tensor moments ($S_{xx}, S_{yy}, S_{xy}$) are accumulated using `_mm256_add_ps` and `_mm256_mul_ps`, processing 8 pixels simultaneously.
2.  **Vectorized Eigenvalues**: Coherence computation uses `_mm256_sqrt_ps`.
3.  **Hessian Determinant**: Computed fully in SIMD registers.
4.  **Fallback**: Orientation (`atan2`) remains a bottleneck handled by a scalar fallback loop over the vector elements, as AVX `atan2` is complex to implement.

### Results (2048x2048 Image)
*   **Baseline (Full-Frame Scalar):** ~282ms per image.
*   **Optimized (2D Tiling + AVX2):** ~244ms per image.
*   **Speedup:** **~1.15x**.
*   **Memory Efficiency:** Drastic reduction in thread-local memory allocation (KB vs MB).

### Bottleneck Analysis (512x512 Image)
Profiling with `cachegrind` reveals the remaining hotspots in the optimized engine:

*   **Main Loop (SIMD Logic + Overhead):** **~57%**. This includes the vectorized feature computation, GED scalar loop, and loop control.
*   **Orientation Calculation (`atan2`):** **~13%**. The scalar fallback for `atan2` (used for Structure Tensor orientation) is the most significant mathematical bottleneck.
*   **Gaussian Blur:** **~8.6%**. The separable convolution steps (`convolve_horizontal` + `convolve_vertical`) take a moderate chunk.
*   **GED Aggregation (`cbrtf`):** **~4.4%**. The cube root calculation for boundary fusion.

**Conclusion:** The implementation is now **Compute Bound**. Memory access (L1/L2 misses) is no longer the primary constraint. Further optimization would require an approximate SIMD `atan2` implementation or algorithmic simplification.

## Validation

*   **Correctness:** Validated via `test_texture_engine.py` (Passes "Perfect Sine", "Spinning Plate", "Zooming Dot").
*   **Performance:** Confirmed via `benchmark_large.py` running 20 iterations on 2048x2048 random input.
