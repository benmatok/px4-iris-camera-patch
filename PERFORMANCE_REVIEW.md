# Blind Dive Performance Review

## Overview
We executed 10 diverse blind dive scenarios with the following parameter ranges:
- Mass: 2.0 - 6.0 kg
- Altitude: 40 - 150 m
- Lateral Distance: 20 - 150 m
- Drag Coefficient: 0.05 - 0.5
- Wind: -5 to +5 m/s

## Results Summary (Run v6)
The Ghost-DPC controller performance has been tuned to handle heavier drones by correcting the thrust capabilities of the internal ghost models.

- **Light Drones (2.0 - 3.5 kg)**: Generally good performance.
    - Scenario 1 (2.23 kg): **9.86m** error.
    - Scenario 6 (3.22 kg): **29.20m** error.
- **Heavy Drones (3.5 - 6.0 kg)**: Variable performance.
    - Scenario 8 (4.19 kg): Improved to **105m** (was 174m in v5), but still high.
    - Scenario 5 (4.43 kg): Remains challenging (**165m**).
    - Scenario 2 (5.33 kg): Consistent around **57m**.

## Improvements Implemented
1.  **Mass Estimation Tuning**:
    - Increased clamping limit to **8.0 kg**.
    - Tuned learning rate to **0.003**.
    - Gated updates by Observability Score to prevent divergence.

2.  **Structural Changes**:
    - Added intermediate ghost models at **0.5 kg increments** (2.0, 2.5, ..., 6.0 kg).
    - **Thrust Correction**: Updated the thrust coefficients of heavy ghost models to scale with mass (approx `mass * 1.0`), ensuring the solver uses physically valid models capable of hovering.
    - Relaxed **TTC Safety Barrier** gain (2.0) and increased **Descent Velocity Limit** (15.0 m/s).

## Analysis
The correction of thrust coefficients ensures that the solver doesn't discard heavy drone hypotheses as "unflyable". However, the high residual errors in some scenarios suggest that:
1.  **Wind/Drag Estimation**: The interaction between high mass and wind is complex. The estimator might still be converging too slowly or getting stuck in local minima.
2.  **Solver Horizon**: Heavy drones have high inertia. The 40-step horizon (2 seconds) might be too short to plan a recovery from a steep dive, leading to conservative behavior (staying high).

## Recommendations
- **Horizon Extension**: Increase solver horizon for heavy drones if computational budget permits.
- **Wind-Specific Heavy Models**: Add heavy models with explicit wind hypotheses to the lattice.
