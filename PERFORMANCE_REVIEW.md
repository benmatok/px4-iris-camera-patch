# Blind Dive Performance Review

## Overview
We executed 10 diverse blind dive scenarios with the following parameter ranges:
- Mass: 2.0 - 6.0 kg
- Altitude: 40 - 150 m
- Lateral Distance: 20 - 150 m
- Drag Coefficient: 0.05 - 0.5
- Wind: -5 to +5 m/s

## Results Summary (Run v5)
The Ghost-DPC controller performance has improved significantly for light-to-medium mass drones (2.0 - 4.0 kg) after tuning.

- **Scenario 1 (2.23 kg)**: Distance error reduced from ~22m to **10.08m**.
- **Scenario 6 (3.22 kg)**: Distance error reduced from ~35m to **17.69m**.
- **Scenario 3 (3.73 kg)**: Distance error reduced from ~58m to **33.81m**.

However, heavy drones (> 4.0 kg) and scenarios with high wind/drag still present challenges:
- **Scenario 5 (4.43 kg)**: Still struggling (Dist 198m).
- **Scenario 4 (3.82 kg)**: Large error (111m), likely due to specific wind/drag combination.

## Improvements Implemented
1.  **Mass Estimation Tuning**:
    - Increased clamping limit to **8.0 kg**.
    - Tuned learning rate to **0.003**.
    - Gated updates by Observability Score to prevent divergence.

2.  **Structural Changes**:
    - Added intermediate ghost models at **0.5 kg increments** (2.0, 2.5, ..., 6.0 kg) to bridge the gap between lattice points.
    - Relaxed **TTC Safety Barrier** gain from 5.0 to 2.0.
    - Increased **Descent Velocity Limit** from 12.0 to 15.0 m/s.

## Recommendations
- **Further Tuning**: The interaction between high mass and wind estimation needs more investigation. The current "Uncertainty Lattice" might need wind-specific heavy models.
- **Trajectory Planning**: For heavy drones, the solver might need a longer horizon or a different cost function to commit to a dive earlier.
