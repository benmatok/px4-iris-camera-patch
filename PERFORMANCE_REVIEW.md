# Blind Dive Performance Review

## Overview
We executed 10 diverse blind dive scenarios with the following parameter ranges:
- Mass: 2.0 - 6.0 kg
- Altitude: 40 - 150 m
- Lateral Distance: 20 - 150 m
- Drag Coefficient: 0.05 - 0.5
- Wind: -5 to +5 m/s

## Results Summary
The Ghost-DPC controller showed mixed performance, with a high failure rate in scenarios involving high mass (> 2.5 kg).

- **Successes**: Scenario 1 (Mass 2.23kg) achieved excellent tracking (Dist 0.89m).
- **Failures**: Most other scenarios (Mass > 3.0kg) failed to close the distance or crashed.

## Analysis
The primary failure mode appears to be **Mass Estimation Saturation**.
- The `PyGhostEstimator` uses a lattice of models with mass range [0.5, 1.5] kg.
- The adaptive mass update is clamped at 5.0 kg, but the learning rate is low (0.001).
- When true mass is high (e.g., 4.0 kg), the estimator likely initializes at 1.0 or 1.5 kg.
- The controller, believing the drone is light, commands aggressive maneuvers (steep dives) that the heavy drone cannot recover from given the thrust limits, leading to crashes or overshoots.
- The "Distance Ghosts" (drag variations) may also contribute to hesitation or incorrect trajectory planning when the dynamic model is mismatched.

## Recommendations
1. **Expand Estimator Lattice**: Include heavier models (e.g., 3.0kg, 5.0kg) in the initial uncertainty lattice to improve convergence speed.
2. **Tune Adaptation**: Increase the learning rate for mass estimation or relax the clamping limits if safe.
3. **Conservative Safety**: Update the controller to be more conservative when observability of mass is low, preventing aggressive dives until mass estimate converges.
