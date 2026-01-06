# GradientController Documentation

This document outlines the core logic and operational principles of the `GradientController`, which refines drone trajectories using a Gradient-based Predictive Control approach.

## 1. Loss Construction

The optimization objective minimizes the weighted sum of squared residuals between the simulated trajectory (forward model) and the Oracle's reference trajectory over the planning horizon.

**Residuals Vector ($r$):**
The residuals are computed as a concatenation of position and attitude errors:
$$ r = [ r_{pos}, r_{att} \cdot w_{att} ] $$

Where:
*   **$r_{pos}$**: The Euclidean difference in Position ($X, Y, Z$) at each time step between the simulated drone state and the Oracle's planned state.
*   **$r_{att}$**: The difference in Attitude (Roll, Pitch, Yaw) at each time step.
    *   *Note:* Yaw differences are explicitly wrapped to the range $[-\pi, \pi]$ to correctly handle angular discontinuities.
*   **$w_{att}$**: A weighting factor (currently set to **5.0**) applied to attitude residuals to prioritize accurate orientation and stability over pure position tracking.

**Optimization Goal:**
The controller solves for parameter updates ($\Delta$) to minimize the squared norm of the residuals ($||r||^2$) using the Levenberg-Marquardt algorithm.

## 2. Exploration Method

The controller uses **Finite Difference Gradient Estimation** to explore the local control landscape and estimate the Jacobian matrix required for optimization.

*   **Parallel Simulations:** The system maintains simulation slots for $(1 + N_{params})$ agents, where $N_{params}$ is the number of optimization parameters (Chebyshev coefficients).
*   **Perturbation Scheme:**
    *   **Slot 0 (Base):** Simulates the trajectory using the current best estimate of the parameters.
    *   **Slots 1..N:** Simulate trajectories where the $j$-th parameter is perturbed by a small scalar $\epsilon$ (epsilon, currently **0.01**).
*   **Jacobian Estimation:**
    The Jacobian matrix $J$, representing the sensitivity of the state outputs to control parameters, is approximated as:
    $$ J_{ij} \approx \frac{\text{State}_{perturbed}^{(j)} - \text{State}_{base}}{\epsilon} $$

## 3. Control -> World Response Propagation

The `GradientController` does not use a simplified analytical model for optimization; instead, it utilizes the actual physics engine (`DroneEnv`) as the forward model. This ensures that the planned trajectories are physically feasible and account for complex dynamics like drag and thrust limits.

**Propagation Steps:**
1.  **Input:** A set of **Chebyshev Coefficients** (degree 4) that define the control actions (Thrust, Roll Rate, Pitch Rate, Yaw Rate) over the time horizon.
2.  **Polynomial Evaluation:** The coefficients are evaluated to generate raw control inputs for every time step in the horizon.
3.  **Clamping:** The control inputs are clamped to feasible hardware limits:
    *   Thrust: $[0.0, 1.0]$ (Normalized)
    *   Angular Rates: $[-10.0, 10.0]$ rad/s
4.  **Simulation:** The internal `sim_env` (a dedicated `DroneEnv` instance) is stepped forward for `horizon_steps` (typically 10 steps, representing 0.5 seconds).
5.  **Output:** The resulting sequence of World States (Position, Attitude) is captured and used to compute the residuals against the target.

## 4. Current Validation Scheme

The controller implements several safety checks and fallback mechanisms to ensure robust operation:

*   **Scanning Override:**
    If the target confidence (from the tracker) drops below **0.1**, the gradient optimization is bypassed. The controller reverts to a hardcoded "Scanning Mode":
    *   **Yaw Rate:** sinusoidal pattern $\sin(t) + t \cos(t)$.
    *   **Thrust:** Maintenance hover thrust.
*   **Solver Safety:**
    The Linear System Solver (used to compute the Levenberg-Marquardt update step) is wrapped in a `try-except` block. If the matrix $(J^T J + \lambda I)$ is ill-conditioned or singular, the update for that iteration is skipped to prevent numerical instability.
*   **Implicit Trust Region:**
    Currently, the algorithm applies the computed update directly without a line-search or explicit cost-reduction check. Stability is managed primarily through the Levenberg-Marquardt damping factor ($\lambda = 0.1$).
