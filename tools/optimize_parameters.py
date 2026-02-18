import sys
import os
import logging
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
import warnings
import io
import contextlib

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.validate_dive_tracking import DiveValidator
from flight_config import FlightConfig

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure Logging
logging.getLogger("ScenarioValidator").setLevel(logging.WARNING)
logging.getLogger("tests.validate_dive_tracking").setLevel(logging.WARNING)
logging.getLogger("sim_interface").setLevel(logging.WARNING)
logging.getLogger("vision.projection").setLevel(logging.WARNING)
logging.getLogger("flight_controller").setLevel(logging.WARNING)
logging.getLogger("mission_manager").setLevel(logging.WARNING)


# Parameters to optimize and their bounds
PARAM_SPACE = {
    'k_pitch': (1.0, 8.0),
    'k_yaw': (1.0, 5.0),
    'dive_trigger_rer': (0.05, 0.3),
    'dive_trigger_v_threshold': (0.2, 0.5),
    'cruise_pitch_gain': (0.1, 0.5),
    'k_rer': (1.0, 4.0),
    'flare_gain': (1.0, 4.0),
    'thrust_base_intercept': (0.4, 0.8),
    'thrust_base_slope': (0.1, 0.5)
}

PARAM_NAMES = list(PARAM_SPACE.keys())
BOUNDS = np.array([PARAM_SPACE[name] for name in PARAM_NAMES])

SCENARIOS = [
    {"id": 1, "alt": 100.0, "dist": 50.0},
    {"id": 2, "alt": 50.0,  "dist": 75.0},
    {"id": 3, "alt": 20.0,  "dist": 50.0},
    {"id": 4, "alt": 60.0,  "dist": 80.0},
    {"id": 5, "alt": 50.0,  "dist": 50.0},
    {"id": 6, "alt": 25.0,  "dist": 150.0}
]

def evaluate_params(params_vector):
    """
    Evaluates a parameter set.
    params_vector: numpy array of shape (N,)
    Returns: scalar loss
    """
    # Create Config
    config = FlightConfig()

    for i, name in enumerate(PARAM_NAMES):
        if hasattr(config.control, name):
            setattr(config.control, name, float(params_vector[i]))

    total_loss = 0.0

    # Suppress stdout/stderr
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        for sc in SCENARIOS:
            validator = DiveValidator(
                use_ground_truth=True,
                use_blind_mode=True, # Validate against strict blind mode constraint
                init_alt=sc['alt'],
                init_dist=sc['dist'],
                config=config
            )

            # Run simulation
            try:
                hist = validator.run(duration=25.0)
                min_dist = min(hist['dist'])
            except Exception:
                min_dist = 1000.0 # High penalty for crash/error

            # Loss: Sum of sqrt(min_dist)
            total_loss += np.sqrt(min_dist)

    return total_loss

def gp_search(n_initial=10, n_iter=20):
    print(f"Starting GP Search with {n_initial} random samples and {n_iter} iterations...")

    # Random Initialization
    X = []
    y = []

    for _ in range(n_initial):
        x = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
        loss = evaluate_params(x)
        X.append(x)
        y.append(loss)
        print(f"  Init Loss: {loss:.4f}")

    X = np.array(X)
    y = np.array(y)

    # GP Kernel
    kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

    for i in range(n_iter):
        gp.fit(X, y)

        # Acquisition Function: Expected Improvement (EI)
        # We want to MINIMIZE y, so standard EI for maximization of -y
        # Or EI for minimization: EI(x) = (y_min - mu(x) - xi) * Phi(Z) + sigma(x) * phi(Z)

        def acquisition(x_candidates):
            mu, sigma = gp.predict(x_candidates, return_std=True)
            y_min = np.min(y)
            xi = 0.01

            with np.errstate(divide='warn'):
                imp = y_min - mu - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0

            return -ei # Minimize negative EI -> Maximize EI

        # Optimize Acquisition Function
        # Random search for acquisition max (simple)
        candidates = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(1000, len(PARAM_NAMES)))
        acq_values = acquisition(candidates)
        best_idx = np.argmin(acq_values)
        next_x = candidates[best_idx]

        # Evaluate
        next_y = evaluate_params(next_x)

        X = np.vstack([X, next_x])
        y = np.append(y, next_y)

        print(f"  Iter {i+1}/{n_iter}: Best Loss So Far: {np.min(y):.4f} (Current: {next_y:.4f})")

    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx]

def local_sgd(start_params, n_steps=10, lr=0.01):
    print(f"\nStarting Local SGD from Loss: {evaluate_params(start_params):.4f}")

    current_params = start_params.copy()

    for step in range(n_steps):
        # Finite Difference Gradient
        grad = np.zeros_like(current_params)
        epsilon = 1e-4

        base_loss = evaluate_params(current_params)

        for i in range(len(current_params)):
            temp_params = current_params.copy()
            temp_params[i] += epsilon
            loss_plus = evaluate_params(temp_params)

            grad[i] = (loss_plus - base_loss) / epsilon

        # Update
        # Normalize gradient to avoid huge steps?
        # Or use adaptive LR?
        # Let's try simple update with decay

        step_lr = lr / (1 + step * 0.1)

        # Clamp update to bounds
        next_params = current_params - step_lr * grad
        next_params = np.clip(next_params, BOUNDS[:, 0], BOUNDS[:, 1])

        new_loss = evaluate_params(next_params)

        print(f"  SGD Step {step+1}: Loss {base_loss:.4f} -> {new_loss:.4f}")

        if new_loss < base_loss:
            current_params = next_params
        else:
            # Backtrack / Reduce LR
            print("    Loss increased, reducing step size.")
            # Simple backtrack: Don't update
            pass

    return current_params, evaluate_params(current_params)

if __name__ == "__main__":
    best_gp_params, best_gp_loss = gp_search(n_initial=5, n_iter=10)
    print(f"\nBest GP Params Loss: {best_gp_loss:.4f}")

    final_params, final_loss = local_sgd(best_gp_params, n_steps=5, lr=0.05)

    print("\n" + "="*40)
    print("OPTIMIZATION COMPLETE")
    print(f"Final Loss: {final_loss:.4f}")
    print("Best Parameters:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name}: {final_params[i]:.4f}")
    print("="*40)
