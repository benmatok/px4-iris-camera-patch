import time
import numpy as np
import sys
import os

# Ensure we can import ghost_dpc
import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ghost_dpc.ghost_dpc import PyDPCSolver, PyGhostModel

def benchmark_solver():
    print("Benchmarking DPC Solver Latency...")

    solver = PyDPCSolver()

    # Model configuration
    model_dict = {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}
    models = [model_dict]
    weights = [1.0]

    # Initial State
    state = {
        'px': 0.0, 'py': 0.0, 'pz': 10.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }

    # Target
    target = [10.0, 10.0, 10.0]

    # Initial Action Guess
    init_action = {
        'thrust': 0.5,
        'roll_rate': 0.0,
        'pitch_rate': 0.0,
        'yaw_rate': 0.0
    }
    dt = 0.05

    # Warmup
    for _ in range(10):
        solver.solve(state, target, init_action, models, weights, dt)

    # Benchmark Loop
    num_iters = 1000
    start_time = time.perf_counter()

    for _ in range(num_iters):
        solver.solve(state, target, init_action, models, weights, dt)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iters) * 1000.0

    print(f"Total time for {num_iters} iterations: {total_time:.4f}s")
    print(f"Average time per solve: {avg_time_ms:.4f}ms")

    if avg_time_ms > 1.5:
        print(f"FAIL: Average time {avg_time_ms:.4f}ms > 1.5ms limit")
        sys.exit(1)
    else:
        print("PASS: Solver latency within limits.")

if __name__ == "__main__":
    benchmark_solver()
