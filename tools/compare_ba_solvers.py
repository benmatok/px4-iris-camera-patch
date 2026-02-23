import numpy as np
import time
import logging
from vision.sliding_window_estimator import SlidingWindowEstimator
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CompareBA")

def run_comparison():
    print("--- Comparing Python vs C++ BA Solvers ---")

    # 1. Setup Solvers
    est_py = SlidingWindowEstimator(window_size=5, use_cpp=False)
    est_cpp = SlidingWindowEstimator(window_size=5, use_cpp=True)

    # 2. Generate Synthetic Data (Straight Line Flight)
    # 10 frames, 0.1s dt
    # v = [5, 0, 0]
    frames = []
    p_true = np.array([0.0, 0.0, -10.0])
    v_true = np.array([5.0, 0.0, 0.0])
    q_true = R.from_euler('xyz', [0, 0, 0]).as_quat()

    points = {
        1: [20.0, 5.0, 5.0],
        2: [25.0, -5.0, 10.0],
        3: [30.0, 0.0, 0.0],
        4: [15.0, 2.0, -5.0]
    }

    # Simple accel (gravity cancellation handled inside?)
    # If hovering, a_w=0, a_body = R.inv * (0 - g).
    # Here v=const, a_w=0.
    # a_b = [0, 0, 9.81] (upward force to counteract gravity)
    imu_data = [(0.1, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]))]

    print(f"{'Frame':<5} {'Py Time':<10} {'Cpp Time':<10} {'Py V-Err':<10} {'Cpp V-Err':<10}")

    for i in range(10):
        t = i * 0.1

        # Ground Truth update
        p_true += v_true * 0.1

        # Observations
        obs = {}
        # Simple PinHole Project
        # P_c = P_w - P_cam (Identity rotation)
        for pid, pt in points.items():
            pc = np.array(pt) - p_true
            if pc[0] > 0.1: # In front (X-forward)
                # u = y/x, v = z/x (Using standard robotic frame X-forward? No, code uses Z-forward convention u=x/z)
                # Code uses: P_body = frame['R'].inv().apply(pt_pos - frame['p'])
                # If R=Identity, P_body = pt - p.
                # Code: u = x/z, v = y/z.
                # So Z must be forward.
                # Our synthetic motion is X-forward (v=[5,0,0]).
                # Let's rotate camera: R_wc = [0,0,1, -1,0,0, 0,-1,0] (ypr?)
                # Simplification: Assume camera Z is forward.
                # Motion v=[0,0,5] (Z-forward)

                # Let's adjust motion to Z-forward
                v_true = np.array([0.0, 0.0, 5.0])

                # Re-calc pc
                # pc = pt - p_true
                # u = pc[0]/pc[2], v = pc[1]/pc[2]

                if pc[2] > 0.1:
                    u = pc[0]/pc[2]
                    v = pc[1]/pc[2]
                    obs[pid] = (u, v)

        # Add Frame Py
        t0 = time.time()
        est_py.add_frame(t, None, None, None, imu_data, obs, vel_prior=None, baro=-p_true[2])
        est_py.solve()
        t_py = (time.time() - t0) * 1000

        # Add Frame Cpp
        t0 = time.time()
        est_cpp.add_frame(t, None, None, None, imu_data, obs, vel_prior=None, baro=-p_true[2])
        est_cpp.solve()
        t_cpp = (time.time() - t0) * 1000

        # Compare V
        state_py = est_py.get_latest_state()
        state_cpp = est_cpp.get_latest_state()

        v_py = np.array([state_py['vx'], state_py['vy'], state_py['vz']])
        v_cpp = np.array([state_cpp['vx'], state_cpp['vy'], state_cpp['vz']])

        err_py = np.linalg.norm(v_py - v_true)
        err_cpp = np.linalg.norm(v_cpp - v_true)

        print(f"{i:<5} {t_py:<10.2f} {t_cpp:<10.2f} {err_py:<10.4f} {err_cpp:<10.4f}")

if __name__ == "__main__":
    run_comparison()
