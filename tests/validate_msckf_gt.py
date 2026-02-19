
import sys
import os
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_interface import SimDroneInterface
from vision.msckf import MSCKF
from flight_config import FlightConfig
from vision.projection import Projector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ValidateMSCKF_GT")

def main():
    print("--- Validating MSCKF with Ground Truth Velocity Update ---")
    config = FlightConfig()
    proj = Projector(640, 480, 120.0, 30.0)
    sim = SimDroneInterface(proj, config=config)

    # Init Scenario 4 (Dynamic)
    sim.reset_to_scenario("Blind Dive", pos_x=0.0, pos_y=0.0, pos_z=60.0, pitch=np.radians(-60.0), yaw=0.0)

    # Init MSCKF
    msckf = MSCKF(proj)
    s = sim.get_state()
    # True NED Init
    q_init = R.from_euler('xyz', [s['roll'], s['pitch'], (np.pi/2)-s['yaw']], degrees=False).as_quat()
    p_init = np.array([s['py'], s['px'], -s['pz']]) # NED Pos
    v_init = np.array([s['vy'], s['vx'], -s['vz']]) # NED Vel? No, vx/vy are Sim Frame.
    # Convert Sim Vel to NED Vel
    # Sim [vx, vy, vz]. NED [N, E, D].
    # N = vy_sim. E = vx_sim. D = -vz_sim.
    v_ned_init = np.array([s['vy'], s['vx'], -s['vz']])

    msckf.initialize(q_init, p_init, v_ned_init)

    steps = 100 # 5s
    errors = []

    for i in range(steps):
        # Step Sim with constant action
        sim.step([0.5, 0.0, 0.0, 0.0])
        s = sim.get_state()

        # Propagate (with correct frame)
        gyro_ned = np.array([s['wx'], -s['wy'], -s['wz']])
        accel_ned = np.array([s.get('ax_b', 0.0), -s.get('ay_b', 0.0), -s.get('az_b', 9.81)])
        msckf.propagate(gyro_ned, accel_ned, 0.05)

        # GT Update (Pos Z and Vel NED)
        height_meas = -s['pz'] # NED pz
        v_ned = np.array([s['vy'], s['vx'], -s['vz']])

        # Update with GT
        msckf.update_velocity_vector(v_ned)
        msckf.update_height(height_meas)

        # Check Error
        est_p = msckf.p
        gt_p = np.array([s['py'], s['px'], -s['pz']])
        err = np.linalg.norm(est_p - gt_p)
        errors.append(err)

        if i % 20 == 0:
            print(f"Step {i}: Pos Error = {err:.4f} m")

    print(f"Mean Pos Error: {np.mean(errors):.4f} m")

if __name__ == "__main__":
    main()
