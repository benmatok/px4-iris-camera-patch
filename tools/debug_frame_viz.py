
import sys
import os
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_interface import SimDroneInterface
from flight_controller import DPCFlightController
from vision.msckf import MSCKF
from flight_config import FlightConfig
from vision.projection import Projector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugFrame")

def main():
    print("--- Debugging Frame Alignment ---")
    config = FlightConfig()
    proj = Projector(640, 480, 120.0, 30.0)
    sim = SimDroneInterface(proj, config=config)

    # 1. Initialize at 0, 0, 100. Hover.
    sim.reset_to_scenario("Blind Dive", pos_x=0.0, pos_y=0.0, pos_z=100.0, pitch=0.0, yaw=0.0)

    # 2. Init VIO
    msckf = MSCKF(proj)
    s = sim.get_state()
    # Init with Truth
    # NED Quat.
    # Sim Yaw 0 -> East.
    # NED Yaw: 90 (East).
    # Sim Roll 0, Pitch 0.
    r_ned = 0.0
    p_ned = 0.0
    y_ned = np.radians(90.0)
    q_init = R.from_euler('xyz', [r_ned, p_ned, y_ned], degrees=False).as_quat()
    p_init = np.array([0.0, 0.0, -100.0]) # NED Pos (N, E, D). Sim (0,0,100) -> NED (0, 0, -100)
    v_init = np.array([0.0, 0.0, 0.0])
    msckf.initialize(q_init, p_init, v_init)

    # 3. Step Sim Forward (East)
    # Apply Pitch Down (-Pitch Sim) -> Forward Force (East).
    # Sim X is East.
    print("\n--- Applying Pitch Down (Sim -Pitch) to move East ---")

    # Step 10 times
    for i in range(10):
        # Action: Thrust 0.5, Pitch Rate -1.0 (Nose Down)
        # Note: I verified in check_frame_alignment that +Rate -> -Pitch.
        # So -1.0 Rate -> +Pitch (Nose Up).
        # Wait. check_frame_alignment:
        # Pitch Rate -1.0 -> Pitch 0.0083 (Positive).
        # Sim Pitch Positive = Nose Up.
        # Nose Up -> Backward (West).
        # We want East.
        # We need Nose Down (Negative Pitch).
        # So we need Positive Rate?
        # check_frame_alignment: Pitch Rate +1.0 -> Pitch -0.0083 (Negative).
        # So +Rate -> Nose Down.

        action = [0.5, 0.0, 1.0, 0.0] # Pitch Rate +1.0

        sim.step(action)
        s = sim.get_state()

        # Propagate VIO
        # Gyro/Accel Transform (Sim -> NED)
        # Gyro: [wx, -wy, -wz]
        # Accel: [ax, -ay, -az]
        gyro_ned = np.array([s['wx'], -s['wy'], -s['wz']])
        accel_ned = np.array([s.get('ax_b', 0.0), -s.get('ay_b', 0.0), -s.get('az_b', 9.81)])

        msckf.propagate(gyro_ned, accel_ned, 0.05)

    print(f"\nSim State (ENU/Sim Frame):")
    print(f"  Pos: ({s['px']:.4f}, {s['py']:.4f}, {s['pz']:.4f})  [X=East, Y=North]")
    print(f"  Vel: ({s['vx']:.4f}, {s['vy']:.4f}, {s['vz']:.4f})  [X=East, Y=North]")

    print(f"\nVIO State (NED Frame):")
    vio_vel = msckf.get_velocity()
    print(f"  Vel: ({vio_vel[0]:.4f}, {vio_vel[1]:.4f}, {vio_vel[2]:.4f}) [0=North, 1=East]")

    # Controller Interpretation
    # flight_controller.py logic:
    # vx_enu = velocity_est['vy'] (East)
    # vy_enu = velocity_est['vx'] (North)

    ctrl_vx = vio_vel[1]
    ctrl_vy = vio_vel[0]

    print(f"\nController Input (ENU):")
    print(f"  Est VX: {ctrl_vx:.4f} (Should match Sim VX)")
    print(f"  Est VY: {ctrl_vy:.4f} (Should match Sim VY)")

    if abs(ctrl_vx - s['vx']) > 0.5 or abs(ctrl_vy - s['vy']) > 0.5:
        print("\n[FAIL] Significant Mismatch!")
    else:
        print("\n[PASS] Velocity Frames Aligned.")

if __name__ == "__main__":
    main()
