
import sys
import os
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_interface import SimDroneInterface
from flight_controller_gdpc import NumPyGhostModel
from flight_config import FlightConfig
from vision.projection import Projector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckAlignment")

def run_step_test(name, action, desc):
    print(f"\n--- Test: {name} ({desc}) ---")
    config = FlightConfig()
    proj = Projector(640, 480, 120.0, 30.0)
    sim = SimDroneInterface(proj, config=config)

    # Init at Hover state
    sim.reset_to_scenario("Blind Dive", pos_x=0.0, pos_y=0.0, pos_z=100.0, pitch=0.0, yaw=0.0)

    # Step Sim
    sim.step(action)
    s = sim.get_state()

    print(f"Sim State After 1 Step:")
    print(f"  Pos: ({s['px']:.4f}, {s['py']:.4f}, {s['pz']:.4f})")
    print(f"  Vel: ({s['vx']:.4f}, {s['vy']:.4f}, {s['vz']:.4f})")
    print(f"  RPY: ({s['roll']:.4f}, {s['pitch']:.4f}, {s['yaw']:.4f})")

    # Run GDPC Model
    phy = config.physics
    model = NumPyGhostModel(
        mass=phy.mass,
        drag_coeff=phy.drag_coeff,
        thrust_coeff=phy.thrust_coeff,
        tau=phy.tau,
        g=phy.g,
        max_thrust_base=phy.max_thrust_base
    )

    # Init Model State (Same as Sim Start)
    s0 = {
        'px': 0.0, 'py': 0.0, 'pz': 100.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'wx': 0.0, 'wy': 0.0, 'wz': 0.0
    }

    # Rollout 1 step
    u_seq = np.array([action])
    traj = model.rollout(s0, u_seq, dt=0.05)
    p = traj[0] # px, py, pz, vx, vy, vz, r, p, y, wx, wy, wz

    print(f"Model State After 1 Step:")
    print(f"  Pos: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
    print(f"  Vel: ({p[3]:.4f}, {p[4]:.4f}, {p[5]:.4f})")
    print(f"  RPY: ({p[6]:.4f}, {p[7]:.4f}, {p[8]:.4f})")

    # Compare
    sim_vel = np.array([s['vx'], s['vy'], s['vz']])
    mod_vel = p[3:6]

    if np.allclose(sim_vel, mod_vel, atol=1e-6):
        print("RESULT: MATCH")
    else:
        print("RESULT: MISMATCH!")
        print(f"Diff: {sim_vel - mod_vel}")

def main():
    # 1. Pitch Rate +1.0 (Nose Up?)
    # In Sim (ENU), Pitch + is Nose Up.
    # Thrust vector tilts back (-X).
    # Velocity X should decrease (become negative).
    run_step_test("Pitch +1", [0.5, 0.0, 1.0, 0.0], "Pitch Rate +1.0 (Nose Up)")

    # 2. Pitch Rate -1.0 (Nose Down?)
    # Velocity X should increase (positive).
    run_step_test("Pitch -1", [0.5, 0.0, -1.0, 0.0], "Pitch Rate -1.0 (Nose Down)")

    # 3. Roll Rate +1.0 (Right Wing Down?)
    # In Sim (ENU), Roll + is Right Wing Down?
    # Thrust vector tilts Right (-Y? or +Y?).
    # Body Y is Left. Roll is about X.
    # Z Up. Y Left. X Fwd.
    # Roll + about X -> Y moves towards Z. Z moves towards -Y.
    # Thrust (Z) tilts to -Y (Right).
    # So Velocity Y should decrease (become negative).
    run_step_test("Roll +1", [0.5, 1.0, 0.0, 0.0], "Roll Rate +1.0")

    # 4. Yaw Rate +1.0 (Turn Left?)
    # Z Up. Yaw + about Z.
    # X moves to Y. Y moves to -X.
    # Turn Left (CCW).
    run_step_test("Yaw +1", [0.5, 0.0, 0.0, 1.0], "Yaw Rate +1.0")

if __name__ == "__main__":
    main()
