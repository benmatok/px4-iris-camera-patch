import sys
import os
import numpy as np
import logging
from sim_interface import SimDroneInterface
from vision.projection import Projector
from vision.vio_system import VIOSystem
from flight_config import FlightConfig

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLinearFlight")

def run_linear_flight_test():
    print("--- Running Linear Flight VIO Test ---")

    # 1. Setup
    config = FlightConfig()
    cam = config.camera
    projector = Projector(width=cam.width, height=cam.height, fov_deg=cam.fov_deg, tilt_deg=cam.tilt_deg)

    # Sim
    sim = SimDroneInterface(projector, config=config)
    sim.reset_to_scenario("Blind Dive", pos_x=0.0, pos_y=0.0, pos_z=50.0, pitch=0.0, yaw=0.0)

    # VIO
    vio = VIOSystem(projector, config=config)

    # Initialize VIO
    s0 = sim.get_state()
    vio.initialize(
        q=np.array([0.0, 0.0, 0.0, 1.0]), # Should match sim init?
        p=np.array([s0['px'], s0['py'], -s0['pz']]), # NED
        v=np.array([s0['vx'], s0['vy'], -s0['vz']]) # NED
    )

    # Constant Control
    # Pitch forward (-0.2 rad), constant thrust
    action = np.array([0.55, 0.0, -0.2, 0.0]) # Thrust, Roll, Pitch, Yaw rates?
    # Wait, sim takes Rates. To hold pitch, we need to stabilize angle?
    # Sim is rates. We can just set pitch rate to 0.0, and init pitch to something.
    # But sim resets to specific pitch.
    # Let's actively control to hold pitch.

    # Target
    target_pitch = -0.3 # Nose down
    target_thrust = 0.58

    errors = []

    duration = 5.0
    dt = 0.05
    steps = int(duration / dt)

    for i in range(steps):
        s = sim.get_state()

        # Ground Truth NED
        gt_v_ned = np.array([s['vy'], s['vx'], -s['vz']]) # Sim is FLU?
        # Sim State: px, py, pz (ENU).
        # vx, vy, vz (ENU).
        # NED: x=North(y), y=East(x), z=Down(-z).
        # Wait, Sim x is East, y is North?
        # Standard ENU: X=East, Y=North, Z=Up.
        # Standard NED: X=North, Y=East, Z=Down.
        # So NED x = ENU y. NED y = ENU x. NED z = -ENU z.

        gt_v_ned = np.array([s['vy'], s['vx'], -s['vz']])

        # VIO Update
        gyro = np.array([s['wx'], -s['wy'], -s['wz']])
        accel = np.array([s.get('ax_b', 0.0), -s.get('ay_b', 0.0), -s.get('az_b', 9.81)])
        vio.propagate(gyro, accel, dt)

        # Image
        # Sim image generation needs a target. Let's put a target far away to generate features?
        # Or just random noise? The Sim interface generates feature tracks from ground truth?
        # SimDroneInterface.get_image returns a list of features?
        # No, get_image returns an image? Or points?
        # SimDroneInterface.get_image(target_pos) returns 'img' which is passed to tracker.
        # tracker.process() detects points.
        # We need a target to look at.
        target_pos = [100.0, 0.0, 0.0]
        img = sim.get_image(target_pos)

        # Tracker update
        # We need to call tracker.
        # VIO system has its own tracker?
        # VIOSystem.track_features(state_ned, body_rates, dt)
        # But we need raw image/features.
        # VIOSystem.update_measurements(height, vz, tracks)

        # We need to simulate the tracker or use the real one.
        # Real tracker needs image.
        # Sim.get_image returns what?
        # Let's check Sim.
        # Assuming Sim returns list of keypoints for simplicity if we want perfect tracking?
        # No, SimDroneInterface generates an image (numpy array).
        # VIOSystem.tracker (FeatureTracker) needs image.

        # Let's bypass the visual tracker for this pure VIO test and feed GT features?
        # Or use the full stack.
        # Full stack is better.

        # Convert Sim State to NED for tracker (it uses it for gyro compensation / masking?)
        dpc_state_ned = {
            'px': s['py'], 'py': s['px'], 'pz': -s['pz'],
            'vx': s['vy'], 'vy': s['vx'], 'vz': -s['vz'],
            'roll': s['roll'], 'pitch': s['pitch'], 'yaw': (np.pi/2) - s['yaw']
        }

        # We need `tracks` for `update_measurements`.
        # VIOSystem.tracker.track(img) -> tracks?
        # FeatureTracker.track(img, state_ned, ...)

        # This is getting complicated to mock.
        # Let's assume we can use `vio.track_features` if it wraps the tracker.
        # `VIOSystem` has `track_features`?
        # Yes, from my previous read.

        vio.track_features(dpc_state_ned, gyro, dt) # Updates internal tracker state

        # Now update measurements (Keyframe)
        # Assuming track_features populates `vio.tracker.prev_projections` or similar
        vio.update_measurements(-s['pz'], -s['vz'], None) # Height, Vz, Tracks=None (pulls from tracker)

        est = vio.get_state_dict()
        est_v_ned = np.array([est['vx'], est['vy'], est['vz']])

        err = np.linalg.norm(est_v_ned - gt_v_ned)
        errors.append(err)

        if i % 10 == 0:
            print(f"T={i*dt:.2f} GT_V={gt_v_ned} Est_V={est_v_ned} Err={err:.3f}")

        # Control
        # Simple P on pitch
        pitch_err = target_pitch - s['pitch']
        q_cmd = pitch_err * 2.0
        sim.step(np.array([target_thrust, 0.0, q_cmd, 0.0]))

    avg_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"Mean Error: {avg_err:.3f} m/s")
    print(f"Max Error: {max_err:.3f} m/s")

    if avg_err > 1.0:
        print("FAIL: Velocity Error too high")
        sys.exit(1)
    else:
        print("PASS: Velocity Error acceptable")

if __name__ == "__main__":
    run_linear_flight_test()
