import sys
import os
import numpy as np
import logging
import cv2
from sim_interface import SimDroneInterface
from vision.projection import Projector
from vision.vio_system import VIOSystem
from flight_config import FlightConfig
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLinearFlight")

def run_linear_flight_test():
    print("--- Running Linear Flight VIO Test ---")

    config = FlightConfig()
    cam = config.camera
    projector = Projector(width=cam.width, height=cam.height, fov_deg=cam.fov_deg, tilt_deg=cam.tilt_deg)

    sim = SimDroneInterface(projector, config=config)
    # Start at high altitude
    sim.reset_to_scenario("Blind Dive", pos_x=0.0, pos_y=0.0, pos_z=100.0, pitch=0.0, yaw=0.0)

    # Mock Sim.get_image to generate features
    original_get_image = sim.get_image
    def get_image_white(target_pos_world):
        width = 640
        height = 480
        img = np.zeros((height, width, 3), dtype=np.uint8)
        s = sim.state
        yaw_ned = (np.pi / 2.0) - s['yaw']
        yaw_ned = (yaw_ned + np.pi) % (2 * np.pi) - np.pi
        drone_state_ned = {
            'px': s['py'], 'py': s['px'], 'pz': -s['pz'],
            'roll': s['roll'], 'pitch': s['pitch'], 'yaw': yaw_ned
        }
        tx_sim, ty_sim, tz_sim = target_pos_world
        tx_ned = ty_sim; ty_ned = tx_sim; tz_ned = -tz_sim
        res = sim.projector.project_point_with_size(tx_ned, ty_ned, tz_ned, drone_state_ned, object_radius=0.5)
        if res:
            u, v, r = res
            if 0 <= u < width and 0 <= v < height:
                draw_radius = max(5, int(r)) # Bigger to ensure detection
                cv2.circle(img, (int(u), int(v)), draw_radius, (255, 255, 255), -1)
        return img
    sim.get_image = get_image_white

    vio = VIOSystem(projector, config=config)

    s0 = sim.get_state()
    p0_ned = np.array([s0['py'], s0['px'], -s0['pz']])
    v0_ned = np.array([s0['vy'], s0['vx'], -s0['vz']])
    yaw_ned = (np.pi / 2.0) - s0['yaw']
    yaw_ned = (yaw_ned + np.pi) % (2 * np.pi) - np.pi
    r_ned = R.from_euler('zyx', [yaw_ned, s0['pitch'], s0['roll']])
    q0_ned = r_ned.as_quat()

    vio.initialize(q=q0_ned, p=p0_ned, v=v0_ned)

    target_pos = [100.0, 50.0, 0.0]

    vel_errors = []
    pos_errors = []
    foe_errors = []

    duration = 2.0
    dt = 0.05
    steps = int(duration / dt)

    target_pitch = -0.1
    target_thrust = 0.6

    print(f"Init: P={p0_ned}, V={v0_ned}, Q={q0_ned}")

    for i in range(steps):
        s = sim.get_state()

        # Ground Truth
        gt_p_ned = np.array([s['py'], s['px'], -s['pz']])
        gt_v_ned = np.array([s['vy'], s['vx'], -s['vz']])

        # Calculate GT FOE
        # Transform velocity to Body Frame
        # R_wb maps Body to World (NED)
        # V_b = R_wb^T * V_w
        # Need current orientation R_wb
        yaw_curr = (np.pi / 2.0) - s['yaw']
        yaw_curr = (yaw_curr + np.pi) % (2 * np.pi) - np.pi
        r_curr = R.from_euler('zyx', [yaw_curr, s['pitch'], s['roll']])
        v_b = r_curr.inv().apply(gt_v_ned)

        # FOE in normalized coords (u = vy/vx, v = vz/vx) assuming X is forward
        # Camera is usually Z forward?
        # Sim Projector assumes NED Body.
        # Projector: X=North(Fwd), Y=East(Right), Z=Down.
        # So FOE u = vy/vx, v = vz/vx is correct for normalized plane Z=1?
        # No, Projector implementation:
        # P_c = T_bc * P_b
        # Let's trust Projector logic.
        # Project velocity vector as a point at infinity?
        # Or just use simple approximation for checking trend.

        gt_foe = None
        if v_b[0] > 0.1: # Moving forward
             # Pinhole model for point at infinity [vx, vy, vz]
             # If Body X is Forward.
             # Camera usually mounted with Tilt.
             # T_bc rotates Body to Camera.
             # V_c = R_bc * V_b
             # Project V_c.
             # Projector handles T_bc.
             # We can't easily access T_bc here without duplicating logic.
             # But we can assume VIO estimates state, so let's check State Error.
             pass

        gyro = np.array([s['wx'], -s['wy'], -s['wz']])
        accel = np.array([s.get('ax_b', 0.0), -s.get('ay_b', 0.0), -s.get('az_b', 9.81)])

        vio.propagate(gyro, accel, dt)

        img = sim.get_image(target_pos)
        vio.update_measurements(-s['pz'], -s['vz'], img)

        est = vio.get_state_dict()
        est_v_ned = np.array([est['vx'], est['vy'], est['vz']])
        est_p_ned = np.array([est['px'], est['py'], est['pz']])

        v_err = np.linalg.norm(est_v_ned - gt_v_ned)
        p_err = np.linalg.norm(est_p_ned - gt_p_ned)

        vel_errors.append(v_err)
        pos_errors.append(p_err)

        # FOE Error
        # VIO FOE is derived from V_est.
        # GT FOE is derived from V_gt.
        # We can just compare the V vectors direction cosine or angle.
        # Cos sim: dot(v1, v2) / (|v1||v2|)

        if np.linalg.norm(gt_v_ned) > 0.1 and np.linalg.norm(est_v_ned) > 0.1:
             cos_sim = np.dot(gt_v_ned, est_v_ned) / (np.linalg.norm(gt_v_ned) * np.linalg.norm(est_v_ned))
             cos_sim = np.clip(cos_sim, -1.0, 1.0)
             angle_err_deg = np.degrees(np.arccos(cos_sim))
             foe_errors.append(angle_err_deg)

        if i % 5 == 0:
            print(f"T={i*dt:.2f} V_Err={v_err:.3f}m/s P_Err={p_err:.3f}m FOE_Err={foe_errors[-1] if foe_errors else 0.0:.1f}deg")
            print(f"  GT_V={gt_v_ned}")
            print(f"  Est_V={est_v_ned}")

        pitch_err = target_pitch - s['pitch']
        q_cmd = pitch_err * 2.0
        sim.step(np.array([target_thrust, 0.0, q_cmd, 0.0]))

    print("-" * 30)
    print(f"Mean Pos Error: {np.mean(pos_errors):.3f} m")
    print(f"Mean Vel Error: {np.mean(vel_errors):.3f} m/s")
    if foe_errors:
        print(f"Mean FOE Angle Error: {np.mean(foe_errors):.3f} deg")
    else:
        print("Mean FOE Angle Error: N/A")

if __name__ == "__main__":
    run_linear_flight_test()
