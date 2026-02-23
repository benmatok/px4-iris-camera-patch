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
    sim.reset_to_scenario("Blind Dive", pos_x=0.0, pos_y=0.0, pos_z=100.0, pitch=0.0, yaw=0.0)

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
                draw_radius = max(5, int(r)) # Bigger
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

    errors = []
    duration = 1.5
    dt = 0.05
    steps = int(duration / dt)

    target_pitch = -0.1
    target_thrust = 0.6

    print(f"Init: P={p0_ned}, V={v0_ned}, Q={q0_ned}")

    for i in range(steps):
        s = sim.get_state()

        gt_v_ned = np.array([s['vy'], s['vx'], -s['vz']])
        gyro = np.array([s['wx'], -s['wy'], -s['wz']])
        accel = np.array([s.get('ax_b', 0.0), -s.get('ay_b', 0.0), -s.get('az_b', 9.81)])

        vio.propagate(gyro, accel, dt)

        img = sim.get_image(target_pos)
        vio.update_measurements(-s['pz'], -s['vz'], img)

        est = vio.get_state_dict()
        est_v_ned = np.array([est['vx'], est['vy'], est['vz']])

        err = np.linalg.norm(est_v_ned - gt_v_ned)
        errors.append(err)

        if i == 10: # Save one frame
            cv2.imwrite("debug_img.png", img)
            # Check GFTT
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            print(f"DEBUG: GFTT found {len(pts) if pts is not None else 0} points.")
            if pts is not None:
                print(f"DEBUG: Point 0 at {pts[0]}")

        if i % 1 == 0:
            tracks = len(vio.tracker.prev_projections) if hasattr(vio.tracker, 'prev_projections') else 0
            print(f"T={i*dt:.2f} Err={err:.3f} Trk={tracks}")
            # print(f"  Est_V={est_v_ned}")
            # print(f"  Q_est={est['q']}")

        pitch_err = target_pitch - s['pitch']
        q_cmd = pitch_err * 2.0
        sim.step(np.array([target_thrust, 0.0, q_cmd, 0.0]))

    avg_err = np.mean(errors)
    print(f"Mean Error: {avg_err:.3f} m/s")

if __name__ == "__main__":
    run_linear_flight_test()
