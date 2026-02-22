import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.projection import Projector
from sim_interface import SimDroneInterface
from visual_tracker import VisualTracker
from vision.vio_system import VIOSystem
from flight_controller import DPCFlightController
from mission_manager import MissionManager
from flight_config import FlightConfig

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DT = 0.05

class VIOValidator:
    def __init__(self, use_ground_truth=False, use_blind_mode=True, init_alt=50.0, init_dist=150.0, config: FlightConfig = None):
        self.use_ground_truth = use_ground_truth
        self.use_blind_mode = use_blind_mode
        self.config = config or FlightConfig()

        cam = self.config.camera

        # Tilt 30.0 (Up)
        self.projector = Projector(width=cam.width, height=cam.height, fov_deg=cam.fov_deg, tilt_deg=cam.tilt_deg)

        # Scenario / Sim
        self.sim = SimDroneInterface(self.projector)

        # Target at X=init_dist, Y=0, Z=0
        self.target_pos_sim_world = [init_dist, 0.0, 0.0]

        # Initial Pos: X=0, Y=0, Z=init_alt
        drone_pos = [0.0, 0.0, init_alt]
        pitch, yaw = self.calculate_initial_orientation(drone_pos, self.target_pos_sim_world)

        self.sim.reset_to_scenario("Blind Dive", pos_x=drone_pos[0], pos_y=drone_pos[1], pos_z=drone_pos[2], pitch=pitch, yaw=yaw)

        # Perception
        self.tracker = VisualTracker(self.projector)

        # VIO
        self.vio_system = VIOSystem(self.projector)

        # Logic
        self.mission = MissionManager(target_alt=init_alt, enable_staircase=self.config.mission.enable_staircase, config=self.config)

        # Control
        self.controller = DPCFlightController(dt=DT, config=self.config)

    def calculate_initial_orientation(self, drone_pos, target_pos):
        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]
        dz = target_pos[2] - drone_pos[2]

        yaw = np.arctan2(dy, dx)
        dist_xy = np.sqrt(dx*dx + dy*dy)
        pitch_vec = np.arctan2(dz, dist_xy)

        camera_tilt = np.deg2rad(self.config.camera.tilt_deg)
        pitch = (pitch_vec - camera_tilt)

        if pitch < -1.48:
            pitch = -1.48

        return pitch, yaw

    def sim_to_ned(self, sim_state):
        # Maps Sim (ENU) to NED (Right-Handed)
        ned = sim_state.copy()
        ned['px'] = sim_state['py']
        ned['py'] = sim_state['px']
        ned['pz'] = -sim_state['pz']

        ned['vx'] = sim_state['vy']
        ned['vy'] = sim_state['vx']
        ned['vz'] = -sim_state['vz']

        ned['roll'] = sim_state['roll']
        ned['pitch'] = sim_state['pitch']
        ned['yaw'] = (math.pi / 2.0) - sim_state['yaw']
        ned['yaw'] = (ned['yaw'] + math.pi) % (2 * math.pi) - math.pi

        return ned

    def sim_pos_to_ned_pos(self, sim_pos):
        return [sim_pos[1], sim_pos[0], -sim_pos[2]]

    def ned_rel_to_sim_rel(self, ned_rel):
        return [ned_rel[1], ned_rel[0], -ned_rel[2]]

    def run(self, duration=10.0):
        steps = int(duration / DT)
        history = {
            't': [],
            'gt_pos': [],
            'est_pos': [],
            'gt_vel': [],
            'est_vel': [],
            'pos_err': [],
            'vel_err': [],
            'vel_reliable': []
        }

        logger.info(f"Running VIO Validation for {duration}s...")

        for i in range(steps):
            t = i * DT

            # 1. Get State (Sim Frame)
            s = self.sim.get_state()

            # 2. Get Perception Data
            img = self.sim.get_image(self.target_pos_sim_world)

            # Convert Sim State to NED (Absolute) for GT
            dpc_state_ned_abs = self.sim_to_ned(s)
            gt_pos = [dpc_state_ned_abs['px'], dpc_state_ned_abs['py'], dpc_state_ned_abs['pz']]
            gt_vel = [dpc_state_ned_abs['vx'], dpc_state_ned_abs['vy'], dpc_state_ned_abs['vz']]

            # Convert Target Sim Pos to NED Pos (Absolute)
            target_pos_ned_abs = self.sim_pos_to_ned_pos(self.target_pos_sim_world)

            ground_truth = target_pos_ned_abs if self.use_ground_truth else None

            # Detect and Localize
            center, target_wp_ned, radius = self.tracker.process(
                img,
                dpc_state_ned_abs, # Used internally for generation?
                ground_truth_target_pos=ground_truth
            )

            # Convert Result (target_wp_ned) back to Sim Relative Frame
            target_wp_sim = None
            if target_wp_ned:
                 target_wp_sim = self.ned_rel_to_sim_rel(target_wp_ned)

            # 3. Update Mission Logic
            sim_state_rel = s.copy()
            sim_state_rel['px'] = 0.0
            sim_state_rel['py'] = 0.0

            mission_state, dpc_target, extra_yaw = self.mission.update(sim_state_rel, (center, target_wp_sim))

            # 4. Compute Control
            state_obs = {
                'px': s['px'],
                'py': s['py'],
                'pz': s['pz'],
                'vx': s['vx'] if not self.use_blind_mode else 0.0,
                'vy': s['vy'] if not self.use_blind_mode else 0.0,
                'vz': s['vz'] if not self.use_blind_mode else None,
                'roll': s['roll'],
                'pitch': s['pitch'],
                'yaw': s['yaw'],
                'wx': s['wx'],
                'wy': s['wy'],
                'wz': s['wz']
            }

            tracking_norm = None
            tracking_size_norm = None
            if center:
                tracking_norm = self.projector.pixel_to_normalized(center[0], center[1])
                tracking_size_norm = radius / 480.0

            # --- VIO UPDATE (BA System) ---

            # IMU Data
            gyro = np.array([s['wx'], -s['wy'], -s['wz']], dtype=np.float64)
            accel = np.array([s.get('ax_b', 0.0), -s.get('ay_b', 0.0), -s.get('az_b', 9.81)], dtype=np.float64)

            # Init VIO if needed (Ground Truth Init)
            if not self.vio_system.initialized:
                from scipy.spatial.transform import Rotation as R
                r_angle = dpc_state_ned_abs['roll']
                p_angle = dpc_state_ned_abs['pitch']
                y_angle = dpc_state_ned_abs['yaw']
                q_init = R.from_euler('zyx', [y_angle, p_angle, r_angle], degrees=False).as_quat()
                p_init = np.array([dpc_state_ned_abs['px'], dpc_state_ned_abs['py'], dpc_state_ned_abs['pz']])
                v_init = np.array([dpc_state_ned_abs['vx'], dpc_state_ned_abs['vy'], dpc_state_ned_abs['vz']])
                self.vio_system.initialize(q_init, p_init, v_init)

            # Propagate (Buffer IMU)
            self.vio_system.propagate(gyro, accel, DT)

            # Track Features (Simulated)
            body_rates_ned = (gyro[0], gyro[1], gyro[2])
            self.vio_system.track_features(dpc_state_ned_abs, body_rates_ned, DT)

            # Update Measurements (Keyframe)
            height_meas = dpc_state_ned_abs['pz']
            vz_meas = dpc_state_ned_abs['vz']

            # Estimate Homography Velocity
            vel_prior = None
            if self.projector:
                h_est = -dpc_state_ned_abs['pz']
                if h_est > 1.0:
                    v_body_hom = self.vio_system.tracker.estimate_homography_velocity(DT, h_est, None)
                    if v_body_hom is not None:
                        # Rotate to World NED
                        from scipy.spatial.transform import Rotation as R
                        q_curr = self.vio_system.state_cache.get('q', np.array([0,0,0,1]))
                        R_mat = R.from_quat(q_curr).as_matrix()
                        vel_prior = R_mat @ v_body_hom

            if vel_prior is not None:
                vp_err = np.linalg.norm(vel_prior - np.array(gt_vel))
                if i % 10 == 0:
                    print(f"T={t:.2f} VelPrior Err: {vp_err:.2f} m/s")

            self.vio_system.update_measurements(height_meas, vz_meas, None, velocity_prior=vel_prior)

            # Get State
            vio_state = self.vio_system.get_state_dict()
            est_pos = [vio_state['px'], vio_state['py'], vio_state['pz']]
            est_vel = [vio_state['vx'], vio_state['vy'], vio_state['vz']]

            vel_reliable = self.vio_system.is_reliable()

            # Calc Errors
            pos_err = np.linalg.norm(np.array(est_pos) - np.array(gt_pos))
            vel_err = np.linalg.norm(np.array(est_vel) - np.array(gt_vel))

            # Store History
            history['t'].append(t)
            history['gt_pos'].append(gt_pos)
            history['est_pos'].append(est_pos)
            history['gt_vel'].append(gt_vel)
            history['est_vel'].append(est_vel)
            history['pos_err'].append(pos_err)
            history['vel_err'].append(vel_err)
            history['vel_reliable'].append(vel_reliable)

            # Use VIO velocity for controller
            vel_est = {'vx': vio_state['vx'], 'vy': vio_state['vy'], 'vz': vio_state['vz']}
            pos_est = {'px': vio_state['px'], 'py': vio_state['py'], 'pz': vio_state['pz']}

            action_out, ghost_paths = self.controller.compute_action(
                state_obs,
                dpc_target,
                tracking_uv=tracking_norm,
                tracking_size=tracking_size_norm,
                extra_yaw_rate=extra_yaw,
                velocity_est=vel_est,
                position_est=pos_est,
                velocity_reliable=vel_reliable
            )

            # 5. Apply Control to Sim
            sim_action = np.array([
                action_out['thrust'],
                action_out['roll_rate'],
                action_out['pitch_rate'],
                action_out['yaw_rate']
            ])

            if mission_state == "TAKEOFF" and s['pz'] < 2.0:
                 sim_action[0] = 0.8

            self.sim.step(sim_action)

            if s['pz'] < 0.2:
                 logger.info("Ground Impact! Stopping.")
                 break

        return history

def plot_vio_accuracy(history, filename="validation_vio_accuracy.png"):
    t = np.array(history['t'])
    gt_pos = np.array(history['gt_pos'])
    est_pos = np.array(history['est_pos'])
    gt_vel = np.array(history['gt_vel'])
    est_vel = np.array(history['est_vel'])
    pos_err = np.array(history['pos_err'])
    vel_err = np.array(history['vel_err'])

    fig, axs = plt.subplots(4, 2, figsize=(15, 20))

    # 1. Position X
    axs[0, 0].set_title("Position X (NED)")
    axs[0, 0].plot(t, gt_pos[:, 0], 'b-', label='GT')
    axs[0, 0].plot(t, est_pos[:, 0], 'r--', label='Est')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Position Y
    axs[0, 1].set_title("Position Y (NED)")
    axs[0, 1].plot(t, gt_pos[:, 1], 'b-', label='GT')
    axs[0, 1].plot(t, est_pos[:, 1], 'r--', label='Est')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Position Z
    axs[1, 0].set_title("Position Z (NED)")
    axs[1, 0].plot(t, gt_pos[:, 2], 'b-', label='GT')
    axs[1, 0].plot(t, est_pos[:, 2], 'r--', label='Est')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Position Error
    axs[1, 1].set_title("Position Error Norm (m)")
    axs[1, 1].plot(t, pos_err, 'k-')
    axs[1, 1].grid(True)

    # 5. Velocity X
    axs[2, 0].set_title("Velocity X (NED)")
    axs[2, 0].plot(t, gt_vel[:, 0], 'b-', label='GT')
    axs[2, 0].plot(t, est_vel[:, 0], 'r--', label='Est')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # 6. Velocity Y
    axs[2, 1].set_title("Velocity Y (NED)")
    axs[2, 1].plot(t, gt_vel[:, 1], 'b-', label='GT')
    axs[2, 1].plot(t, est_vel[:, 1], 'r--', label='Est')
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # 7. Velocity Z
    axs[3, 0].set_title("Velocity Z (NED)")
    axs[3, 0].plot(t, gt_vel[:, 2], 'b-', label='GT')
    axs[3, 0].plot(t, est_vel[:, 2], 'r--', label='Est')
    axs[3, 0].legend()
    axs[3, 0].grid(True)

    # 8. Velocity Error
    axs[3, 1].set_title("Velocity Error Norm (m/s)")
    axs[3, 1].plot(t, vel_err, 'k-')
    axs[3, 1].grid(True)

    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
    except:
        pass

if __name__ == "__main__":
    validator = VIOValidator(use_ground_truth=True, use_blind_mode=True, init_alt=50.0, init_dist=150.0)
    # Reduced duration for faster feedback loop
    history = validator.run(duration=2.0)

    plot_vio_accuracy(history)

    # Metrics
    mean_pos_err = np.mean(history['pos_err'])
    max_pos_err = np.max(history['pos_err'])
    mean_vel_err = np.mean(history['vel_err'])
    max_vel_err = np.max(history['vel_err'])

    print(f"\n--- VIO Accuracy Metrics ---")
    print(f"Mean Pos Error: {mean_pos_err:.4f} m")
    print(f"Max Pos Error: {max_pos_err:.4f} m")
    print(f"Mean Vel Error: {mean_vel_err:.4f} m/s")
    print(f"Max Vel Error: {max_vel_err:.4f} m/s")
