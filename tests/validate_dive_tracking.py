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
from vision.msckf import MSCKF
from vision.feature_tracker import FeatureTracker
from flight_controller import DPCFlightController
from mission_manager import MissionManager
from flight_config import FlightConfig

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DT = 0.05

class DiveValidator:
    def __init__(self, use_ground_truth=True, use_blind_mode=False, init_alt=50.0, init_dist=150.0, config: FlightConfig = None):
        self.use_ground_truth = use_ground_truth
        self.use_blind_mode = use_blind_mode
        self.config = config or FlightConfig()

        cam = self.config.camera

        # Tilt 30.0 (Up) as in theshow.py
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
        self.msckf = MSCKF(self.projector)
        self.feature_tracker = FeatureTracker(self.projector)
        self.msckf.initialized = False

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

        # Camera Tilt is 30.0 (Up)
        camera_tilt = np.deg2rad(self.config.camera.tilt_deg)

        # PyGhostModel Convention: Positive Pitch = Nose Down.
        # But flight_controller uses Positive Pitch = Nose Up.
        # We need to set initial pitch in Sim convention?
        # Let's try to match target.
        # Vector is ~ -18 deg (Down). Camera is +30 deg (Up).
        # We need Body = -48 deg (Nose Up).
        # If Sim Pitch is + = Nose Down, then -48 is -48.
        # Wait, if Sim Pitch is - = Nose Up.
        # Then we need -48.
        # Sim Pitch Negative = Nose Down.
        pitch = (pitch_vec - camera_tilt)

        # Clamp pitch
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
        # Sim Pitch (Nose Down -) -> NED Pitch (Nose Down -). Match signs.
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
            'drone_pos': [],
            'target_est': [],
            'dist': [],
            'state': [],
            'vel_reliable': [],
            'velocity_error': [],
            'foe_error': []
        }

        logger.info(f"Running Validation (Ground Truth: {self.use_ground_truth}, Blind Mode: {self.use_blind_mode}) for {duration}s...")

        for i in range(steps):
            t = i * DT

            # 1. Get State (Sim Frame)
            s = self.sim.get_state()

            # 2. Get Perception Data
            img = self.sim.get_image(self.target_pos_sim_world)

            # Convert Sim State to NED (Absolute)
            dpc_state_ned_abs = self.sim_to_ned(s)

            # Convert Target Sim Pos to NED Pos (Absolute)
            target_pos_ned_abs = self.sim_pos_to_ned_pos(self.target_pos_sim_world)

            ground_truth = target_pos_ned_abs if self.use_ground_truth else None

            # Detect and Localize
            center, target_wp_ned, radius = self.tracker.process(
                img,
                dpc_state_ned_abs,
                ground_truth_target_pos=ground_truth
            )

            if i < 5:
                print(f"DEBUG: T={t:.2f}")
                print(f"  Drone NED: px={dpc_state_ned_abs['px']:.1f}, py={dpc_state_ned_abs['py']:.1f}, pz={dpc_state_ned_abs['pz']:.1f}, pitch={math.degrees(dpc_state_ned_abs['pitch']):.1f}, yaw={math.degrees(dpc_state_ned_abs['yaw']):.1f}")
                print(f"  Target NED: {target_pos_ned_abs}")
                print(f"  Center UV: {center}")

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
                'vx': s['vx'] if not self.use_blind_mode else 0.0, # Blind Mode Check
                'vy': s['vy'] if not self.use_blind_mode else 0.0,
                'vz': s['vz'] if not self.use_blind_mode else None, # Blind Mode: No VZ
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

            # --- VIO UPDATE ---

            # IMU Data
            # s has ax_b, ay_b, az_b from my patch
            gyro = np.array([s['wx'], s['wy'], s['wz']], dtype=np.float64)
            accel = np.array([s.get('ax_b', 0.0), s.get('ay_b', 0.0), s.get('az_b', 9.81)], dtype=np.float64)

            # Init VIO if needed (Ground Truth Init)
            if not self.msckf.initialized:
                from scipy.spatial.transform import Rotation as R
                r_angle = dpc_state_ned_abs['roll']
                p_angle = dpc_state_ned_abs['pitch']
                y_angle = dpc_state_ned_abs['yaw']
                q_init = R.from_euler('xyz', [r_angle, p_angle, y_angle], degrees=False).as_quat()

                p_init = np.array([dpc_state_ned_abs['px'], dpc_state_ned_abs['py'], dpc_state_ned_abs['pz']])
                v_init = np.array([dpc_state_ned_abs['vx'], dpc_state_ned_abs['vy'], dpc_state_ned_abs['vz']])

                self.msckf.initialize(q_init, p_init, v_init)

            # Propagate
            self.msckf.propagate(gyro, accel, DT)

            # Augment
            self.msckf.augment_state()

            # Features
            # Body Rates
            body_rates = (s['wx'], s['wy'], s['wz'])
            current_clone_idx = self.msckf.cam_clones[-1]['id'] if self.msckf.cam_clones else 0

            foe, finished_tracks = self.feature_tracker.update(dpc_state_ned_abs, body_rates, DT, current_clone_idx)

            if finished_tracks:
                self.msckf.update_features(finished_tracks)

            # Height
            height_meas = dpc_state_ned_abs['pz']
            self.msckf.update_height(height_meas)

            # Get VIO Output
            vio_vel = self.msckf.get_velocity()
            vel_est = {'vx': vio_vel[0], 'vy': vio_vel[1], 'vz': vio_vel[2]}

            # Calculate Errors
            v_true = np.array([dpc_state_ned_abs['vx'], dpc_state_ned_abs['vy'], dpc_state_ned_abs['vz']])
            v_est_arr = np.array(vio_vel)
            vel_err = np.linalg.norm(v_est_arr - v_true)

            # FOE Error
            foe_err = 0.0
            if foe:
                from scipy.spatial.transform import Rotation as R
                r = dpc_state_ned_abs['roll']
                p = dpc_state_ned_abs['pitch']
                y = dpc_state_ned_abs['yaw']
                R_wb = R.from_euler('xyz', [r, p, y], degrees=False).as_matrix()

                v_body = R_wb.T @ v_true
                v_cam = self.projector.R_c2b.T @ v_body

                if v_cam[2] > 0.1:
                    true_u = v_cam[0] / v_cam[2]
                    true_v = v_cam[1] / v_cam[2]

                    est_u, est_v = foe
                    foe_err = np.sqrt((est_u - true_u)**2 + (est_v - true_v)**2)

            history['velocity_error'].append(vel_err)
            history['foe_error'].append(foe_err)

            vel_reliable = self.msckf.is_reliable()

            # Use VIO velocity for controller
            action_out, _ = self.controller.compute_action(
                state_obs,
                dpc_target,
                tracking_uv=tracking_norm,
                tracking_size=tracking_size_norm,
                extra_yaw_rate=extra_yaw,
                velocity_est=vel_est,
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

            dist = np.sqrt((s['px'] - self.target_pos_sim_world[0])**2 +
                           (s['py'] - self.target_pos_sim_world[1])**2 +
                           (s['pz'] - self.target_pos_sim_world[2])**2)
            history['dist'].append(dist)
            history['state'].append(mission_state)
            history['vel_reliable'].append(vel_reliable)

            # Logging
            history['t'].append(t)
            history['drone_pos'].append([s['px'], s['py'], s['pz']])
            if i % 20 == 0:
                uv_str = f"UV=({center[0]:.2f}, {center[1]:.2f})" if center else "UV=None"
                print(f"T={t:.2f} State={mission_state} Dist={dist:.2f} Pos=({s['px']:.1f}, {s['py']:.1f}, {s['pz']:.1f}) {uv_str}")

            # Terminate if collision (close enough)
            if dist < 2.0: # Close collision (Relaxed)
                 logger.info("Collision Detected! Stopping.")
                 break

            if s['pz'] < 0.2:
                 logger.info("Ground Impact! Stopping.")
                 break

            # Estimate Target Pos (Absolute Sim Frame)
            if target_wp_sim:
                est_x = s['px'] + target_wp_sim[0]
                est_y = s['py'] + target_wp_sim[1]
                est_z = s['pz'] + target_wp_sim[2]
                history['target_est'].append([est_x, est_y, est_z])
            else:
                history['target_est'].append([None, None, None])

        # Print Error Statistics
        vel_errs = np.array(history['velocity_error'])
        foe_errs = np.array(history['foe_error'])
        if len(vel_errs) > 0:
            logger.info(f"Velocity Error: Mean={np.mean(vel_errs):.4f}, Max={np.max(vel_errs):.4f}")
            logger.info(f"FOE Error: Mean={np.mean(foe_errs):.4f}, Max={np.max(foe_errs):.4f}")

        return history

def plot_results(hist_gt, hist_vis, hist_blind=None, filename="validation_dive_tracking.png", target_pos=None):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Trajectory Side View (X-Z)
    axs[0, 0].set_title("Trajectory Side View (X-Z)")

    # Ground Truth Run
    pos_gt = np.array(hist_gt['drone_pos'])
    axs[0, 0].plot(pos_gt[:, 0], pos_gt[:, 2], 'b-', label='Full State (GT Tracking)')

    # Vision Run
    if hist_vis:
        pos_vis = np.array(hist_vis['drone_pos'])
        axs[0, 0].plot(pos_vis[:, 0], pos_vis[:, 2], 'g--', label='Full State (Vision Tracking)')

    # Blind Run (Web App)
    if hist_blind:
        pos_blind = np.array(hist_blind['drone_pos'])
        axs[0, 0].plot(pos_blind[:, 0], pos_blind[:, 2], 'r:', linewidth=2, label='Blind Mode (Web App)')

        # Mark VIO Lock
        if 'vel_reliable' in hist_blind:
            try:
                # Find first index where True
                lock_idx = next(i for i, v in enumerate(hist_blind['vel_reliable']) if v)
                lock_pos = pos_blind[lock_idx]
                axs[0, 0].plot(lock_pos[0], lock_pos[2], 'mo', markersize=8, label='VIO Lock', zorder=5)
            except StopIteration:
                pass

    # Target
    if target_pos is not None:
        axs[0, 0].plot(target_pos[0], target_pos[2], 'rx', markersize=10, label='Target')
    else:
        axs[0, 0].plot(150.0, 0.0, 'rx', markersize=10, label='Target (Default)')

    axs[0, 0].set_xlabel("X (m)")
    axs[0, 0].set_ylabel("Z (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Distance to Target
    axs[0, 1].set_title("Distance to Target")
    axs[0, 1].plot(hist_gt['t'], hist_gt['dist'], 'b-', label='Full GT')
    if hist_vis:
        axs[0, 1].plot(hist_vis['t'], hist_vis['dist'], 'g--', label='Full Vision')
    if hist_blind:
        axs[0, 1].plot(hist_blind['t'], hist_blind['dist'], 'r:', label='Blind Web')
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Distance (m)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Velocity Error
    axs[1, 0].set_title("Velocity Error (Norm)")
    if hist_vis:
        axs[1, 0].plot(hist_vis['t'], hist_vis['velocity_error'], 'g--', label='Vision Vel Error')
    if hist_blind:
        axs[1, 0].plot(hist_blind['t'], hist_blind['velocity_error'], 'r:', label='Blind Web Vel Error')
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Error (m/s)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. FOE Error
    axs[1, 1].set_title("FOE Error (Normalized UV)")
    if hist_vis:
        axs[1, 1].plot(hist_vis['t'], hist_vis['foe_error'], 'g--', label='Vision FOE Error')
    if hist_blind:
        axs[1, 1].plot(hist_blind['t'], hist_blind['foe_error'], 'r:', label='Blind Web FOE Error')
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Error (Unitless)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
    except:
        pass

if __name__ == "__main__":
    # Run with Ground Truth (Full State)
    validator_gt = DiveValidator(use_ground_truth=True, use_blind_mode=False, init_alt=50.0, init_dist=150.0)
    hist_gt = validator_gt.run(duration=25.0)

    # Run with Vision (Full State)
    validator_vis = DiveValidator(use_ground_truth=False, use_blind_mode=False, init_alt=50.0, init_dist=150.0)
    hist_vis = validator_vis.run(duration=25.0)

    # Run with Blind Mode (Web App Logic)
    validator_blind = DiveValidator(use_ground_truth=True, use_blind_mode=True, init_alt=50.0, init_dist=150.0) # Use GT tracking for fair comparison of control
    hist_blind = validator_blind.run(duration=25.0)

    plot_results(hist_gt, hist_vis, hist_blind, target_pos=[150.0, 0.0, 0.0])

    # Validation Checks
    final_dist_gt = hist_gt['dist'][-1]
    final_dist_vis = hist_vis['dist'][-1]
    final_dist_blind = hist_blind['dist'][-1]

    print(f"Final Distance (Full GT): {final_dist_gt:.2f}m")
    print(f"Final Distance (Full Vision): {final_dist_vis:.2f}m")
    print(f"Final Distance (Blind Web): {final_dist_blind:.2f}m")

    # STRICT CRITERIA: Must be close to 0 (Collision)
    if final_dist_gt < 2.0:
        print("SUCCESS: Full GT Dive Successful")
    else:
        print("FAILURE: Full GT Dive Failed")

    if final_dist_vis < 2.0:
        print("SUCCESS: Full Vision Dive Successful")
    else:
        print("FAILURE: Full Vision Dive Failed")

    if final_dist_blind < 2.0:
        print("SUCCESS: Blind Web Dive Successful")
    else:
        print("FAILURE: Blind Web Dive Failed")

    # Similarity Check (Full GT vs Blind Web)
    # This is to check if the control logic is reasonably consistent
    pos_gt = np.array(hist_gt['drone_pos'])
    pos_blind = np.array(hist_blind['drone_pos'])

    # Calculate Max Deviation over overlapping time
    min_len = min(len(pos_gt), len(pos_blind))
    diffs = []
    for i in range(min_len):
        d = np.linalg.norm(pos_gt[i] - pos_blind[i])
        diffs.append(d)

    max_dev = max(diffs)
    mean_dev = np.mean(diffs)
    print(f"Max Deviation (Full vs Blind): {max_dev:.2f}m")
    print(f"Mean Deviation (Full vs Blind): {mean_dev:.2f}m")
