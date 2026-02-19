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
            'ghost_paths': []
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
            # s has ax_b, ay_b, az_b in Sim Frame (Forward-Left-Up)
            # VIO expects NED Body Frame (Forward-Right-Down)
            # Transformation: X->X, Y->-Y, Z->-Z
            gyro = np.array([s['wx'], -s['wy'], -s['wz']], dtype=np.float64)
            accel = np.array([s.get('ax_b', 0.0), -s.get('ay_b', 0.0), -s.get('az_b', 9.81)], dtype=np.float64)

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
            # Body Rates (NED for Feature Tracker)
            # Use the already transformed 'gyro' array
            body_rates_ned = (gyro[0], gyro[1], gyro[2])
            current_clone_idx = self.msckf.cam_clones[-1]['id'] if self.msckf.cam_clones else 0

            foe, finished_tracks = self.feature_tracker.update(dpc_state_ned_abs, body_rates_ned, DT, current_clone_idx)

            if finished_tracks:
                self.msckf.update_features(finished_tracks)

            # Height
            height_meas = dpc_state_ned_abs['pz']
            self.msckf.update_height(height_meas)

            # Get VIO Output
            vio_vel = self.msckf.get_velocity()
            vel_est = {'vx': vio_vel[0], 'vy': vio_vel[1], 'vz': vio_vel[2]}

            vel_reliable = self.msckf.is_reliable()

            # Use VIO velocity for controller
            action_out, ghost_paths = self.controller.compute_action(
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
            history['ghost_paths'].append(ghost_paths)

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

        return history

def plot_results(hist_gt, hist_vis, hist_blind=None, filename="validation_dive_tracking.png", target_pos=None):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # 1. Trajectory Side View (X-Z)
    axs[0, 0].set_title("Trajectory Side View (X-Z)")

    # Blind Run (Web App) - This is the one we care about for GDPC prediction
    if hist_blind:
        pos_blind = np.array(hist_blind['drone_pos'])
        axs[0, 0].plot(pos_blind[:, 0], pos_blind[:, 2], 'k-', linewidth=2, label='Actual Trajectory')

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

    # Overlay Ghost Paths for Blind Run (if available)
    if hist_blind and 'ghost_paths' in hist_blind:
        t = np.array(hist_blind['t'])
        pos = np.array(hist_blind['drone_pos'])
        # Plot every 1.0s (approx 20 steps)
        for i in range(0, len(t), 20):
            if i >= len(hist_blind['ghost_paths']): break
            gps = hist_blind['ghost_paths'][i]
            if not gps: continue
            gp = gps[0] # Take first path

            # Check reliability for frame
            reliable = hist_blind['vel_reliable'][i]

            # Convert to absolute for plotting
            gp_abs = []
            curr_pos = pos[i]

            start_z = gp[0]['pz']

            for p in gp:
                px = p['px']
                py = p['py']
                pz = p['pz']

                abs_x = curr_pos[0] + px
                abs_y = curr_pos[1] + py

                if reliable:
                    abs_z = curr_pos[2] + pz
                else:
                    rel_z = pz - start_z
                    abs_z = curr_pos[2] + rel_z

                gp_abs.append([abs_x, abs_y, abs_z])

            gp_abs = np.array(gp_abs)
            label = 'GDPC Prediction' if i == 0 else None
            axs[0, 0].plot(gp_abs[:, 0], gp_abs[:, 2], 'c-', alpha=0.6, linewidth=1.5, label=label)

    axs[0, 0].set_xlabel("X (m)")
    axs[0, 0].set_ylabel("Z (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)


    # 2. Distance to Target
    axs[0, 1].set_title("Distance to Target")
    if hist_gt:
        axs[0, 1].plot(hist_gt['t'], hist_gt['dist'], 'b-', label='Full GT')
    if hist_vis:
        axs[0, 1].plot(hist_vis['t'], hist_vis['dist'], 'g--', label='Full Vision')
    if hist_blind:
        axs[0, 1].plot(hist_blind['t'], hist_blind['dist'], 'r:', label='Blind Web')
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Distance (m)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Target Estimation Error (Vision Run Only)
    axs[1, 0].set_title("Target Estimation Error (Vision Run)")

    # Use hist_vis if available, else hist_gt just to show something (or None)
    ref_hist = hist_vis if hist_vis else (hist_gt if hist_gt else hist_blind)

    t = np.array(ref_hist['t'])
    est = ref_hist['target_est']
    errs = []
    valid_t = []

    target_true_vec = np.array(target_pos) if target_pos else np.array([150.0, 0.0, 0.0])

    for i, p in enumerate(est):
        if p[0] is not None:
            e = np.linalg.norm(np.array(p) - target_true_vec)
            errs.append(e)
            valid_t.append(t[i])

    if valid_t:
        axs[1, 0].plot(valid_t, errs, 'r-', label='Pos Error')
    else:
        axs[1, 0].text(0.5, 0.5, "No Tracking Data", ha='center')

    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Error (m)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Mission State
    axs[1, 1].set_title("Mission State")
    states = ref_hist['state']
    # Map states to ints
    state_map = {"TAKEOFF": 0, "SCAN": 1, "HOMING": 2, "STAIRCASE_DESCEND": 3, "STAIRCASE_STABILIZE": 4}
    y_vals = [state_map.get(s, -1) for s in states]
    axs[1, 1].plot(ref_hist['t'], y_vals, 'k-')
    axs[1, 1].set_yticks([0, 1, 2, 3, 4])
    axs[1, 1].set_yticklabels(["TAKEOFF", "SCAN", "HOMING", "DESCEND", "STABILIZE"])
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].grid(True)

    # 5. Prediction Error
    axs[2, 0].set_title("Mean Prediction Error (Dynamics)")
    if hist_blind and 'ghost_paths' in hist_blind:
        t_vals = []
        err_vals = []
        pos = np.array(hist_blind['drone_pos'])

        for i in range(len(hist_blind['t'])):
            if i >= len(hist_blind['ghost_paths']): break
            gps = hist_blind['ghost_paths'][i]
            if not gps: continue
            gp = gps[0]

            # Calculate Mean Euclidean Error over the horizon
            errors = []

            # Determine start Z for relative calculation
            start_z = gp[0]['pz']
            reliable = hist_blind['vel_reliable'][i]

            for k, p in enumerate(gp):
                future_idx = i + k + 1
                if future_idx >= len(pos): break

                actual_rel = pos[future_idx] - pos[i]

                pred_rel_x = p['px']
                pred_rel_y = p['py']

                if reliable:
                    pred_rel_z = p['pz']
                else:
                    pred_rel_z = p['pz'] - start_z # Approximate relative

                pred_rel = np.array([pred_rel_x, pred_rel_y, pred_rel_z])

                dist = np.linalg.norm(pred_rel - actual_rel)
                errors.append(dist)

            if errors:
                mean_err = np.mean(errors)
                t_vals.append(hist_blind['t'][i])
                err_vals.append(mean_err)

        axs[2, 0].plot(t_vals, err_vals, 'm-', label='Pred Error')
        axs[2, 0].set_xlabel("Time (s)")
        axs[2, 0].set_ylabel("Mean Error (m)")
        axs[2, 0].legend()
        axs[2, 0].grid(True)

    # 6. VIO Reliability
    axs[2, 1].set_title("VIO Reliability")
    if hist_blind:
        rel = [1 if r else 0 for r in hist_blind['vel_reliable']]
        axs[2, 1].plot(hist_blind['t'], rel, 'k-', label='Reliable')
        axs[2, 1].set_ylim(-0.1, 1.1)
        axs[2, 1].set_xlabel("Time (s)")
        axs[2, 1].grid(True)

    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
    except:
        pass

if __name__ == "__main__":
    # Run with Blind Mode (Web App Logic) - Short run for plot validation
    validator_blind = DiveValidator(use_ground_truth=True, use_blind_mode=True, init_alt=50.0, init_dist=150.0)
    hist_blind = validator_blind.run(duration=3.0)

    # Pass None for others to skip plotting them
    plot_results(None, None, hist_blind, target_pos=[150.0, 0.0, 0.0])

    # Validation Checks
    hist_gt = None # Define variable
    hist_vis = None

    if hist_gt:
        final_dist_gt = hist_gt['dist'][-1]
        print(f"Final Distance (Full GT): {final_dist_gt:.2f}m")
        if final_dist_gt < 2.0:
            print("SUCCESS: Full GT Dive Successful")
        else:
            print("FAILURE: Full GT Dive Failed")

    if hist_vis:
        final_dist_vis = hist_vis['dist'][-1]
        print(f"Final Distance (Full Vision): {final_dist_vis:.2f}m")
        if final_dist_vis < 2.0:
            print("SUCCESS: Full Vision Dive Successful")
        else:
            print("FAILURE: Full Vision Dive Failed")

    if hist_blind:
        final_dist_blind = hist_blind['dist'][-1]
        print(f"Final Distance (Blind Web): {final_dist_blind:.2f}m")
        if final_dist_blind < 2.0:
            print("SUCCESS: Blind Web Dive Successful")
        else:
            print("FAILURE: Blind Web Dive Failed")

    # Similarity Check (Full GT vs Blind Web)
    if hist_gt and hist_blind:
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

    # --- Prediction Error Analysis (Blind Run) ---
    if hist_blind and 'ghost_paths' in hist_blind:
        print("\n--- Prediction Error Analysis (Blind/VIO Run) ---")
        pred_errors = []
        vio_vel_errors = []

        pos = np.array(hist_blind['drone_pos'])

        # Calculate errors
        for i in range(len(hist_blind['t'])):
            # VIO Velocity Error
            # Re-calculate VIO vel from reliable flag? No, we didn't store VIO vel in history.
            # We can't compute VIO error unless we stored 'velocity_est' in history.
            # Let's assume prediction error covers it.

            if i >= len(hist_blind['ghost_paths']): break
            gps = hist_blind['ghost_paths'][i]
            if not gps: continue
            gp = gps[0]

            start_z = gp[0]['pz']
            reliable = hist_blind['vel_reliable'][i]

            step_errors = []
            for k, p in enumerate(gp):
                future_idx = i + k + 1
                if future_idx >= len(pos): break

                actual_rel = pos[future_idx] - pos[i]
                pred_rel_x = p['px']
                pred_rel_y = p['py']
                if reliable:
                    pred_rel_z = p['pz']
                else:
                    pred_rel_z = p['pz'] - start_z

                pred_rel = np.array([pred_rel_x, pred_rel_y, pred_rel_z])
                dist = np.linalg.norm(pred_rel - actual_rel)
                step_errors.append(dist)

            if step_errors:
                pred_errors.append(np.mean(step_errors))

        if pred_errors:
            avg_pred_err = np.mean(pred_errors)
            max_pred_err = np.max(pred_errors)
            print(f"Mean Prediction Error: {avg_pred_err:.4f}m")
            print(f"Max Prediction Error: {max_pred_err:.4f}m")
        else:
            print("No prediction data available.")
