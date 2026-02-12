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
from flight_controller import DPCFlightController
from mission_manager import MissionManager

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DT = 0.05

class DiveValidator:
    def __init__(self, use_ground_truth=True):
        self.use_ground_truth = use_ground_truth

        # Tilt 30.0 (Up) as in theshow.py
        self.projector = Projector(width=640, height=480, fov_deg=120.0, tilt_deg=30.0)

        # Scenario / Sim
        self.sim = SimDroneInterface(self.projector)
        self.target_pos_sim_world = [50.0, 0.0, 0.0] # Blind Dive Target

        # Initial Pos for Blind Dive (Adjusted to avoid Gimbal Lock at -90 deg pitch)
        drone_pos = [0.0, 0.0, 50.0]
        pitch, yaw = self.calculate_initial_orientation(drone_pos, self.target_pos_sim_world)

        self.sim.reset_to_scenario("Blind Dive", pos_x=drone_pos[0], pos_y=drone_pos[1], pos_z=drone_pos[2], pitch=pitch, yaw=yaw)

        # Perception
        self.tracker = VisualTracker(self.projector)

        # Logic
        self.mission = MissionManager(target_alt=50.0) # Start with target alt matching start

        # Control
        self.controller = DPCFlightController(dt=DT)

    def calculate_initial_orientation(self, drone_pos, target_pos):
        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]
        dz = target_pos[2] - drone_pos[2]

        yaw = np.arctan2(dy, dx)
        dist_xy = np.sqrt(dx*dx + dy*dy)
        pitch_vec = np.arctan2(dz, dist_xy)

        # Camera Tilt is 30.0 (Up)
        camera_tilt = np.deg2rad(30.0)

        # PyGhostModel Convention: Positive Pitch = Nose Down.
        # Standard Math (atan2): Positive Angle = Up.
        # Target Angle (Std) = Body Angle (Std) + Camera Tilt (Std)
        # Body Angle (Std) = Target Angle (Std) - Camera Tilt (Std)
        # Model Pitch = - Body Angle (Std) = Camera Tilt (Std) - Target Angle (Std)

        pitch = camera_tilt - pitch_vec

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
        # Sim: Positive Pitch = Nose Down. NED: Positive Pitch = Nose Up.
        ned['pitch'] = -sim_state['pitch']
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
            'state': []
        }

        logger.info(f"Running Validation (Ground Truth: {self.use_ground_truth}) for {duration}s...")

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
                'pz': s['pz'],
                'vz': s['vz'],
                'roll': s['roll'],
                'pitch': s['pitch'],
                'yaw': s['yaw'],
                'wx': s['wx'],
                'wy': s['wy'],
                'wz': s['wz']
            }

            action_out, _ = self.controller.compute_action(
                state_obs,
                dpc_target,
                tracking_uv=center,
                extra_yaw_rate=extra_yaw
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

            # Logging
            history['t'].append(t)
            history['drone_pos'].append([s['px'], s['py'], s['pz']])
            if i % 20 == 0:
                print(f"T={t:.2f} State={mission_state} Dist={dist:.2f} Pos=({s['px']:.1f}, {s['py']:.1f}, {s['pz']:.1f})")

            # Estimate Target Pos (Absolute Sim Frame)
            if target_wp_sim:
                est_x = s['px'] + target_wp_sim[0]
                est_y = s['py'] + target_wp_sim[1]
                est_z = s['pz'] + target_wp_sim[2] # dpc_target[2] is Absolute Z, but target_wp_sim is Relative
                # Wait, mission_manager uses target_wp (Relative) to set dpc_target (Relative XY, Absolute Z?)
                # MissionManager: dpc_target = [target_wp[0], target_wp[1], 2.0] -> 2.0 is Abs Z.
                # Actually, target_wp from tracker is [RelX, RelY, RelZ].
                # So Est Target Pos = Drone Pos + Rel Target Pos
                # Let's verify `target_wp_sim` components.
                # `ned_rel_to_sim_rel`: [dy, dx, -dz].
                # NED Rel: Target - Drone.
                # NED Rel Z: Target Z (Down) - Drone Z (Down).
                # Sim Rel Z = - (Target Z_ned - Drone Z_ned) = Drone Z_ned - Target Z_ned
                # Drone Z_ned = -Drone Z_sim. Target Z_ned = -Target Z_sim.
                # Sim Rel Z = (-Drone Z_sim) - (-Target Z_sim) = Target Z_sim - Drone Z_sim.
                # Correct.

                # So Est Target Absolute = Drone Pos + Target Rel Sim
                est_x = s['px'] + target_wp_sim[0]
                est_y = s['py'] + target_wp_sim[1]
                est_z = s['pz'] + target_wp_sim[2]
                history['target_est'].append([est_x, est_y, est_z])
            else:
                history['target_est'].append([None, None, None])

        return history

def plot_results(hist_gt, hist_vis, filename="validation_dive_tracking.png"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Trajectory Side View (X-Z)
    axs[0, 0].set_title("Trajectory Side View (X-Z)")

    # Ground Truth Run
    pos_gt = np.array(hist_gt['drone_pos'])
    axs[0, 0].plot(pos_gt[:, 0], pos_gt[:, 2], 'b-', label='GT Tracking')

    # Vision Run
    pos_vis = np.array(hist_vis['drone_pos'])
    axs[0, 0].plot(pos_vis[:, 0], pos_vis[:, 2], 'g--', label='Vision Tracking')

    # Target
    axs[0, 0].plot(50.0, 0.0, 'rx', markersize=10, label='Target')

    axs[0, 0].set_xlabel("X (m)")
    axs[0, 0].set_ylabel("Z (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Distance to Target
    axs[0, 1].set_title("Distance to Target")
    axs[0, 1].plot(hist_gt['t'], hist_gt['dist'], 'b-', label='GT Tracking')
    axs[0, 1].plot(hist_vis['t'], hist_vis['dist'], 'g--', label='Vision Tracking')
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Distance (m)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Target Estimation Error (Vision Run Only)
    axs[1, 0].set_title("Target Estimation Error (Vision Run)")
    t = np.array(hist_vis['t'])
    est = hist_vis['target_est']
    errs = []
    valid_t = []

    target_true = np.array([50.0, 0.0, 0.0])

    for i, p in enumerate(est):
        if p[0] is not None:
            e = np.linalg.norm(np.array(p) - target_true)
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
    axs[1, 1].set_title("Mission State (Vision Run)")
    states = hist_vis['state']
    # Map states to ints
    state_map = {"TAKEOFF": 0, "SCAN": 1, "HOMING": 2}
    y_vals = [state_map.get(s, -1) for s in states]
    axs[1, 1].plot(hist_vis['t'], y_vals, 'k-')
    axs[1, 1].set_yticks([0, 1, 2])
    axs[1, 1].set_yticklabels(["TAKEOFF", "SCAN", "HOMING"])
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    # Run with Ground Truth (Web Parity)
    validator_gt = DiveValidator(use_ground_truth=True)
    hist_gt = validator_gt.run(duration=15.0)

    # Run with Vision (Detection)
    validator_vis = DiveValidator(use_ground_truth=False)
    hist_vis = validator_vis.run(duration=15.0)

    plot_results(hist_gt, hist_vis)

    # Validation Checks
    final_dist_gt = hist_gt['dist'][-1]
    final_dist_vis = hist_vis['dist'][-1]

    print(f"Final Distance (GT Tracking): {final_dist_gt:.2f}m")
    print(f"Final Distance (Vision Tracking): {final_dist_vis:.2f}m")

    if final_dist_gt < 5.0:
        print("SUCCESS: GT Tracking Dive Successful")
    else:
        print("FAILURE: GT Tracking Dive Failed")

    if final_dist_vis < 5.0:
        print("SUCCESS: Vision Tracking Dive Successful")
    else:
        print("FAILURE: Vision Tracking Dive Failed")
