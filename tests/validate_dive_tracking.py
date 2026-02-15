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
    def __init__(self, use_ground_truth=True, use_blind_mode=False, init_alt=50.0, init_dist=150.0):
        self.use_ground_truth = use_ground_truth
        self.use_blind_mode = use_blind_mode

        # Tilt 30.0 (Up) as in theshow.py
        self.projector = Projector(width=640, height=480, fov_deg=120.0, tilt_deg=30.0)

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

        # Logic
        self.mission = MissionManager(target_alt=init_alt, enable_staircase=False)

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
        # But flight_controller uses Positive Pitch = Nose Up.
        # We need to set initial pitch in Sim convention?
        # Let's try to match target.
        # Vector is ~ -18 deg (Down). Camera is +30 deg (Up).
        # We need Body = -48 deg (Nose Up).
        # If Sim Pitch is + = Nose Down, then -48 is -48.
        # Wait, if Sim Pitch is - = Nose Up.
        # Then we need -48.
        # Let's try positive.
        # CORRECTION: pitch_vec is Angle to Target (Negative for Down).
        # Camera is Tilted Down (+30 deg relative to body).
        # To point camera at Target: BodyPitch + CameraTilt = TargetPitch
        # BodyPitch = TargetPitch - CameraTilt ?
        # Example: Target -60. Camera +30. Body = -90 (Nose Down). Correct.
        # Wait, if Body is -90 (Nose Straight Down), Camera (-90+30)=-60. Correct.
        # So Body = Target - Tilt is correct?
        # My previous test showed Scenario 1 flying backwards.
        # Scenario 1: Alt 100, Dist 50. Target = -63 deg.
        # Body = -63 - 30 = -93 deg.
        # -93 deg is "Beyond Vertical" (Upside down/Backwards).
        # If we want to fly forward, we need Body to be roughly -30 to -60.
        # Camera Tilt in Projector(tilt_deg=30) usually means "Up" relative to standard drone -Z?
        # Or usually drone cameras are tilted UP to see forward when pitched down?
        # "Web interface ... rotated 30 degrees up (pitch) to match the camera mounting".
        # Memory says: "rotated 30 degrees up".
        # If Camera is tilted UP (+30).
        # BodyPitch + 30 = TargetPitch.
        # BodyPitch = TargetPitch - 30.
        # -63 - 30 = -93. Still -93.
        #
        # Let's try the other way. Maybe Pitch Sign convention is different.
        # If Pitch Positive is Nose Up. Target (-63) is Down.
        # If Camera is tilted UP 30 deg.
        # We need to Pitch Down MORE to compensate for Up tilt.
        # So -93 makes sense? But -93 is pointing backwards?
        #
        # Let's try assuming Camera is tilted DOWN (common for surveys).
        # "rotated 30 degrees up ... to match camera". This implies camera is UP.
        #
        # Let's try just setting it to point roughly at target.
        # pitch_vec (-63).
        # Let's try Body = -30.
        # -30 + 30 (Cam) = 0 (Level). No.
        # -30 + (-30)?
        #
        # Let's try: pitch = pitch_vec + camera_tilt
        # -63 + 30 = -33.
        # If Body is -33 (Nose Down 33). Camera is Up 30? Net -3.
        # Target is -63. Gap 60 deg.
        #
        # Let's try ignoring "Camera Tilt logic" and just pointing the DRONE at the target.
        # pitch = pitch_vec.
        # Then UV will be off center but maybe in FOV.
        #
        # Trial: Just point body at target. Camera (Offset < FOV/2) should see it.
        pitch = pitch_vec

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
            'state': []
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

            # Convert Result (target_wp_ned) back to Sim Relative Frame
            target_wp_sim = None
            if target_wp_ned:
                 target_wp_sim = self.ned_rel_to_sim_rel(target_wp_ned)

            # 3. Update Mission Logic
            sim_state_rel = s.copy()
            sim_state_rel['px'] = 0.0
            sim_state_rel['py'] = 0.0

            mission_state, dpc_target, extra_yaw = self.mission.update(sim_state_rel, (center, target_wp_sim))

            # FORCE TARGET IN BLIND MODE TEST:
            # If we are testing Blind Mode flight control, we don't want MissionManager to abort to SCAN
            # just because the camera lost the target. We want to verify the BLIND HOMING capability.
            # So we override dpc_target with Ground Truth relative vector.
            if self.use_blind_mode:
                 # Calculate relative target vector (Sim Frame)
                 dx = self.target_pos_sim_world[0] - s['px']
                 dy = self.target_pos_sim_world[1] - s['py']
                 dz = self.target_pos_sim_world[2] - s['pz']

                 # Convert to NED for Controller (dpc_target is NED relative)
                 # Sim X (East) -> NED Y
                 # Sim Y (North) -> NED X
                 # Sim Z (Up) -> NED -Z

                 # flight_controller expects [dx_rel, dy_rel, Z_ABS].
                 # Target Absolute Z (NED) = -Target_Sim_Z = 0.0.
                 dpc_target_override = [dy, dx, -self.target_pos_sim_world[2]]

                 # Only override if MissionManager isn't giving us a valid homing target
                 # Or always override to ensure we test the flight controller, not the mission manager.
                 dpc_target = dpc_target_override

                 # Also force state to HOMING for logging clarity (optional)
                 mission_state = "HOMING"

            # 4. Compute Control
            # Convert Sim State to NED for Controller (as theshow.py does)
            s_ned = self.sim_to_ned(s)

            state_obs = {
                'px': s_ned['px'],
                'py': s_ned['py'],
                'pz': s_ned['pz'],
                'vx': s_ned['vx'] if not self.use_blind_mode else 0.0,
                'vy': s_ned['vy'] if not self.use_blind_mode else 0.0,
                'vz': s_ned['vz'] if not self.use_blind_mode else None,
                'roll': s_ned['roll'],
                'pitch': s_ned['pitch'],
                'yaw': s_ned['yaw'],
                'wx': s['wx'], # Rates are body frame usually? Or NED? sim_to_ned doesn't convert rates.
                'wy': s['wy'], # Assuming Body Rates are same.
                'wz': s['wz']
            }

            tracking_norm = None
            if center:
                tracking_norm = self.projector.pixel_to_normalized(center[0], center[1])

            # Normalize Radius (screen width fraction?)
            # Projector width=640. Radius is in pixels.
            # Normalize by width/2 (center to edge)
            tracking_rad_norm = None
            if radius:
                tracking_rad_norm = radius / (self.projector.width / 2.0)

            action_out, _ = self.controller.compute_action(
                state_obs,
                dpc_target,
                tracking_uv=tracking_norm,
                extra_yaw_rate=extra_yaw,
                tracking_radius=tracking_rad_norm
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

def plot_results(hist_gt, hist_vis, hist_blind=None, filename="validation_dive_tracking.png"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Trajectory Side View (X-Z)
    axs[0, 0].set_title("Trajectory Side View (X-Z)")

    # Ground Truth Run
    pos_gt = np.array(hist_gt['drone_pos'])
    axs[0, 0].plot(pos_gt[:, 0], pos_gt[:, 2], 'b-', label='Full State (GT Tracking)')

    # Vision Run
    pos_vis = np.array(hist_vis['drone_pos'])
    axs[0, 0].plot(pos_vis[:, 0], pos_vis[:, 2], 'g--', label='Full State (Vision Tracking)')

    # Blind Run (Web App)
    if hist_blind:
        pos_blind = np.array(hist_blind['drone_pos'])
        axs[0, 0].plot(pos_blind[:, 0], pos_blind[:, 2], 'r:', linewidth=2, label='Blind Mode (Web App)')

    # Target
    axs[0, 0].plot(150.0, 0.0, 'rx', markersize=10, label='Target') # 150m target

    axs[0, 0].set_xlabel("X (m)")
    axs[0, 0].set_ylabel("Z (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Distance to Target
    axs[0, 1].set_title("Distance to Target")
    axs[0, 1].plot(hist_gt['t'], hist_gt['dist'], 'b-', label='Full GT')
    axs[0, 1].plot(hist_vis['t'], hist_vis['dist'], 'g--', label='Full Vision')
    if hist_blind:
        axs[0, 1].plot(hist_blind['t'], hist_blind['dist'], 'r:', label='Blind Web')
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

    target_true = np.array([150.0, 0.0, 0.0]) # 150m target

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
    state_map = {"TAKEOFF": 0, "SCAN": 1, "HOMING": 2, "STAIRCASE_DESCEND": 3, "STAIRCASE_STABILIZE": 4}
    y_vals = [state_map.get(s, -1) for s in states]
    axs[1, 1].plot(hist_vis['t'], y_vals, 'k-')
    axs[1, 1].set_yticks([0, 1, 2, 3, 4])
    axs[1, 1].set_yticklabels(["TAKEOFF", "SCAN", "HOMING", "DESCEND", "STABILIZE"])
    axs[1, 1].set_xlabel("Time (s)")
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

    plot_results(hist_gt, hist_vis, hist_blind)

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
