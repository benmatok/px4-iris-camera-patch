import sys
import os
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.projection import Projector
from sim_interface import SimDroneInterface
from visual_tracker import VisualTracker
from flight_controller import DPCFlightController
from mission_manager import MissionManager
from vision.flow_estimator import FlowVelocityEstimator

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DT = 0.05

class ControllerComparator:
    def __init__(self, mode='PID'):
        self.mode = mode

        # Setup similar to theshow.py / validate_dive_tracking.py
        self.projector = Projector(width=640, height=480, fov_deg=120.0, tilt_deg=30.0)
        self.sim = SimDroneInterface(self.projector)
        self.target_pos = [50.0, 0.0, 0.0]

        # Start high
        drone_pos = [0.0, 0.0, 50.0]
        pitch, yaw = self.calculate_initial_orientation(drone_pos, self.target_pos)

        self.sim.reset_to_scenario("Blind Dive", pos_x=drone_pos[0], pos_y=drone_pos[1], pos_z=drone_pos[2], pitch=pitch, yaw=yaw)

        self.tracker = VisualTracker(self.projector)
        self.flow_estimator = FlowVelocityEstimator(self.projector)
        self.mission = MissionManager(target_alt=50.0) # Start in TAKEOFF then SCAN/HOMING

        # Init Controller in specified mode
        self.controller = DPCFlightController(dt=DT, mode=mode)

    def calculate_initial_orientation(self, drone_pos, target_pos):
        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]
        dz = target_pos[2] - drone_pos[2]
        yaw = np.arctan2(dy, dx)
        dist_xy = np.sqrt(dx*dx + dy*dy)
        pitch_vec = np.arctan2(dz, dist_xy)
        camera_tilt = np.deg2rad(30.0)
        pitch = pitch_vec - camera_tilt
        return pitch, yaw

    def sim_to_ned(self, sim_state):
        ned = sim_state.copy()
        ned['px'] = sim_state['py']
        ned['py'] = sim_state['px']
        ned['pz'] = -sim_state['pz']
        ned['vx'] = sim_state['vy']
        ned['vy'] = sim_state['vx']
        ned['vz'] = -sim_state['vz']
        ned['roll'] = sim_state['roll']
        ned['pitch'] = sim_state['pitch']
        ned['yaw'] = (np.pi / 2.0) - sim_state['yaw']
        ned['yaw'] = (ned['yaw'] + np.pi) % (2 * np.pi) - np.pi
        return ned

    def sim_pos_to_ned_pos(self, sim_pos):
        return [sim_pos[1], sim_pos[0], -sim_pos[2]]

    def ned_rel_to_sim_rel(self, ned_rel):
        return [ned_rel[1], ned_rel[0], -ned_rel[2]]

    def run(self, duration=10.0):
        steps = int(duration / DT)
        min_dist = 1000.0
        success = False

        logger.info(f"Running Simulation with {self.mode} Controller...")

        for i in range(steps):
            # 1. Sim State
            s = self.sim.get_state()

            # 2. Perception
            img = self.sim.get_image(self.target_pos)
            dpc_state_ned = self.sim_to_ned(s)
            target_pos_ned = self.sim_pos_to_ned_pos(self.target_pos)

            # Use Ground Truth for robust tracking comparison (isolate control performance)
            center, target_wp_ned, radius = self.tracker.process(
                img, dpc_state_ned, ground_truth_target_pos=target_pos_ned
            )

            # Flow / FOE
            body_rates = (s['wx'], s['wy'], s['wz'])
            foe = self.flow_estimator.update(dpc_state_ned, body_rates, DT)

            target_wp_sim = None
            if target_wp_ned:
                target_wp_sim = self.ned_rel_to_sim_rel(target_wp_ned)

            tracking_norm = None
            if center:
                tracking_norm = self.projector.pixel_to_normalized(center[0], center[1])

            # 3. Mission
            sim_state_rel = s.copy(); sim_state_rel['px']=0; sim_state_rel['py']=0
            mission_state, dpc_target, extra_yaw = self.mission.update(sim_state_rel, (center, target_wp_sim))

            # 4. Control
            state_obs = {
                'pz': s['pz'], 'vz': s['vz'], 'roll': s['roll'], 'pitch': s['pitch'], 'yaw': s['yaw'],
                'wx': s['wx'], 'wy': s['wy'], 'wz': s['wz'],
                # Add extra state if needed for GDPC init? compute_action_gdpc reads vx, vy.
                'vx': s['vx'], 'vy': s['vy']
            }

            action, _ = self.controller.compute_action(
                state_obs, dpc_target, tracking_uv=tracking_norm, extra_yaw_rate=extra_yaw, foe_uv=foe
            )

            # 5. Apply
            sim_action = np.array([action['thrust'], action['roll_rate'], action['pitch_rate'], action['yaw_rate']])
            if mission_state == "TAKEOFF" and s['pz'] < 2.0: sim_action[0] = 0.8
            self.sim.step(sim_action)

            # Check Dist
            dist = np.sqrt((s['px'] - self.target_pos[0])**2 + (s['py'] - self.target_pos[1])**2 + (s['pz'] - self.target_pos[2])**2)
            if dist < min_dist: min_dist = dist

            if dist < 2.0:
                success = True
                break
            if s['pz'] < 0.2:
                break

        return min_dist, success

if __name__ == "__main__":
    # PID Run
    runner_pid = ControllerComparator(mode='PID')
    dist_pid, success_pid = runner_pid.run(duration=15.0)

    # GDPC Run
    runner_gdpc = ControllerComparator(mode='GDPC')
    dist_gdpc, success_gdpc = runner_gdpc.run(duration=15.0)

    print("\n" + "="*40)
    print(f"PID Controller: Min Dist = {dist_pid:.2f}m, Success = {success_pid}")
    print(f"GDPC Controller: Min Dist = {dist_gdpc:.2f}m, Success = {success_gdpc}")
    print("="*40)

    if success_gdpc and not success_pid:
        print("WINNER: GDPC")
    elif success_pid and not success_gdpc:
        print("WINNER: PID")
    elif dist_gdpc < dist_pid:
        print("WINNER: GDPC (Closer approach)")
    else:
        print("WINNER: PID (Closer approach)")
    print("="*40 + "\n")
