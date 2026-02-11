import numpy as np
import logging
from collections import deque
from ghost_dpc.ghost_dpc import PyDPCSolver

logger = logging.getLogger(__name__)

class DPCFlightController:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.solver = PyDPCSolver()
        # Binary Exact Parity: Mass=1.0, Thrust=1.0, Drag=0.1, Tau=0.1
        self.models_config = [{'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0, 'tau': 0.1}]
        self.weights = [1.0]
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

        # History buffer for 4 seconds at 20Hz (dt=0.05) -> 80 steps
        self.history = deque(maxlen=80)

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.history.clear()
        logger.info("DPCFlightController reset")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0):
        """
        Computes the optimal control action using DPC with Relative State Estimation.

        Args:
            state_obs: dict (pz, vz, roll, pitch, yaw, wx, wy, wz) - No px, py, vx, vy
            target_cmd: list [rel_x, rel_y, abs_z] (Command Target). Only abs_z is used as goal_z.
            tracking_uv: tuple (u, v) or None (Tracking measurement)
            extra_yaw_rate: float (additional yaw rate for scanning)

        Returns:
            action_dict: dict (thrust, roll_rate, pitch_rate, yaw_rate)
            ghost_paths: list of trajectory lists
        """

        # 1. Update History
        # Observation Dict matching _estimate_relative_state expectation
        obs = {
            'time': 0.0, # Not strictly used yet but good practice
            'roll': state_obs['roll'],
            'pitch': state_obs['pitch'],
            'yaw': state_obs['yaw'],
            'pz': state_obs['pz'], # Altitude
            'vz': state_obs.get('vz', 0.0),
            'wx': state_obs.get('wx', 0.0),
            'wy': state_obs.get('wy', 0.0),
            'wz': state_obs.get('wz', 0.0),
            'thrust': self.last_action['thrust'],
            'roll_rate': self.last_action['roll_rate'],
            'pitch_rate': self.last_action['pitch_rate'],
            'yaw_rate': self.last_action['yaw_rate'],
            'u': tracking_uv[0] if tracking_uv else None,
            'v': tracking_uv[1] if tracking_uv else None
        }
        self.history.append(obs)

        # 2. Extract Goal Z
        # target_cmd is [RelX, RelY, AbsZ]
        goal_z = target_cmd[2]

        # 3. Solve
        forced_yaw = extra_yaw_rate if extra_yaw_rate != 0.0 else None

        action_out, ghost_paths = self.solver.solve(
            list(self.history),
            self.last_action,
            self.models_config,
            self.weights,
            self.dt,
            forced_yaw_rate=forced_yaw,
            goal_z=goal_z
        )
        self.last_action = action_out

        return action_out, ghost_paths
