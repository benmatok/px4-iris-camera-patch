import numpy as np
import logging
from ghost_dpc.ghost_dpc import PyDPCSolver, PyGhostModel

logger = logging.getLogger(__name__)

class DPCFlightController:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.solver = PyDPCSolver()
        # thrust_coeff=2.725 matches sim_interface.py (2.725 * 20.0 = 54.5N Total Thrust)
        self.models_config = [{'mass': 3.33, 'drag_coeff': 0.3, 'thrust_coeff': 2.725}]
        self.weights = [1.0]
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

    def compute_action(self, state_ned, target_ned, extra_yaw_rate=0.0):
        """
        Computes the optimal control action using DPC.

        Args:
            state_ned: dict (px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz)
            target_ned: list [x, y, z]
            extra_yaw_rate: float (additional yaw rate for scanning)

        Returns:
            action_dict: dict (thrust, roll_rate, pitch_rate, yaw_rate)
            ghost_paths: list of trajectory lists
        """

        # Solve
        action_out = self.solver.solve(
            state_ned,
            target_ned,
            self.last_action,
            self.models_config,
            self.weights,
            self.dt
        )
        self.last_action = action_out

        # Apply extra yaw rate (e.g. for scanning)
        if extra_yaw_rate != 0.0:
            action_out['yaw_rate'] = extra_yaw_rate

        # Rollout Ghosts for Visualization
        ghost_paths = self.rollout_ghosts(state_ned, action_out)

        return action_out, ghost_paths

    def rollout_ghosts(self, start_state, action_dict, horizon=10):
        ghosts = []
        for cfg in self.models_config:
            model = PyGhostModel(cfg['mass'], cfg['drag_coeff'], cfg['thrust_coeff'])
            path = []
            curr = start_state.copy()

            for _ in range(horizon):
                next_s = model.step(curr, action_dict, self.dt)
                path.append(next_s)
                curr = next_s
            ghosts.append(path)
        return ghosts
