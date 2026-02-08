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

        # Velocity Estimation
        self.estimated_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Predictor model (Nominal)
        cfg = self.models_config[0]
        self.predictor = PyGhostModel(cfg['mass'], cfg['drag_coeff'], cfg['thrust_coeff'])
        self.last_target_rel_pos = None

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.estimated_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_target_rel_pos = None
        logger.info("DPCFlightController reset")

    def compute_action(self, state_obs, target_cmd, raw_target_rel_ned=None, extra_yaw_rate=0.0):
        """
        Computes the optimal control action using DPC with Relative State Estimation.

        Args:
            state_obs: dict (pz, vz, roll, pitch, yaw, wx, wy, wz) - No px, py, vx, vy
            target_cmd: list [rel_x, rel_y, abs_z] (Command Target)
            raw_target_rel_ned: list [dx, dy, dz] (Measured Relative Target Position in NED) or None
            extra_yaw_rate: float (additional yaw rate for scanning)

        Returns:
            action_dict: dict (thrust, roll_rate, pitch_rate, yaw_rate)
            ghost_paths: list of trajectory lists
        """

        # 1. Prediction Step (Velocity)
        # Construct state for predictor using previous velocity estimate
        pred_state_in = {
            'px': 0.0, 'py': 0.0, 'pz': state_obs.get('pz', 0.0),
            'vx': self.estimated_velocity[0],
            'vy': self.estimated_velocity[1],
            'vz': state_obs.get('vz', self.estimated_velocity[2]), # Use measured Vz if available
            'roll': state_obs['roll'],
            'pitch': state_obs['pitch'],
            'yaw': state_obs['yaw'],
            'wx': state_obs.get('wx', 0.0),
            'wy': state_obs.get('wy', 0.0),
            'wz': state_obs.get('wz', 0.0)
        }

        # Predict next state to get velocity change
        predicted_state_next = self.predictor.step(pred_state_in, self.last_action, self.dt)
        v_pred = np.array([predicted_state_next['vx'], predicted_state_next['vy'], predicted_state_next['vz']])

        # 2. Measurement Update (Velocity from Optical Flow)
        v_meas = None
        if raw_target_rel_ned is not None and self.last_target_rel_pos is not None:
             # Calculate derivative of relative position
             curr_rel = np.array(raw_target_rel_ned)
             prev_rel = np.array(self.last_target_rel_pos)
             d_rel = (curr_rel - prev_rel) / self.dt

             # V_meas_ned = -d_rel (assuming static target)
             v_meas_ned = -d_rel

             # Convert NED Velocity to Sim Frame (X-North, Y-East, Z-Up)
             # NED: X-North, Y-East, Z-Down
             v_meas = np.array([v_meas_ned[0], v_meas_ned[1], -v_meas_ned[2]])

        # Update History
        self.last_target_rel_pos = raw_target_rel_ned

        # 3. Fuse Velocity
        alpha = 0.1 # Measurement weight
        if v_meas is not None:
            # Simple Complementary Filter
            v_est = (1.0 - alpha) * v_pred + alpha * v_meas
        else:
            v_est = v_pred

        self.estimated_velocity = v_est

        # 4. Construct Solver State (Relative Frame)
        solver_state = {
            'px': 0.0, 'py': 0.0, 'pz': state_obs['pz'],
            'vx': v_est[0], 'vy': v_est[1], 'vz': v_est[2],
            'roll': state_obs['roll'],
            'pitch': state_obs['pitch'],
            'yaw': state_obs['yaw'],
            'wx': state_obs.get('wx', 0.0),
            'wy': state_obs.get('wy', 0.0),
            'wz': state_obs.get('wz', 0.0)
        }

        # 5. Construct Solver Target
        # target_cmd is [RelX, RelY, AbsZ]
        # Solver Frame: Drone at (0,0,pz).
        # Target X (Abs) = DroneX + RelX = 0 + RelX
        # Target Y (Abs) = DroneY + RelY = 0 + RelY
        # Target Z (Abs) = AbsZ
        solver_target = [
            target_cmd[0],
            target_cmd[1],
            target_cmd[2]
        ]

        # Solve
        action_out = self.solver.solve(
            solver_state,
            solver_target,
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
        ghost_paths = self.rollout_ghosts(solver_state, action_out)

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
