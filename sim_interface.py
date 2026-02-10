import numpy as np
import logging
import cv2
from ghost_dpc.ghost_dpc import PyGhostModel

logger = logging.getLogger(__name__)

class SimDroneInterface:
    def __init__(self, projector):
        self.projector = projector
        # Replace DroneEnv with PyGhostModel for binary exactness with planner
        # Default Params: Mass=1.0, Drag=0.1, Thrust=1.0, Tau=0.1 (Ideal)
        self.mass = 1.0
        self.drag_coeff = 0.1
        self.thrust_coeff = 1.0
        self.tau = 0.1

        self.model = PyGhostModel(
            mass=self.mass,
            drag=self.drag_coeff,
            thrust_coeff=self.thrust_coeff,
            tau=self.tau,
            wind_x=0.0,
            wind_y=0.0
        )

        # Initialize State Dictionary (Sim Frame Z-Up)
        self.state = {
            'px': 0.0, 'py': 0.0, 'pz': 1.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'wx': 0.0, 'wy': 0.0, 'wz': 0.0
        }

        # Legacy attributes for compatibility (if needed)
        self.masses = [self.mass]
        self.thrust_coeffs = [self.thrust_coeff]
        self.dd = {'drag_coeffs': [self.drag_coeff]}

        logger.info("SimDroneInterface initialized with PyGhostModel (Binary Exact Mode).")

    def reset_to_scenario(self, name, **kwargs):
        if name == "Blind Dive":
            # High Altitude Dive
            self.state['px'] = kwargs.get('pos_x', 0.0)
            self.state['py'] = kwargs.get('pos_y', 0.0)
            self.state['pz'] = kwargs.get('pos_z', 100.0)

            # Zero Velocities
            self.state['vx'] = 0.0
            self.state['vy'] = 0.0
            self.state['vz'] = 0.0

            # Zero Attitude
            self.state['roll'] = 0.0
            self.state['pitch'] = kwargs.get('pitch', 0.0)
            self.state['yaw'] = kwargs.get('yaw', 0.0)

            # Zero Angular Velocities
            self.state['wx'] = 0.0
            self.state['wy'] = 0.0
            self.state['wz'] = 0.0

            logger.info(f"Reset to Scenario: {name}")
        else:
            logger.warning(f"Unknown Scenario: {name}")

    def step(self, action):
        # Action: [thrust, roll_rate, pitch_rate, yaw_rate]
        action_dict = {
            'thrust': float(action[0]),
            'roll_rate': float(action[1]),
            'pitch_rate': float(action[2]),
            'yaw_rate': float(action[3])
        }

        dt = 0.05 # Fixed Time Step matching Planner

        # Binary Exact Step using the same code as planner
        self.state = self.model.step(self.state, action_dict, dt)

    def get_state(self):
        # Return State (Copy)
        return self.state.copy()

    def get_image(self, target_pos_world):
        # Synthetic Vision using current state
        width = 640
        height = 480
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Sim State to NED for Projector
        s = self.state
        drone_state_ned = {
            'px': s['px'],
            'py': s['py'],
            'pz': -s['pz'],
            'roll': s['roll'],
            'pitch': s['pitch'],
            'yaw': s['yaw']
        }

        tx, ty, tz = target_pos_world
        uv = self.projector.world_to_pixel(tx, ty, tz, drone_state_ned)

        if uv:
            u, v = uv
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), -1)

        return img
