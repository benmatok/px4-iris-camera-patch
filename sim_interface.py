import numpy as np
import logging
import cv2
from drone_env.drone import DroneEnv

logger = logging.getLogger(__name__)

class SimDroneInterface:
    def __init__(self, projector):
        self.projector = projector
        try:
            self.env = DroneEnv(num_agents=1, episode_length=100000)
            self.env.reset_all_envs()
            self.dd = self.env.data_dictionary

            # Init State
            self.dd['pos_x'][0] = 0.0
            self.dd['pos_y'][0] = 0.0
            self.dd['pos_z'][0] = 1.0 # 1m Up (Sim Z is Up)

            self.masses = self.dd['masses']
            self.masses[0] = 3.33
            self.thrust_coeffs = self.dd['thrust_coeffs']
            self.thrust_coeffs[0] = 2.725 # Matches 54.5 total
            logger.info("DroneEnv initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize DroneEnv: {e}")
            raise

    def reset_to_scenario(self, name):
        if name == "Blind Dive":
            # High Altitude Dive
            self.dd['pos_x'][0] = 0.0
            self.dd['pos_y'][0] = 0.0
            self.dd['pos_z'][0] = 100.0

            # Zero Velocities
            self.dd['vel_x'][0] = 0.0
            self.dd['vel_y'][0] = 0.0
            self.dd['vel_z'][0] = 0.0

            # Zero Attitude
            self.dd['roll'][0] = 0.0
            self.dd['pitch'][0] = 0.0
            self.dd['yaw'][0] = 0.0

            # Zero Angular Velocities
            self.dd['ang_vel_x'][0] = 0.0
            self.dd['ang_vel_y'][0] = 0.0
            self.dd['ang_vel_z'][0] = 0.0

            logger.info(f"Reset to Scenario: {name}")
        else:
            logger.warning(f"Unknown Scenario: {name}")

    def step(self, action):
        # Action: [thrust, roll_rate, pitch_rate, yaw_rate]
        self.dd['actions'][:] = action.astype(np.float32)

        kwargs = self.env.get_step_function_kwargs()
        args = {}
        for k, v in kwargs.items():
            if v in self.dd:
                args[k] = self.dd[v]
            elif k == "num_agents":
                args[k] = self.env.num_agents
            elif k == "episode_length":
                args[k] = self.env.episode_length
            else:
                pass

        self.env.step_function(**args)
        self.dd['done_flags'][:] = 0.0 # Prevent auto-reset logic from interfering

    def get_state(self):
        # Return Sim State (Z-Up, Rads)
        return {
            'px': float(self.dd['pos_x'][0]),
            'py': float(self.dd['pos_y'][0]),
            'pz': float(self.dd['pos_z'][0]),
            'vx': float(self.dd['vel_x'][0]),
            'vy': float(self.dd['vel_y'][0]),
            'vz': float(self.dd['vel_z'][0]),
            'roll': float(self.dd['roll'][0]),
            'pitch': float(self.dd['pitch'][0]),
            'yaw': float(self.dd['yaw'][0]),
            'wx': float(self.dd['ang_vel_x'][0]),
            'wy': float(self.dd['ang_vel_y'][0]),
            'wz': float(self.dd['ang_vel_z'][0])
        }

    def get_image(self, target_pos_world):
        # Synthetic Vision
        width = 640
        height = 480
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Sim State to NED for Projector
        s = self.get_state()
        drone_state_ned = {
            'px': s['px'],
            'py': s['py'],
            'pz': -s['pz'],
            'roll': s['roll'],
            'pitch': s['pitch'],
            'yaw': -s['yaw']
        }

        tx, ty, tz = target_pos_world
        uv = self.projector.world_to_pixel(tx, ty, tz, drone_state_ned)

        if uv:
            u, v = uv
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), -1)

        return img
