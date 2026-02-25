import numpy as np
import logging
import math
from flight_config import FlightConfig
from flight_controller_gdpc import GDPCOptimizer
from vision.spline_smoother import SplineSmoother
import time

logger = logging.getLogger(__name__)

class DPCFlightController:
    """
    Standard ENU (East-North-Up) Flight Controller.
    """
    def __init__(self, dt=0.01, mode='PID', config: FlightConfig = None):
        self.dt = dt
        self.mode = mode
        self.config = config or FlightConfig()

        # State Variables
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

        # Blind Mode Estimation
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0

        self.last_tracking_size = None
        self.last_tracking_v = None
        self.rer_smoothed = 0.0

        # Cruise / Dive State
        self.dive_initiated = False

        # Final Mode State
        self.final_mode = False
        self.locked_pitch = 0.0

        # GDPC
        self.gdpc = GDPCOptimizer(self.config)

        # Smoother
        # Minimal lag configuration
        self.smoother = SplineSmoother(window_size=3, smoothing_factor=0.01)
        self.start_time = time.time()

        # Camera Intrinsics (Approximate)
        # f = width / (2 * tan(fov/2))
        self.cam_w = self.config.camera.width
        self.cam_h = self.config.camera.height
        self.cam_cx = self.cam_w / 2.0
        self.cam_cy = self.cam_h / 2.0
        fov_rad = math.radians(self.config.camera.fov_deg)
        self.cam_fx = self.cam_w / (2.0 * math.tan(fov_rad / 2.0))
        self.cam_fy = self.cam_fx # Square pixels assumption

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.gdpc.reset()
        self.smoother.reset()
        self.start_time = time.time()
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0
        self.last_tracking_size = None
        self.last_tracking_v = None
        self.rer_smoothed = 0.0

        self.dive_initiated = False
        self.final_mode = False
        self.locked_pitch = 0.0
        logger.info(f"DPCFlightController reset.")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, tracking_size=None, extra_yaw_rate=0.0, foe_uv=None, velocity_est=None, position_est=None, velocity_reliable=False):
        # Unpack State
        obs_pz = state_obs.get('pz')
        obs_vz = state_obs.get('vz')
        roll = state_obs['roll']
        pitch = state_obs['pitch']
        yaw = state_obs['yaw']

        ctrl = self.config.control
        vis = self.config.vision

        # Blind Mode Logic (IMU/Baro Integration)
        if obs_pz is None:
            # Full Blind Integration
            th = self.last_action['thrust']
            g = 9.81
            T_max_accel = 20.0
            accel_z = (th * T_max_accel) * (math.cos(roll) * math.cos(pitch)) - g
            accel_z -= 0.1 * self.est_vz
            self.est_vz += accel_z * self.dt
            self.est_pz += self.est_vz * self.dt
            pz = self.est_pz
            vz = self.est_vz
        else:
            self.est_pz = obs_pz
            if obs_vz is None:
                # Estimate VZ
                th = self.last_action['thrust']
                g = 9.81
                T_max_accel = 20.0
                accel_z = (th * T_max_accel) * (math.cos(roll) * math.cos(pitch)) - g
                accel_z -= 0.1 * self.est_vz
                self.est_vz += accel_z * self.dt
            else:
                self.est_vz = obs_vz

            pz = self.est_pz
            vz = self.est_vz

        # --- Visual RER / TTC Calculation ---
        raw_rer = 0.0
        if tracking_size is not None and self.last_tracking_size is not None and self.last_tracking_size > 0.001:
            raw_rer = (tracking_size - self.last_tracking_size) / (self.last_tracking_size * self.dt)

        alpha_rer = vis.rer_smoothing_alpha
        self.rer_smoothed = (1.0 - alpha_rer) * self.rer_smoothed + alpha_rer * raw_rer
        self.last_tracking_size = tracking_size

        # Check velocity reliability against limit
        vel_mag = 0.0
        if velocity_est:
             vel_mag = math.sqrt(velocity_est['vx']**2 + velocity_est['vy']**2 + velocity_est['vz']**2)

        is_reliable = velocity_reliable
        if vel_mag > ctrl.velocity_limit:
             is_reliable = False

        # Use Raw VIO Velocity if available
        if velocity_est:
             vx_enu = velocity_est['vy']
             vy_enu = velocity_est['vx']
             vz_enu = -velocity_est['vz']

             alpha_vel = ctrl.velocity_smoothing_alpha
             self.est_vx = alpha_vel * vx_enu + (1-alpha_vel) * self.est_vx
             self.est_vy = alpha_vel * vy_enu + (1-alpha_vel) * self.est_vy
             self.est_vz = alpha_vel * vz_enu + (1-alpha_vel) * self.est_vz

        # 2. Tracking Control (Visual Servoing)
        pitch_rate_cmd = 0.0
        yaw_rate_cmd = 0.0
        thrust_track = 0.6

        # --- CONSTANT PITCH MODE (Sim Pitch Positive = Nose Up, Negative = Nose Down) ---
        target_pitch = ctrl.fixed_pitch # e.g. -0.10 rad (6 deg Nose Down)

        # P-Controller for Pitch RATE to hold Pitch ANGLE
        pitch_err = target_pitch - pitch

        # P-Gain for Holding
        # We need Negative Pitch Rate to Pitch Up (increase pitch value towards 0 or positive).
        # We need Positive Pitch Rate to Pitch Down (decrease pitch value towards negative).
        # Wait. Test 19: Cmd +1.0 -> Pitch -48 to -54 (More Negative).
        # So Positive Cmd = Pitch Down (Negative Change).
        # If Err = Target (-0.10) - Current (-0.84) = +0.74.
        # We want to go UP (Pitch Up).
        # Pitch Up needs Negative Cmd.
        # So Gain must be Negative. `cmd = -K * err`.
        pitch_rate_cmd = -ctrl.k_pitch_hold * pitch_err
        pitch_rate_cmd = max(-1.0, min(1.0, pitch_rate_cmd))

        if tracking_uv:
            u, v = tracking_uv
            v_dot = 0.0
            if self.last_tracking_v is not None:
                v_dot = (v - self.last_tracking_v) / self.dt
            self.last_tracking_v = v

            # Trigger Dive Logic
            if not self.dive_initiated:
                trigger_rer = self.rer_smoothed > ctrl.dive_trigger_rer
                trigger_v_pos = v > ctrl.dive_trigger_v_threshold

                # IMMEDIATE TRIGGER if track is solid and near center
                if abs(u - self.cam_cx) < 100 and abs(v - self.cam_cy) < 100:
                     logger.info("Target Centered - FORCING DIVE.")
                     self.dive_initiated = True

                if trigger_rer or trigger_v_pos:
                    logger.info(f"DIVE INITIATED! RER={self.rer_smoothed:.2f}, v={v:.2f}")
                    self.dive_initiated = True

            # FOE Estimation (Normalized)
            calc_foe_norm = None
            if foe_uv:
                 calc_foe_px = foe_uv
            elif velocity_est:
                cr = math.cos(roll); sr = math.sin(roll)
                cp = math.cos(pitch); sp = math.sin(pitch)
                cy = math.cos(yaw); sy = math.sin(yaw)
                R_b2w = np.array([
                    [r11 := cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr]
                ])
                v_w = np.array([self.est_vx, self.est_vy, self.est_vz])
                v_b = R_b2w.T @ v_w

                theta = np.radians(self.config.camera.tilt_deg)
                ct, st = math.cos(theta), math.sin(theta)
                R_tilt = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
                R_base = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
                R_c2b = R_tilt @ R_base
                v_c = R_c2b.T @ v_b

                if v_c[2] > 0.1:
                     u_foe_n = v_c[0] / v_c[2]
                     v_foe_n = v_c[1] / v_c[2]
                     calc_foe_norm = (u_foe_n, v_foe_n)

            calc_foe_px = None
            if calc_foe_norm:
                 u_px = calc_foe_norm[0] * self.cam_fx + self.cam_cx
                 v_px = calc_foe_norm[1] * self.cam_fy + self.cam_cy
                 calc_foe_px = (u_px, v_px)

            # --- YAW CONTROL (Steering) ---
            if calc_foe_px:
                 u_err = u - calc_foe_px[0]
                 u_err_norm = u_err / self.cam_fx
                 yaw_rate_cmd = +ctrl.k_yaw * u_err_norm
            else:
                 u_err_norm = (u - self.cam_cx) / self.cam_fx
                 yaw_rate_cmd = +ctrl.k_yaw * u_err_norm

            yaw_rate_cmd = max(-1.0, min(1.0, yaw_rate_cmd))

            # --- THRUST CONTROL (Vertical Tracking) ---
            if not self.dive_initiated:
                thrust_track = 0.55
            else:
                # Use v_err to modulate thrust
                # If target is low (v > center), we are too high -> Decrease Thrust (Sink)
                # If target is high (v < center), we are too low -> Increase Thrust (Climb/Loft)

                v_err = v - self.cam_cy
                v_err_norm = v_err / self.cam_fy

                # Thrust Base
                thrust_base = 0.55

                # Correction
                # v_err_norm > 0 (Target Low) -> Want Less Thrust.
                # v_err_norm < 0 (Target High) -> Want More Thrust.
                thrust_correction = ctrl.k_thrust_vertical * v_err_norm

                # If Pitch is significantly steeper than target (initial dive), cut thrust to avoid overspeed
                if pitch < target_pitch - 0.2:
                     thrust_track = 0.1 # Idle
                else:
                     thrust_track = thrust_base - thrust_correction

                thrust_track = max(ctrl.thrust_min, min(ctrl.thrust_max, thrust_track))

        else:
            yaw_rate_cmd = 0.0
            # Keep constant pitch
            thrust_track = 0.55

        # --- Active Braking (Override if too fast) ---
        if velocity_est:
             limit = ctrl.velocity_limit # 10.0
             vio_vz_enu = -velocity_est['vz']
             est_vz_error = abs(vio_vz_enu - self.est_vz)
             vio_plausible = (est_vz_error < 5.0) and (vel_mag < 15.0)

             if vel_mag > limit:
                  if vio_plausible:
                      logger.info(f"HARD LIMIT: v={vel_mag:.2f} > {limit}. Engaging Airbrake.")

                      target_pitch_brake = 0.0 # Level Out
                      pitch_err_brake = target_pitch_brake - pitch
                      pitch_rate_cmd = -ctrl.k_pitch_brake * pitch_err_brake

                      thrust_track = 0.1 # Cut throttle

        # --- Safety Floor ---
        if obs_pz is not None and obs_pz < 15.0 and self.est_vz < -3.0:
             target_pitch_floor = 0.3 # Nose Up
             pitch_err_floor = target_pitch_floor - pitch
             pitch_rate_cmd = -ctrl.k_pitch_floor * pitch_err_floor
             thrust_track = 0.8

        # --- Final Command ---

        yaw_rate_cmd += extra_yaw_rate
        roll_rate_cmd = 4.0 * (0.0 - roll)

        action = {
            'thrust': thrust_track,
            'roll_rate': roll_rate_cmd,
            'pitch_rate': pitch_rate_cmd,
            'yaw_rate': -yaw_rate_cmd
        }
        self.last_action = action

        # Ghost Paths
        sim_vx = self.est_vx
        sim_vy = self.est_vy
        sim_vz = vz
        ghost_paths = []
        path = []
        sim_x = 0.0; sim_y = 0.0; sim_z = 0.0
        for i in range(20):
             sim_x += sim_vx * self.dt
             sim_y += sim_vy * self.dt
             sim_z += sim_vz * self.dt
             path.append({'px': sim_x, 'py': sim_y, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths
