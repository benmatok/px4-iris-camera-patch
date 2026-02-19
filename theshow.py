import sys
import os
import asyncio
import numpy as np
import math
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from vision.projection import Projector
    from sim_interface import SimDroneInterface
    from visual_tracker import VisualTracker
    from vision.feature_tracker import FeatureTracker
    from vision.msckf import MSCKF
    from flight_controller import DPCFlightController
    from mission_manager import MissionManager
except ImportError as e:
    logger.error(f"Could not import project modules: {e}")
    sys.exit(1)

# Constants
DT = 0.05

class TheShow:
    def __init__(self):
        try:
            # 1. Initialize Components
            # Tilt 30.0 (Up) as requested
            self.projector = Projector(width=640, height=480, fov_deg=120.0, tilt_deg=30.0)

            # Scenario / Sim
            self.sim = SimDroneInterface(self.projector)
            self.target_pos_sim_world = [50.0, 0.0, 0.0] # Blind Dive Target

            # Initial Pos for Blind Dive (Hardcoded Default)
            drone_pos = [0.0, 0.0, 100.0]
            pitch, yaw = self.calculate_initial_orientation(drone_pos, self.target_pos_sim_world)

            self.sim.reset_to_scenario("Blind Dive", pos_x=drone_pos[0], pos_y=drone_pos[1], pos_z=drone_pos[2], pitch=pitch, yaw=yaw)

            # Perception
            self.tracker = VisualTracker(self.projector)
            self.feature_tracker = FeatureTracker(self.projector)
            self.msckf = MSCKF(self.projector)

            # Logic
            self.mission = MissionManager()

            # Control
            controller_mode = os.environ.get('CONTROLLER_MODE', 'PID')
            logger.info(f"Initializing DPCFlightController in {controller_mode} mode")
            self.controller = DPCFlightController(dt=DT, mode=controller_mode)

            self.loops = 0
            self.prediction_history = []
            self.websockets = set()
            self.paused = False
            self.step_once = False
            self.time_scale = 1.0
            logger.info("TheShow initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TheShow: {e}")
            raise

    def calculate_initial_orientation(self, drone_pos, target_pos):
        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]
        dz = target_pos[2] - drone_pos[2]

        yaw = np.arctan2(dy, dx)
        dist_xy = np.sqrt(dx*dx + dy*dy)
        pitch_vec = np.arctan2(dz, dist_xy)

        # Camera Tilt is 30.0 (Up)
        camera_tilt = np.deg2rad(30.0)

        # Body Pitch = Vec Pitch - Camera Tilt
        # Sim Pitch is Negative = Nose Down.
        pitch = (pitch_vec - camera_tilt)

        # Clamp to avoid inversion
        if pitch < -1.48:
            pitch = -1.48

        return pitch, yaw

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.websockets.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.websockets.remove(websocket)

    async def handle_message(self, message: str):
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'reset':
                logger.info(f"Received reset command: {data}")

                # Default Blind Dive parameters
                # Target is at [50.0, 0.0, 0.0]

                alt = float(data.get('altitude', 100.0))
                dist = float(data.get('distance', 50.0))

                # Calculate Drone Pos
                # Drone X = Target X - Distance
                # Drone Y = Target Y
                # Drone Z = Altitude

                drone_x = self.target_pos_sim_world[0] - dist
                drone_y = self.target_pos_sim_world[1]
                drone_z = alt

                pitch, yaw = self.calculate_initial_orientation([drone_x, drone_y, drone_z], self.target_pos_sim_world)

                # Reset Sim
                self.sim.reset_to_scenario("Blind Dive", pos_x=drone_x, pos_y=drone_y, pos_z=drone_z, pitch=pitch, yaw=yaw)

                # Reset Logic
                self.mission.reset(target_alt=alt)
                self.controller.reset()

                # Reset VIO
                self.msckf = MSCKF(self.projector)
                self.feature_tracker = FeatureTracker(self.projector)
                self.msckf.initialized = False

                self.prediction_history = []
                self.loops = 0
                self.paused = False

            elif msg_type == 'pause':
                self.paused = True
                logger.info("Simulation Paused")

            elif msg_type == 'resume':
                self.paused = False
                logger.info("Simulation Resumed")

            elif msg_type == 'step':
                self.paused = True
                self.step_once = True
                logger.info("Simulation Step")

            elif msg_type == 'update_target':
                tx = float(data.get('x', self.target_pos_sim_world[0]))
                ty = float(data.get('y', self.target_pos_sim_world[1]))
                tz = float(data.get('z', self.target_pos_sim_world[2]))
                self.target_pos_sim_world = [tx, ty, tz]
                logger.info(f"Target Updated: {self.target_pos_sim_world}")

            elif msg_type == 'set_speed':
                try:
                    speed = float(data.get('speed', 1.0))
                    self.time_scale = max(0.1, min(2.0, speed))
                    logger.info(f"Speed set to: {self.time_scale}")
                except ValueError:
                    logger.error("Invalid speed value")

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def sim_to_ned(self, sim_state):
        # Maps Sim (ENU) to NED (Right-Handed)
        # Sim: X=East, Y=North, Z=Up
        # NED: X=North, Y=East, Z=Down
        # Yaw: Sim 0=East -> NED pi/2=East
        #      Sim pi/2=North -> NED 0=North
        #      Yaw_ned = pi/2 - Yaw_sim

        # NED X (North) = Sim Y (North)
        # NED Y (East) = Sim X (East)

        ned = sim_state.copy()
        ned['px'] = sim_state['py']
        ned['py'] = sim_state['px']
        ned['pz'] = -sim_state['pz']

        ned['vx'] = sim_state.get('vy', 0.0)
        ned['vy'] = sim_state.get('vx', 0.0)
        ned['vz'] = -sim_state.get('vz', 0.0)

        # Roll/Pitch align if we only care about body frame definition (Fwd/Right/Down)
        # Sim: Fwd/Left/Up. Roll(Fwd), Pitch(Left).
        # NED: Fwd/Right/Down. Roll(Fwd), Pitch(Right).
        # Pitch sign: Nose Up is Positive in both.
        # Roll sign: Right Wing Down is Positive in both.

        ned['roll'] = sim_state.get('roll', 0.0)
        # Sim Pitch (Nose Down -) -> NED Pitch (Nose Down -). Match signs.
        ned['pitch'] = sim_state.get('pitch', 0.0)
        sim_yaw = sim_state.get('yaw', 0.0)
        ned['yaw'] = (math.pi / 2.0) - sim_yaw

        # Normalize Yaw
        ned['yaw'] = (ned['yaw'] + math.pi) % (2 * math.pi) - math.pi

        return ned

    def sim_pos_to_ned_pos(self, sim_pos):
        # [x, y, z] -> [y, x, -z]
        return [sim_pos[1], sim_pos[0], -sim_pos[2]]

    def ned_rel_to_sim_rel(self, ned_rel):
        # [dx, dy, dz] NED -> [dy, dx, -dz] Sim
        # Because NED Rel X (North) corresponds to Sim Rel Y (North)
        return [ned_rel[1], ned_rel[0], -ned_rel[2]]

    async def broadcast(self, data):
        if not self.websockets: return

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        msg = json.dumps(data, cls=NumpyEncoder)
        logger.info(f"Broadcasting State: Drone={data.get('drone', {})}, Target={data.get('target', [])}")
        for ws in list(self.websockets):
            try:
                await ws.send_text(msg)
            except:
                self.websockets.discard(ws)

    def compute_step(self):
        """
        Runs one iteration of the control loop using the modular architecture.
        """
        # Define Ground Truth Target for Sim (Visualization Only)
        target_pos_sim_world = self.target_pos_sim_world

        # 1. Get State (Sim Frame)
        s = self.sim.get_state()

        # 2. Get Perception Data
        # Get synthetic image
        img = self.sim.get_image(target_pos_sim_world)

        # Convert Sim State to NED (Absolute)
        dpc_state_ned_abs = self.sim_to_ned(s)

        # Convert Target Sim Pos to NED Pos (Absolute)
        target_pos_ned_abs = self.sim_pos_to_ned_pos(target_pos_sim_world)

        # Detect and Localize (Returns Relative Target Position in NED)
        # Using Ground Truth Target (Perfect Projection) per user request to bypass detection for now
        # We pass Absolute Drone State (NED) and Target (NED)
        center, target_wp_ned, radius = self.tracker.process(
            img,
            dpc_state_ned_abs,
            ground_truth_target_pos=target_pos_ned_abs
        )

        # Convert Result (target_wp_ned) back to Sim Relative Frame
        target_wp_sim = None
        if target_wp_ned:
             target_wp_sim = self.ned_rel_to_sim_rel(target_wp_ned)

        # Normalize Tracking UV for Controller
        tracking_norm = None
        tracking_size_norm = None
        if center:
            tracking_norm = self.projector.pixel_to_normalized(center[0], center[1])
            # Normalize radius by image height (480)
            tracking_size_norm = radius / 480.0

        # 3. Update Mission Logic
        # Pass sanitized relative state to mission (px=0, py=0)
        sim_state_rel = s.copy()
        sim_state_rel['px'] = 0.0
        sim_state_rel['py'] = 0.0

        mission_state, dpc_target, extra_yaw = self.mission.update(sim_state_rel, (center, target_wp_sim))

        # --- VIO UPDATE ---

        # 1. IMU Propagation
        # Get IMU data from Sim State (Sim Frame: Forward-Left-Up)
        # VIO expects NED Body Frame (Forward-Right-Down)
        # Transformation: X->X, Y->-Y, Z->-Z
        gyro = np.array([s['wx'], -s['wy'], -s['wz']], dtype=np.float64)
        accel = np.array([s.get('ax_b', 0.0), -s.get('ay_b', 0.0), -s.get('az_b', 9.81)], dtype=np.float64)

        # Initialize if needed
        if not self.msckf.initialized:
            # Init with Truth for now (In real life, static alignment)
            # q_wb = from rpy
            # NED Quaternion
            r = dpc_state_ned_abs['roll']
            p = dpc_state_ned_abs['pitch']
            y = dpc_state_ned_abs['yaw']

            # Use scipy to get quat
            from scipy.spatial.transform import Rotation as R
            q_init = R.from_euler('xyz', [r, p, y], degrees=False).as_quat()

            p_init = np.array([dpc_state_ned_abs['px'], dpc_state_ned_abs['py'], dpc_state_ned_abs['pz']])
            v_init = np.array([dpc_state_ned_abs['vx'], dpc_state_ned_abs['vy'], dpc_state_ned_abs['vz']])

            self.msckf.initialize(q_init, p_init, v_init)

        self.msckf.propagate(gyro, accel, DT)

        # 2. State Augmentation (Camera Image)
        # Clone current pose
        self.msckf.augment_state()

        # 3. Feature Tracking & Update
        current_clone_idx = self.msckf.cam_clones[-1]['id'] if self.msckf.cam_clones else 0

        # dpc_state_ned_abs used for generation inside tracker, but we should rely on image content ideally
        # Here we use synthetic generation
        # Body rates for feature tracker (Sim Frame or NED? Usually NED for unrotation)
        # Using aligned gyro (NED)
        body_rates_ned = (gyro[0], gyro[1], gyro[2])

        foe, finished_tracks = self.feature_tracker.update(dpc_state_ned_abs, body_rates_ned, DT, current_clone_idx)

        if finished_tracks:
            self.msckf.update_features(finished_tracks)

        # 4. Measurement Updates (Height, Vz, Features)
        height_meas = dpc_state_ned_abs['pz'] # NED Pz
        vz_meas = dpc_state_ned_abs['vz'] # NED Vz

        self.msckf.update_measurements(height_meas, vz_meas, finished_tracks)

        # Get Estimated State (Velocity & Position)
        vio_state = self.msckf.get_state_dict()
        vel_est = {'vx': vio_state['vx'], 'vy': vio_state['vy'], 'vz': vio_state['vz']}
        pos_est = {'px': vio_state['px'], 'py': vio_state['py'], 'pz': vio_state['pz']}
        vel_reliable = self.msckf.is_reliable()

        foe_px = None
        if foe:
            u_norm, v_norm = foe
            u_px = u_norm * self.projector.fx + self.projector.cx
            v_px = v_norm * self.projector.fy + self.projector.cy
            foe_px = {'u': u_px, 'v': v_px}

        # 4. Compute Control
        # Construct observed state for controller
        state_obs = {
            'pz': s['pz'],
            # 'vz': s['vz'], # Velocity Z removed per user request
            'roll': s['roll'],
            'pitch': s['pitch'],
            'yaw': s['yaw'],
            'wx': s['wx'],
            'wy': s['wy'],
            'wz': s['wz']
        }

        # dpc_target is [RelX, RelY, AbsZ] (Sim Frame)
        # target_wp_ned is [RelX, RelY, RelZ] (NED Frame) needed for Optical Flow calc in Controller
        action_out, ghost_paths = self.controller.compute_action(
            state_obs,
            dpc_target,
            tracking_uv=tracking_norm,
            tracking_size=tracking_size_norm,
            extra_yaw_rate=extra_yaw,
            foe_uv=foe,
            velocity_est=vel_est,
            position_est=pos_est,
            velocity_reliable=vel_reliable
        )

        # 5. Apply Control to Sim
        # Controller (Z-Up) -> Sim (Z-Up)
        sim_action = np.array([
            action_out['thrust'],
            action_out['roll_rate'],
            action_out['pitch_rate'],
            action_out['yaw_rate']
        ])

        # Special Launch Kick (Legacy Logic preserved? Or allow controller to handle?)
        # Legacy logic in theshow.py: if TAKEOFF and alt < 2.0: thrust = 0.8
        # Let's preserve it for robustness if mission is in TAKEOFF
        if mission_state == "TAKEOFF" and s['pz'] < 2.0:
             sim_action[0] = 0.8

        self.sim.step(sim_action)

        # Prediction Logging
        current_time = self.loops * DT
        if ghost_paths and len(ghost_paths) > 0 and len(ghost_paths[0]) > 0:
            # 1-Step Prediction
            p1 = ghost_paths[0][0]
            self.prediction_history.append({
                'time': current_time + DT,
                'state': p1,
                'target_used': dpc_target,
                'type': 'step1'
            })

            # Horizon Prediction
            ph = ghost_paths[0][-1]
            horizon_steps = len(ghost_paths[0])
            self.prediction_history.append({
                'time': current_time + horizon_steps * DT,
                'state': ph,
                'target_used': dpc_target,
                'type': 'horizon'
            })

        # Process Mature Predictions
        dpc_error = {}
        valid_preds = [p for p in self.prediction_history if p['time'] <= current_time]

        # Remove processed predictions from history
        self.prediction_history = [x for x in self.prediction_history if x['time'] > current_time]

        # Calc Frontend Error (Horizon Only)
        horizon_preds = [p for p in valid_preds if p['type'] == 'horizon']
        if horizon_preds:
            p = horizon_preds[-1]

            pred_s = p['state']
            # act_s = dpc_state_ned_rel # Removed. Use dpc_state_ned_abs

            # 1. Height Error (Altitude)
            # pred_s is Z-Up (Sim). dpc_state_ned_abs['pz'] is Z-Down (NED).
            pred_alt = pred_s['pz']
            act_alt = -dpc_state_ned_abs['pz']
            height_err = pred_alt - act_alt

            # 2. Projection Error
            # Convert pred_s (Sim Z-Up) to NED for Projector
            pred_s_ned = self.sim_to_ned(pred_s)

            tgt_used = p['target_used'] # Sim Frame
            # Convert to NED
            tx, ty, tz = self.sim_pos_to_ned_pos(tgt_used)

            pred_u, pred_v, pred_rad = 0, 0, 0
            res = self.projector.project_point_with_size(tx, ty, tz, pred_s_ned, object_radius=0.5)
            if res:
                pred_u, pred_v, pred_rad = res

            # Actual Measured
            act_u, act_v, act_rad = 0, 0, 0
            if center:
                act_u, act_v = center
                act_rad = radius

            u_err = pred_u - act_u
            v_err = pred_v - act_v
            size_err = pred_rad - act_rad

            dpc_error = {
                'height_error': round(height_err, 2),
                'u_error': round(u_err, 1),
                'v_error': round(v_err, 1),
                'size_error': round(size_err, 1),
                'pred_time': round(p['time'], 2)
            }

        # 6. Prepare Payload
        # Construct Absolute NED State for Visualization (Already Computed)
        # dpc_state_ned_abs = ...

        # Construct Absolute Target Viz (NED)
        # dpc_target is [RelX, RelY, AbsZ_Up] (Sim Frame)
        # Need Target Viz in NED Absolute.
        # Target Sim Abs = Drone Sim Abs + Rel Target Sim
        # Wait. dpc_target[0,1] is Relative. dpc_target[2] is Absolute Z.

        target_sim_abs_x = s['px'] + dpc_target[0]
        target_sim_abs_y = s['py'] + dpc_target[1]
        target_sim_abs_z = dpc_target[2]

        target_viz_ned = self.sim_pos_to_ned_pos([target_sim_abs_x, target_sim_abs_y, target_sim_abs_z])

        # Transform Ghosts to Absolute NED Frame for Frontend
        # ghost_paths: [px, py, pz] (Z-Up, Relative to Solver Origin [px=0, py=0])
        # Need: [px, py, pz] (NED, Absolute)

        ghosts_viz = []
        if ghost_paths:
            for path in ghost_paths:
                new_path = []
                if not path:
                    continue

                # Calculate offset to anchor trajectory to drone's true position
                # ghost path uses estimated Z, drone uses true Z
                # We want trajectory to visually emanate from the drone
                ghost_start_z = path[0]['pz']
                true_start_z = s['pz']
                delta_z = true_start_z - ghost_start_z

                for pt in path:
                    # Construct Sim Absolute
                    sim_abs_x = s['px'] + pt['px']
                    sim_abs_y = s['py'] + pt['py']
                    sim_abs_z = pt['pz'] + delta_z

                    # Convert to NED
                    ned_pos = self.sim_pos_to_ned_pos([sim_abs_x, sim_abs_y, sim_abs_z])

                    new_pt = {
                        'px': ned_pos[0],
                        'py': ned_pos[1],
                        'pz': ned_pos[2]
                    }
                    new_path.append(new_pt)
                ghosts_viz.append(new_path)

        # Convert Sim Target to NED for Frontend
        sim_target_ned = self.sim_pos_to_ned_pos(target_pos_sim_world)

        payload = {
            'state': mission_state,
            'drone': dpc_state_ned_abs, # Absolute for Viz
            'control': action_out,
            'target': target_viz_ned, # Absolute for Viz (Red)
            'sim_target': sim_target_ned, # Absolute NED (Green)
            'tracker': {'u': center[0] if center else 0, 'v': center[1] if center else 0, 'size': radius},
            'flight_direction': foe_px,
            'velocity_est': vel_est,
            'dpc_error': dpc_error,
            'ghosts': ghosts_viz,
            'paused': self.paused
        }
        return payload

    async def control_loop(self):
        logger.info("Starting Control Loop...")
        last_payload = None
        try:
            while True:
                start_time = asyncio.get_running_loop().time()

                if not self.paused or self.step_once:
                    # Offload heavy computation to a thread
                    payload = await asyncio.to_thread(self.compute_step)
                    last_payload = payload
                    self.loops += 1
                    self.step_once = False
                else:
                    # If paused, we might still want to broadcast the last state
                    # but maybe update the 'paused' status if it changed?
                    # For now, just reuse last payload if available, ensuring 'paused' is correct.
                    if last_payload:
                        payload = last_payload.copy()
                        payload['paused'] = True
                        # Also update target position in payload if it changed while paused
                        payload['sim_target'] = self.target_pos_sim_world
                    else:
                        payload = {'paused': True, 'state': 'WAITING'}

                # Broadcast result
                await self.broadcast(payload)

                elapsed = asyncio.get_running_loop().time() - start_time
                delay = max(0, (DT / self.time_scale) - elapsed)
                await asyncio.sleep(delay)

        except asyncio.CancelledError:
            logger.info("Loop Cancelled")
        except Exception as e:
            logger.error(f"Loop Error: {e}")
            import traceback
            traceback.print_exc()

# Global Controller Instance
try:
    the_show = TheShow()
except:
    logger.critical("Failed to create Global Controller. Application will exit.")
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Lifespan Startup...")
    loop_task = asyncio.create_task(the_show.control_loop())
    yield
    # Shutdown
    logger.info("Lifespan Shutdown...")
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await the_show.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await the_show.handle_message(data)
    except WebSocketDisconnect:
        the_show.disconnect(websocket)

# Determine absolute path to web directory
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(current_dir, "web")

if os.path.isdir(web_dir):
    logger.info(f"Mounting {web_dir} to /")
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
else:
    logger.warning(f"Warning: 'web' directory not found at {web_dir}. Static files will not be served.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
