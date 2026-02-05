import sys
import os
import asyncio
import numpy as np
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
            # Tilt -45.0 (Down) to see ground targets
            self.projector = Projector(width=640, height=480, fov_deg=110.0, tilt_deg=-45.0)

            # Scenario / Sim
            self.sim = SimDroneInterface(self.projector)
            self.sim.reset_to_scenario("Blind Dive") # Init Blind Dive
            self.target_pos_sim_world = [50.0, 0.0, 0.0] # Blind Dive Target

            # Perception
            self.tracker = VisualTracker(self.projector)

            # Logic
            self.mission = MissionManager()

            # Control
            self.controller = DPCFlightController(dt=DT)

            self.loops = 0
            self.prediction_history = []
            self.websockets = set()
            logger.info("TheShow initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TheShow: {e}")
            raise

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.websockets.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.websockets.remove(websocket)

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

        # Convert Sim State to NED for Tracking/Control
        dpc_state_ned = {
            'px': s['px'], 'py': s['py'], 'pz': -s['pz'],
            'vx': s['vx'], 'vy': s['vy'], 'vz': -s['vz'],
            'roll': s['roll'], 'pitch': s['pitch'], 'yaw': -s['yaw']
        }

        # Detect and Localize
        center, target_wp, radius = self.tracker.process(img, dpc_state_ned)

        # 3. Update Mission Logic
        mission_state, dpc_target, extra_yaw = self.mission.update(s, (center, target_wp))

        # 4. Compute Control (Using Z-Up State "s")
        action_out, ghost_paths = self.controller.compute_action(s, dpc_target, extra_yaw)

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
            pred_state = ghost_paths[0][-1]
            horizon_steps = len(ghost_paths[0])
            pred_time = current_time + horizon_steps * DT
            self.prediction_history.append({
                'time': pred_time,
                'state': pred_state,
                'target_used': dpc_target
            })

        # Error Calculation
        dpc_error = {}
        valid_preds = [p for p in self.prediction_history if p['time'] <= current_time]
        if valid_preds:
            # Use the latest valid prediction
            p = valid_preds[-1]

            # Remove processed/old predictions
            self.prediction_history = [x for x in self.prediction_history if x['time'] > current_time]

            pred_s = p['state']
            act_s = dpc_state_ned

            # 1. Height Error (Altitude)
            # pred_s is Z-Up. act_s is NED.
            pred_alt = pred_s['pz']
            act_alt = -act_s['pz']
            height_err = pred_alt - act_alt

            # 2. Projection Error
            # Convert pred_s (Z-Up) to NED for Projector
            pred_s_ned = {
                'px': pred_s['px'], 'py': pred_s['py'], 'pz': -pred_s['pz'],
                'vx': pred_s['vx'], 'vy': pred_s['vy'], 'vz': -pred_s['vz'],
                'roll': pred_s['roll'], 'pitch': pred_s['pitch'], 'yaw': -pred_s['yaw']
            }

            tgt_used = p['target_used']
            tx, ty, tz = tgt_used

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
        payload = {
            'state': mission_state,
            'drone': dpc_state_ned,
            'target': dpc_target,
            'sim_target': target_pos_sim_world,
            'tracker': {'u': center[0] if center else 0, 'v': center[1] if center else 0, 'size': radius},
            'dpc_error': dpc_error,
            'ghosts': ghost_paths
        }
        return payload

    async def control_loop(self):
        logger.info("Starting Control Loop...")
        try:
            while True:
                start_time = asyncio.get_running_loop().time()

                # Offload heavy computation to a thread
                payload = await asyncio.to_thread(self.compute_step)

                # Broadcast result
                await self.broadcast(payload)

                self.loops += 1
                elapsed = asyncio.get_running_loop().time() - start_time
                delay = max(0, DT - elapsed)
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
            await websocket.receive_text()
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
