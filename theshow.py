import sys
import os
import asyncio
import math
import numpy as np
import cv2
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from vision.detector import RedObjectDetector
    from vision.projection import Projector
    from ghost_dpc.ghost_dpc import PyDPCSolver, PyGhostModel
    from drone_env.drone import DroneEnv
except ImportError as e:
    logger.error(f"Could not import project modules: {e}")
    sys.exit(1)

# Constants
TARGET_ALT = 50.0
DT = 0.05

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
            'yaw': float(self.dd['yaw'][0])
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

class TheShow:
    def __init__(self):
        try:
            self.projector = Projector(width=640, height=480, fov_deg=110.0, tilt_deg=30.0)
            self.sim = SimDroneInterface(self.projector)
            self.detector = RedObjectDetector()

            self.solver = PyDPCSolver()
            self.models_config = [{'mass': 3.33, 'drag_coeff': 0.3, 'thrust_coeff': 54.5}]
            self.weights = [1.0]

            self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

            self.state = "INIT"
            self.dpc_target = [0.0, 0.0, -TARGET_ALT] # NED Target
            self.loops = 0

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
        msg = json.dumps(data)
        # Iterate over a copy to avoid runtime error if set changes during iteration
        for ws in list(self.websockets):
            try:
                await ws.send_text(msg)
            except:
                self.websockets.discard(ws)

    def rollout_ghosts(self, start_state, action_dict, horizon=10):
        ghosts = []
        for cfg in self.models_config:
            model = PyGhostModel(cfg['mass'], cfg['drag_coeff'], cfg['thrust_coeff'])
            path = []
            curr = start_state.copy()

            for _ in range(horizon):
                # Step
                next_s = model.step(curr, action_dict, DT)
                path.append(next_s)
                curr = next_s
            ghosts.append(path)
        return ghosts

    def compute_step(self):
        """
        Runs one iteration of the control loop (Logic, Solver, Physics).
        This is CPU intensive and should be run in a separate thread.
        """
        target_pos_sim = [30.0, 30.0, 0.0] # World Z=0

        # 1. Get State (Sim Frame)
        s = self.sim.get_state()

        # Convert to NED for Solver
        dpc_state = {
            'px': s['px'], 'py': s['py'], 'pz': -s['pz'],
            'vx': s['vx'], 'vy': s['vy'], 'vz': -s['vz'],
            'roll': s['roll'], 'pitch': s['pitch'], 'yaw': -s['yaw']
        }

        # 2. Logic (Detector, State Machine)
        img = self.sim.get_image(target_pos_sim)
        current_alt = -dpc_state['pz']

        if self.state == "TAKEOFF":
            self.dpc_target = [s['px'], s['py'], -TARGET_ALT]
            if current_alt >= TARGET_ALT - 5.0:
                self.state = "SCAN"

        elif self.state == "SCAN":
            if img is not None:
                center, _, _ = self.detector.detect(img)
                if center:
                    self.state = "HOMING"

        elif self.state == "HOMING":
            if img is not None:
                center, _, _ = self.detector.detect(img)
                if center:
                    wp = self.projector.pixel_to_world(center[0], center[1], dpc_state)
                    if wp:
                        self.dpc_target = [wp[0], wp[1], -2.0]

        # 3. Solve (DPC)
        action_out = self.solver.solve(
            dpc_state,
            self.dpc_target,
            self.last_action,
            self.models_config,
            self.weights,
            DT
        )
        self.last_action = action_out

        if self.state == "SCAN":
            action_out['yaw_rate'] = math.radians(15.0)

        # 4. Step Sim
        sim_action = np.array([
            action_out['thrust'],
            action_out['roll_rate'],
            action_out['pitch_rate'],
            -action_out['yaw_rate']
        ])

        if self.state == "TAKEOFF" and current_alt < 2.0:
             sim_action[0] = 0.8

        self.sim.step(sim_action)

        # 5. Rollout Ghosts
        ghost_paths = self.rollout_ghosts(dpc_state, action_out)

        # 6. Prepare Payload
        payload = {
            'state': self.state,
            'drone': dpc_state,
            'target': self.dpc_target,
            'ghosts': ghost_paths
        }
        return payload

    async def control_loop(self):
        logger.info("Starting Control Loop...")
        self.state = "TAKEOFF"

        try:
            while True:
                start_time = asyncio.get_running_loop().time()

                # Offload heavy computation to a thread
                payload = await asyncio.to_thread(self.compute_step)

                # Broadcast result (must happen on event loop)
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
            await websocket.receive_text() # Keep connection alive, listen for msgs (ignored)
    except WebSocketDisconnect:
        the_show.disconnect(websocket)

# Determine absolute path to web directory
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(current_dir, "web")

if os.path.isdir(web_dir):
    logger.info(f"Serving web content from {web_dir}")

    # Explicit route for root to ensure index.html is served
    @app.get("/")
    async def read_index():
        return FileResponse(os.path.join(web_dir, "index.html"))

    # Mount static files at root for other assets (main.js, etc)
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
else:
    logger.warning(f"Warning: 'web' directory not found at {web_dir}. Static files will not be served.")

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to bind to all interfaces
    uvicorn.run(app, host="0.0.0.0", port=8080)
