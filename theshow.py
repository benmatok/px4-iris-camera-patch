import sys
import os
import asyncio
import math
import numpy as np
import cv2
import time
import json
import argparse
from aiohttp import web

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from vision.detector import RedObjectDetector
    from vision.projection import Projector
    from ghost_dpc.ghost_dpc import PyDPCSolver, PyGhostModel
    from drone_env.drone import DroneEnv
except ImportError as e:
    print(f"Error: Could not import project modules: {e}")
    sys.exit(1)

# Constants
TARGET_ALT = 50.0        # Meters (Relative) - Target Altitude
SCAN_YAW_RATE = 15.0     # deg/s (Override for scanning)
DT = 0.05                # 20Hz

# -----------------------------------------------------------------------------
# Interfaces
# -----------------------------------------------------------------------------

class DroneInterface:
    async def connect(self): raise NotImplementedError
    def attitude_euler(self): raise NotImplementedError # async generator
    def position_velocity_ned(self): raise NotImplementedError # async generator
    async def arm(self): raise NotImplementedError
    async def offboard_start(self): raise NotImplementedError
    async def offboard_stop(self): raise NotImplementedError
    async def set_attitude_rate(self, roll, pitch, yaw, thrust): raise NotImplementedError
    async def land(self): raise NotImplementedError
    def get_latest_image(self): return None # Returns cv2 image or None
    def get_dpc_state_sync(self): return None # Returns dict or None

class MockAttitudeEuler:
    def __init__(self, r, p, y):
        self.roll_deg = r
        self.pitch_deg = p
        self.yaw_deg = y

class MockPosition:
    def __init__(self, n, e, d):
        self.north_m = n
        self.east_m = e
        self.down_m = d

class MockVelocity:
    def __init__(self, n, e, d):
        self.north_m_s = n
        self.east_m_s = e
        self.down_m_s = d

class MockPositionVelocityNed:
    def __init__(self, n, e, d, vn, ve, vd):
        self.position = MockPosition(n, e, d)
        self.velocity = MockVelocity(vn, ve, vd)

class SimDroneInterface(DroneInterface):
    def __init__(self, projector):
        self.projector = projector
        # Initialize DroneEnv
        # Single Agent
        self.env = DroneEnv(num_agents=1, episode_length=100000)
        self.env.reset_all_envs()

        # Access raw arrays
        self.dd = self.env.data_dictionary
        self.px = self.dd['pos_x']
        self.py = self.dd['pos_y']
        self.pz = self.dd['pos_z'] # Positive Up in DroneEnv

        # Force start on ground
        self.px[0] = 0.0
        self.py[0] = 0.0
        self.pz[0] = 1.0

        self.vx = self.dd['vel_x']
        self.vy = self.dd['vel_y']
        self.vz = self.dd['vel_z']

        self.roll = self.dd['roll']
        self.pitch = self.dd['pitch']
        self.yaw = self.dd['yaw']

        self.masses = self.dd['masses']
        self.masses[0] = 3.33
        self.thrust_coeffs = self.dd['thrust_coeffs']
        self.thrust_coeffs[0] = 2.725

        # Red Object Position (World Frame - Sim Frame Z Up)
        self.target_pos = [30.0, 30.0, 0.0]

        # State for Async Generators
        self.running = True

    async def connect(self):
        print("Connected to Sim Drone.")

    async def attitude_euler(self):
        while self.running:
            # NED Conversion
            # Pitch_ned = -Pitch_sim
            # Yaw_ned = -Yaw_sim

            r_deg = math.degrees(self.roll[0])
            p_deg = math.degrees(self.pitch[0])
            y_deg = math.degrees(self.yaw[0])

            yield MockAttitudeEuler(r_deg, p_deg, -y_deg)
            await asyncio.sleep(0.01)

    async def position_velocity_ned(self):
        while self.running:
            # Sim (Z-Up) to NED (Z-Down)
            n = self.px[0]
            e = self.py[0]
            d = -self.pz[0]

            vn = self.vx[0]
            ve = self.vy[0]
            vd = -self.vz[0]

            yield MockPositionVelocityNed(n, e, d, vn, ve, vd)
            await asyncio.sleep(0.01)

    async def arm(self):
        print("Sim: Armed.")

    async def offboard_start(self):
        print("Sim: Offboard Started.")

    async def offboard_stop(self):
        print("Sim: Offboard Stopped.")

    async def set_attitude_rate(self, roll_deg, pitch_deg, yaw_deg, thrust):
        self.masses[0] = 3.33
        self.thrust_coeffs[0] = 5.0

        rr = math.radians(roll_deg)
        pr = math.radians(pitch_deg)
        yr = -math.radians(yaw_deg)

        actions = np.array([thrust, rr, pr, yr], dtype=np.float32)

        self.dd['actions'][:] = actions

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
        self.dd['done_flags'][:] = 0.0

    async def land(self):
        print("Sim: Landing.")
        self.running = False

    def get_latest_image(self):
        width = 1280
        height = 800
        img = np.zeros((height, width, 3), dtype=np.uint8)

        drone_state = {
            'px': self.px[0],
            'py': self.py[0],
            'pz': -self.pz[0], # NED
            'roll': self.roll[0],
            'pitch': self.pitch[0],
            'yaw': -self.yaw[0] # NED
        }

        tx, ty, tz = self.target_pos
        uv = self.projector.world_to_pixel(tx, ty, tz, drone_state)

        if uv:
            u, v = uv
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(img, (int(u), int(v)), 20, (0, 0, 255), -1)

        return img

    def get_dpc_state_sync(self):
        n = self.px[0]
        e = self.py[0]
        d = -self.pz[0]

        vn = self.vx[0]
        ve = self.vy[0]
        vd = -self.vz[0]

        r = self.roll[0]
        p = -self.pitch[0]
        y = -self.yaw[0]

        return {
            'px': n, 'py': e, 'pz': d,
            'vx': vn, 'vy': ve, 'vz': vd,
            'roll': r, 'pitch': p, 'yaw': y
        }

# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------

class BenchmarkLogger:
    def __init__(self, output_file="benchmark_results.json"):
        self.output_file = output_file
        self.start_time = None
        self.homing_start_time = None
        self.end_time = None
        self.trajectory = []
        self.events = []
        self.success = False

    def log_event(self, name, timestamp=None):
        t = timestamp if timestamp is not None else time.time()
        self.events.append({'time': t, 'event': name})
        if name == "HOMING_START":
            self.homing_start_time = t
        if name == "DONE":
            self.end_time = t
            self.success = True

    def log_step(self, state, target, action, phase, timestamp=None):
        t = timestamp if timestamp is not None else time.time()
        if self.start_time is None:
            self.start_time = t

        entry = {
            'time': t,
            'phase': phase,
            'pos': [state['px'], state['py'], state['pz']],
            'target': target,
            'action': action
        }
        self.trajectory.append(entry)

    def save_report(self):
        if not self.trajectory:
            print("No trajectory logged.")
            return

        total_time = self.trajectory[-1]['time'] - self.trajectory[0]['time']

        homing_duration = 0.0
        homing_path_len = 0.0
        final_error = 0.0

        homing_entries = [e for e in self.trajectory if e['phase'] == "HOMING"]
        if homing_entries:
            homing_duration = homing_entries[-1]['time'] - homing_entries[0]['time']
            pts = np.array([e['pos'] for e in homing_entries])
            diffs = np.diff(pts, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            homing_path_len = np.sum(dists)

            last_pos = np.array(homing_entries[-1]['pos'])
            last_tgt = np.array(homing_entries[-1]['target'])
            err_xy = np.linalg.norm(last_pos[:2] - last_tgt[:2])
            final_error = err_xy

        report = {
            "success": self.success,
            "total_duration": total_time,
            "homing_duration": homing_duration,
            "homing_path_length": homing_path_len,
            "final_position_error_xy": final_error,
            "num_steps": len(self.trajectory)
        }

        print("\n" + "="*40)
        print("BENCHMARK REPORT")
        print("="*40)
        print(json.dumps(report, indent=4))
        print("="*40 + "\n")
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Report saved to {self.output_file}")


# -----------------------------------------------------------------------------
# The Show
# -----------------------------------------------------------------------------

class TheShow:
    def __init__(self, interface: DroneInterface, benchmark=False, headless=False):
        self.bridge = None # Removed ROS CvBridge
        self.detector = RedObjectDetector()

        self.interface = interface
        self.benchmark = benchmark
        self.headless = headless

        if self.benchmark:
            self.logger = BenchmarkLogger()
        else:
            self.logger = None

        self.projector = Projector(width=1280, height=800, fov_deg=110.0, tilt_deg=30.0)

        # DPC Solver
        self.solver = PyDPCSolver()
        self.models_config = [
            {'mass': 3.33, 'drag_coeff': 0.3, 'thrust_coeff': 54.5}, # Nominal
            {'mass': 3.33, 'drag_coeff': 0.3, 'thrust_coeff': 54.5, 'wind_x': 5.0}, # Headwind
            {'mass': 3.33, 'drag_coeff': 0.3, 'thrust_coeff': 54.5, 'wind_y': 5.0}, # Crosswind
        ]
        self.weights = [1.0, 0.0, 0.0] # Only use nominal for now unless we implement adaptive

        # Instantiate PyGhostModels for trajectory rollout
        self.ghost_models = [
            PyGhostModel(
                m['mass'], m['drag_coeff'], m['thrust_coeff'],
                m.get('wind_x', 0.0), m.get('wind_y', 0.0)
            ) for m in self.models_config
        ]

        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

        self.pos_ned = None
        self.vel_ned = None
        self.att_euler = None

        self.state = "INIT"
        self.loops = 0
        self.dpc_target = [0.0, 0.0, -TARGET_ALT]
        self.start_time_real = None

        # Web Server
        self.websockets = set()
        self.app = web.Application()
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_static('/', path='web', name='static')
        self.runner = None

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        print("Web Client Connected")
        try:
            async for msg in ws:
                pass
        finally:
            self.websockets.remove(ws)
            print("Web Client Disconnected")
        return ws

    async def start_server(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        # Port 3000 for Live Preview tool compatibility
        site = web.TCPSite(self.runner, '0.0.0.0', 3000)
        await site.start()
        print("Web Server started at http://localhost:3000")

    async def broadcast_state(self, state, target, ghosts):
        if not self.websockets: return
        data = {
            'drone': {'pos': [state['px'], state['py'], state['pz']], 'att': [state['roll'], state['pitch'], state['yaw']]},
            'target': target,
            'ghosts': ghosts
        }
        msg = json.dumps(data)
        for ws in set(self.websockets):
            try:
                await ws.send_str(msg)
            except:
                self.websockets.discard(ws)

    async def connect_drone(self):
        await self.interface.connect()
        asyncio.create_task(self.telemetry_loop())

    async def telemetry_loop(self):
        async def att_loop():
            async for att in self.interface.attitude_euler():
                self.att_euler = att

        async def pos_vel_loop():
            async for pv in self.interface.position_velocity_ned():
                self.pos_ned = pv.position
                self.vel_ned = pv.velocity

        await asyncio.gather(att_loop(), pos_vel_loop())

    def get_dpc_state(self):
        sync_state = self.interface.get_dpc_state_sync()
        if sync_state is not None:
            return sync_state
        return None

    def generate_ghost_trajectories(self, state, action):
        trajectories = []
        horizon = 10
        for model in self.ghost_models:
            traj = []
            curr_s = state
            for _ in range(horizon):
                next_s = model.step(curr_s, action, DT)
                traj.append([next_s['px'], next_s['py'], next_s['pz']])
                curr_s = next_s
            trajectories.append(traj)
        return trajectories

    async def control_loop(self):
        print("Arming...")
        try:
            await self.interface.arm()
        except:
            print("Arming failed.")

        print("Wait for Telemetry...")
        while self.pos_ned is None:
            await asyncio.sleep(0.1)

        print("Starting Offboard...")
        try:
            await self.interface.offboard_start()
        except Exception as e:
            print(f"Offboard failed: {e}")
            return

        self.state = "TAKEOFF"
        start_x = self.pos_ned.north_m
        start_y = self.pos_ned.east_m
        self.dpc_target = [start_x, start_y, -TARGET_ALT]

        if self.start_time_real is None:
            self.start_time_real = time.time()

        try:
            while True:
                current_time = self.loops * DT

                dpc_state = self.get_dpc_state()
                if dpc_state is None:
                    await asyncio.sleep(0.01)
                    continue

                img = self.interface.get_latest_image()

                self.update_logic(dpc_state, img, timestamp=current_time)

                action_out = self.solver.solve(
                    dpc_state,
                    self.dpc_target,
                    self.last_action,
                    self.models_config,
                    self.weights,
                    DT
                )
                self.last_action = action_out

                # Ghost Visualization
                ghosts = self.generate_ghost_trajectories(dpc_state, action_out)
                await self.broadcast_state(dpc_state, self.dpc_target, ghosts)

                roll_rate = math.degrees(action_out['roll_rate'])
                pitch_rate = math.degrees(action_out['pitch_rate'])
                yaw_rate = math.degrees(action_out['yaw_rate'])
                thrust = action_out['thrust']

                if self.state == "SCAN":
                    yaw_rate = SCAN_YAW_RATE

                if self.state == "TAKEOFF" and -dpc_state['pz'] < 2.0:
                    thrust = 0.8

                thrust = max(0.0, min(1.0, thrust))

                await self.interface.set_attitude_rate(roll_rate, pitch_rate, yaw_rate, thrust)

                if self.logger:
                    self.logger.log_step(dpc_state, self.dpc_target, action_out, self.state, timestamp=current_time)

                # Optional CV2 window (local debug only)
                if not self.headless and img is not None:
                    disp = img.copy()
                    cv2.putText(disp, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("The Show", disp)
                    cv2.waitKey(1)

                await asyncio.sleep(DT)
                self.loops += 1

        except KeyboardInterrupt:
            pass
        finally:
            print("Stopping...")
            if not self.headless:
                cv2.destroyAllWindows()
            if self.runner:
                await self.runner.cleanup()
            try:
                await self.interface.offboard_stop()
                await self.interface.land()
            except:
                pass
            if self.logger:
                self.logger.save_report()

    def update_logic(self, state, img, timestamp=None):
        current_alt = -state['pz']

        if self.state == "TAKEOFF":
            if current_alt >= TARGET_ALT - 5.0:
                print("Altitude Reached. Switching to SCAN.")
                self.state = "SCAN"
                if self.logger: self.logger.log_event("SCAN_START", timestamp=timestamp)

        elif self.state == "SCAN":
            if img is not None:
                center, area, bbox = self.detector.detect(img)
                if center:
                    print("Target Detected! Switching to HOMING.")
                    self.state = "HOMING"
                    if self.logger: self.logger.log_event("HOMING_START", timestamp=timestamp)

        elif self.state == "HOMING":
            if img is None:
                return

            center, area, bbox = self.detector.detect(img)

            if not center:
                return

            world_pt = self.projector.pixel_to_world(center[0], center[1], state)

            if world_pt:
                self.dpc_target = [world_pt[0], world_pt[1], -2.0]

                dist_xy = math.sqrt((state['px'] - world_pt[0])**2 + (state['py'] - world_pt[1])**2)
                dist_z = abs(state['pz'] - (-2.0))

                if dist_xy < 1.0 and dist_z < 1.0:
                     print("Target Reached! Stopping.")
                     self.state = "DONE"
                     if self.logger: self.logger.log_event("DONE", timestamp=timestamp)

        elif self.state == "DONE":
             raise KeyboardInterrupt

def main():
    parser = argparse.ArgumentParser()
    # Mode deprecated (always sim now), but keep arg for compatibility or just ignore
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmarking")
    parser.add_argument("--headless", action="store_true", help="Run without cv2 window")
    parser.add_argument("--target-pos", nargs=3, type=float, default=[30.0, 30.0, 0.0], help="Target Position X Y Z")
    args = parser.parse_args()

    proj = Projector(width=1280, height=800, fov_deg=110.0, tilt_deg=30.0)
    interface = SimDroneInterface(proj)
    interface.target_pos = args.target_pos

    node = TheShow(interface, benchmark=args.benchmark, headless=args.headless)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start Server
    loop.run_until_complete(node.start_server())

    loop.run_until_complete(node.connect_drone())

    try:
        loop.run_until_complete(node.control_loop())
    except KeyboardInterrupt:
        print("Loop Interrupted")
    finally:
        if node.logger:
            node.logger.save_report()

if __name__ == "__main__":
    main()
