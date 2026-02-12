import sys
import os
import asyncio
import math
import numpy as np
import cv2
import time
import json
import argparse

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    _HAS_ROS = True
except ImportError:
    print("Warning: ROS2 (rclpy) not found. Real mode will run in HEADLESS configuration (Synthetic Vision via MAVSDK only).")
    _HAS_ROS = False
    class Node:
        def __init__(self, name): pass
        def create_subscription(self, *args): pass
        def destroy_node(self): pass
        def get_logger(self):
             class Logger:
                 def error(self, msg): print(f"Error: {msg}")
             return Logger()
    class CvBridge:
        def imgmsg_to_cv2(self, msg, encoding="bgr8"): return None

# MAVSDK Imports
from mavsdk import System
from mavsdk.offboard import AttitudeRate, OffboardError, VelocityBodyYawspeed
# Removed explicit telemetry imports to avoid version issues

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from vision.detector import RedObjectDetector
    from vision.projection import Projector
    from ghost_dpc.ghost_dpc import PyDPCSolver
    from drone_env.drone import DroneEnv  # For Sim Mode
except ImportError as e:
    print(f"Error: Could not import project modules: {e}")
    sys.exit(1)

# Constants
TARGET_ALT = 2.0        # Meters (Relative) - Target Altitude
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
    async def set_velocity_body(self, vf, vr, vd, yaw_rate): raise NotImplementedError


class RealDroneInterface(DroneInterface):
    def __init__(self, projector=None, target_pos=None, system_address="udp://:14540"):
        self.drone = System()
        self.system_address = system_address
        self.latest_image = None # Updated by external callback
        self.projector = projector
        self.target_pos = target_pos if target_pos is not None else [30.0, 30.0, 0.0]

        # Cache state for synthetic vision
        self.last_pos = None
        self.last_att = None

    async def connect(self):
        print(f"Connecting to drone on {self.system_address}...")
        await self.drone.connect(system_address=self.system_address)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Drone Connected!")
                break

    async def attitude_euler(self):
        async for x in self.drone.telemetry.attitude_euler():
            self.last_att = x
            yield x

    async def position_velocity_ned(self):
        async for x in self.drone.telemetry.position_velocity_ned():
            self.last_pos = x
            yield x

    async def arm(self):
        await self.drone.action.arm()

    async def is_armed(self):
        async for armed in self.drone.telemetry.armed():
            return armed

    async def offboard_stop(self):
        try:
            await self.drone.offboard.stop()
        except:
            pass


    async def offboard_start(self):
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )
        await self.drone.offboard.start()

    async def set_velocity_body(self, vf, vr, vd, yaw_rate):
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vf, vr, vd, yaw_rate)
        )




    async def land(self):
        await self.drone.action.land()




    def set_latest_image(self, img):
        self.latest_image = img

    def get_latest_image(self):
        # If ROS is available and provided an image, use it
        if self.latest_image is not None:
            return self.latest_image

        # Fallback: Synthetic Vision if Projector is available
        if self.projector is not None and self.last_pos is not None and self.last_att is not None:
            width = 1280
            height = 800
            img = np.zeros((height, width, 3), dtype=np.uint8)

            drone_state = {
                'px': self.last_pos.position.north_m,
                'py': self.last_pos.position.east_m,
                'pz': self.last_pos.position.down_m,
                'roll': math.radians(self.last_att.roll_deg),
                'pitch': math.radians(self.last_att.pitch_deg),
                'yaw': math.radians(self.last_att.yaw_deg)
            }

            tx, ty, tz = self.target_pos
            uv = self.projector.world_to_pixel(tx, ty, tz, drone_state)

            if uv:
                u, v = uv
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(img, (int(u), int(v)), 20, (0, 0, 255), -1)
            return img

        return None

    def get_dpc_state_sync(self):
        return None




# Mock Classes for Sim Interface
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
        self.pz = self.dd['pos_z'] # Positive Up in DroneEnv??
        # Wait, DroneEnv `step_cpu`: `pz += vz * dt`.
        # `az_gravity = -g`.
        # This implies Z is Up (Gravity acts down).
        # MAVSDK is NED (Z Down).
        # I need to convert.
        # In DroneEnv:
        # z=0 is ground? `underground = pz < terr_z`.
        # So Z is Up.

        # Force start on ground (but above 0.5m to avoid immediate DroneEnv termination)
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
        self.masses[0] = 3.33 # Matched to Controller Model
        self.thrust_coeffs = self.dd['thrust_coeffs']
        self.thrust_coeffs[0] = 2.725 # Match Controller TC=54.5 (20 * 2.725 = 54.5)
        # step_cpu: `max_thrust = 20.0 * thrust_coeffs`.

        # Red Object Position (World Frame - Sim Frame Z Up)
        self.target_pos = [30.0, 30.0, 0.0]

        # State for Async Generators
        self.running = True

    async def connect(self):
        print("Connected to Sim Drone.")

    async def attitude_euler(self):
        while self.running:
            # DroneEnv R,P,Y are radians.
            # MAVSDK expects degrees.
            # DroneEnv Frame: Z Up. MAVSDK: NED.
            # Sim R,P,Y -> NED R,P,Y?
            # If Z_sim = -Z_ned.
            # Rotations:
            # Roll (about X): Same.
            # Pitch (about Y): Inverted?
            # Yaw (about Z): Inverted?
            # Let's keep it simple: assume Sim is aligned ENU (East North Up).
            # NED: North, East, Down.
            # X_sim=East?? usually X=Forward.
            # Let's assume Sim is: X=North, Y=East, Z=Up.
            # NED: X=North, Y=East, Z=Down.
            # So Z_ned = -Z_sim.
            # P_ned = -P_sim. (Pitch up is + in Sim, - in NED? No, Pitch up is + in both usually).
            # Pitching Nose Up:
            # Sim: Rot around Y. Z goes back.
            # NED: Rot around Y. Z goes forward.
            # Let's trust standard conversion.
            # Pitch_ned = -Pitch_sim.
            # Yaw_ned = -Yaw_sim + 90? (ENU to NED).
            # Let's stick to X=North, Y=East, Z=Up convention for Sim.

            r_deg = math.degrees(self.roll[0])
            p_deg = math.degrees(self.pitch[0]) # Pitch Up is positive in sim?
            # step_cpu: `az_thrust = thrust_force * (cp * cr) / masses`.
            # `az_gravity = -g`.
            # So Thrust +Z opposes Gravity -Z.
            # If Pitch=0, Thrust is Up.
            # This is standard Quadcopter Z-Up.
            # Pitching positive usually means Nose Up.

            # MAVSDK (NED): Pitching Positive is Nose Up.
            # So Pitch_deg = Pitch_sim_deg.

            # Yaw:
            # Sim: X=North.
            # NED: X=North.
            # Yaw=0 -> North.
            # Sim: Counter-Clockwise is positive?
            # NED: Clockwise is positive.
            # So Yaw_ned = -Yaw_sim.

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
        # Enforce Model Parameters (in case of auto-reset)
        self.masses[0] = 3.33
        # Boost Thrust for 60deg tilt climb (T/W > 2.0 required)
        self.thrust_coeffs[0] = 5.0

        # Step Physics
        # Map inputs to actions array
        # Actions: [Thrust, RollRate, PitchRate, YawRate]
        # MAVSDK: thrust [0,1].
        # Sim: thrust [0,1].

        # Rates: MAVSDK deg/s. Sim rad/s.
        rr = math.radians(roll_deg)
        pr = math.radians(pitch_deg)
        # NED Yaw Rate (+Clockwise). Sim Yaw Rate (+Counter-Clockwise).
        # So yr = -radians(yaw_deg)
        yr = -math.radians(yaw_deg)

        actions = np.array([thrust, rr, pr, yr], dtype=np.float32)

        # Step
        # step_cpu expects flattened actions for all agents
        # We need to construct arguments for step_cpu manually or use `env.step_function`.
        # The `env.step_function` signature is huge.
        # But `env.step` is not implemented in DroneEnv (it's WarpDrive style).
        # We can call `env.step_function` with kwargs.

        self.dd['actions'][:] = actions

        kwargs = self.env.get_step_function_kwargs()
        # Resolve args from data_dictionary
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
        # Prevent DroneEnv internal resets (we handle logic in TheShow)
        self.dd['done_flags'][:] = 0.0

    async def land(self):
        print("Sim: Landing.")
        self.running = False

    def get_latest_image(self):
        # Generate Synthetic Image
        width = 1280
        height = 800
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Drone State for Projection
        # Projector expects NED?
        # `pixel_to_world` uses `drone_state['pz']` (NED Z).
        # `drone_state['roll']`, etc.
        # If I pass NED state to Projector, it should work.

        drone_state = {
            'px': self.px[0],
            'py': self.py[0],
            'pz': -self.pz[0], # NED
            'roll': self.roll[0],
            'pitch': self.pitch[0],
            'yaw': -self.yaw[0] # NED
        }

        # Target Pos (Sim Z-Up to World NED? No, Target is World Coords.)
        # If Projector expects World NED, then Target Z should be negative if up?
        # Actually `Projector` defines Z=0 as ground usually.
        # `pixel_to_world`: `intersection = p0 + t * vec_w`. `t = -p0[2] / vz`.
        # If p0[2] (Drone Z NED) is -10 (10m Up).
        # intersection Z will be 0.
        # So Target Z=0 corresponds to Ground.

        tx, ty, tz = self.target_pos
        # Sim Target is at Z=0.

        uv = self.projector.world_to_pixel(tx, ty, tz, drone_state)

        if uv:
            u, v = uv
            if 0 <= u < width and 0 <= v < height:
                # Draw Red Circle
                cv2.circle(img, (int(u), int(v)), 20, (0, 0, 255), -1)

        return img

    def get_dpc_state_sync(self):
        # Return state directly from DroneEnv arrays (Zero Latency)
        # NED Conversion
        n = self.px[0]
        e = self.py[0]
        d = -self.pz[0]

        vn = self.vx[0]
        ve = self.vy[0]
        vd = -self.vz[0]

        # Euler (Radians)
        # Sim is Rads. DPC expects Rads.
        # But DPC expects NED Euler?
        # Sim (Z-Up) to NED (Z-Down).
        # Pitch_ned = -Pitch_sim (as discussed)
        # Yaw_ned = -Yaw_sim
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
        self.control_history = []
        self.events = []
        self.target_detected = False
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
        # state: DPC state dict
        t = timestamp if timestamp is not None else time.time()
        if self.start_time is None:
            self.start_time = t

        entry = {
            'time': t,
            'phase': phase,
            'pos': [state['px'], state['py'], state['pz']],
            'target': target,
            'action': action # dict
        }
        self.trajectory.append(entry)

    def save_report(self):
        if not self.trajectory:
            print("No trajectory logged.")
            return

        total_time = self.trajectory[-1]['time'] - self.trajectory[0]['time']

        # Homing Metrics
        homing_duration = 0.0
        homing_path_len = 0.0
        final_error = 0.0

        homing_entries = [e for e in self.trajectory if e['phase'] == "HOMING"]
        if homing_entries:
            homing_duration = homing_entries[-1]['time'] - homing_entries[0]['time']

            # Path Length
            pts = np.array([e['pos'] for e in homing_entries])
            diffs = np.diff(pts, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            homing_path_len = np.sum(dists)

            # Final Error (Distance to Estimated Target)
            last_pos = np.array(homing_entries[-1]['pos'])
            last_tgt = np.array(homing_entries[-1]['target'])
            # Target is [x, y, -2].
            # Error in XY?
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
        print("Saving report to JSON...")
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Report saved to {self.output_file}")


# -----------------------------------------------------------------------------
# The Show Node
# -----------------------------------------------------------------------------

class TheShow(Node):
    def __init__(self, interface: DroneInterface, benchmark=False, headless=False):
        super().__init__('the_show')
        self.bridge = CvBridge()
        self.detector = RedObjectDetector()

        self.interface = interface
        self.benchmark = benchmark
        self.headless = headless

        if self.benchmark:
            self.logger = BenchmarkLogger()
        else:
            self.logger = None

        # Projector Config (Matches video_viewer.py)
        # 1280x800, 110 deg FOV, 30 deg Tilt
        self.projector = Projector(width=1280, height=800, fov_deg=110.0, tilt_deg=30.0)

        # DPC Solver
        self.solver = PyDPCSolver()
        self.models = [{'mass': 3.33, 'drag_coeff': 0.3, 'thrust_coeff': 54.5}]
        self.weights = [1.0]
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

        # Subscriptions (Only for Real mode)
        # If Sim mode, the interface handles image generation internally or we poll it.
        # But for Real mode we need ROS subscription if ROS is available.
        if isinstance(interface, RealDroneInterface) and _HAS_ROS:
            self.create_subscription(Image, '/forward_camera/image_raw', self.img_cb, 10)

        # Telemetry State
        self.pos_ned = None
        self.vel_ned = None
        self.att_euler = None
        self.connected = False

        # Logic State
        self.state = "INIT"
        self.loops = 0
        self.dpc_target = [0.0, 0.0, -TARGET_ALT]
        self.start_time_real = None

        # --- Attitude targets (ABSOLUTE) ---
        self.target_roll = 0.0          # rad
        self.target_pitch = 0.0         # rad
        self.target_yaw = None          # rad (will lock on takeoff)


    def img_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.interface.set_latest_image(img)
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    async def connect_drone(self):
        await self.interface.connect()
        self.connected = True
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
        # Try Sync State first (for Sim)
        sync_state = self.interface.get_dpc_state_sync()
        if sync_state is not None:
            return sync_state

        # Fallback to Telemetry (for Real)
        if self.pos_ned is None or self.vel_ned is None or self.att_euler is None:
            return None

        return {
            'px': self.pos_ned.north_m,
            'py': self.pos_ned.east_m,
            'pz': self.pos_ned.down_m,
            'vx': self.vel_ned.north_m_s,
            'vy': self.vel_ned.east_m_s,
            'vz': self.vel_ned.down_m_s,
            'roll': math.radians(self.att_euler.roll_deg),
            'pitch': math.radians(self.att_euler.pitch_deg),
            'yaw': math.radians(self.att_euler.yaw_deg)
        }

    async def control_loop(self):
        print("Arming...")

        try:
            await self.interface.arm()
        except Exception as e:
            print(f"Arming command failed: {e}")
            return

        # Wait up to 10 seconds for ARM confirmation
        armed = False
        for i in range(100):  # 100 × 0.1s = 10s
            try:
                if await self.interface.is_armed():
                    armed = True
                    break
            except:
                pass
            await asyncio.sleep(0.1)

        if not armed:
            print("ERROR: Drone did not arm after 10 seconds")
            return

        print("Drone armed.")


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
        # Lock yaw at takeoff
        self.target_yaw = self.get_dpc_state()['yaw']
        start_x = self.pos_ned.north_m
        start_y = self.pos_ned.east_m
        self.dpc_target = [start_x, start_y, -TARGET_ALT]

        print(f"State: {self.state} - Target: {self.dpc_target}")

        self.scan_yaw = math.degrees(self.target_yaw)


        if not self.headless:
            try:
                cv2.namedWindow("The Show", cv2.WINDOW_NORMAL)
            except:
                print("Warning: Could not create window. Headless?")

        if self.start_time_real is None:
            self.start_time_real = time.time()

        try:
            while True:

                if _HAS_ROS:
                    if not rclpy.ok():
                        break
                    rclpy.spin_once(self, timeout_sec=0.001)

                # Time Management
                if isinstance(self.interface, SimDroneInterface):
                    current_time = self.loops * DT
                else:
                    current_time = time.time() - self.start_time_real

                #get drone state
                dpc_state = self.get_dpc_state()
                if dpc_state is None:
                    await asyncio.sleep(0.01)
                    continue

                # Get Image
                img = self.interface.get_latest_image()
                if img is None:
                  print("NO IMAGE")
                  continue


                #system state (TAKEOFF, SCAN, HOMING, DONE)
                self.update_logic(dpc_state, img, timestamp=current_time)

                action_out = self.solver.solve(
                    dpc_state,
                    self.dpc_target,
                    self.last_action,
                    self.models,
                    self.weights,
                    DT
                )
                self.last_action = action_out

                thrust = action_out['thrust']

                # ======================================
                # SELECT COMMAND BY STATE
                # ======================================

                if self.state == "TAKEOFF":

                    v_forward = 0.0
                    v_right   = 0.0
                    v_down    = -0.5
                    yaw_rate  = 0.0

                    vx = 0.0
                    vy = 0.0
                    vz = -0.5

                else:

                    # Position P-controller
                    Kp_pos = 0.6

                    vx = Kp_pos * (self.dpc_target[0] - dpc_state['px'])
                    vy = Kp_pos * (self.dpc_target[1] - dpc_state['py'])
                    vz = Kp_pos * (self.dpc_target[2] - dpc_state['pz'])

                    # Velocity limits
                    MAX_V = 2.0
                    vx = max(-MAX_V, min(MAX_V, vx))
                    vy = max(-MAX_V, min(MAX_V, vy))
                    vz = max(-MAX_V, min(MAX_V, vz))

                    # ===============================
                    # Convert NED velocity -> Body velocity
                    # ===============================

                    yaw = dpc_state['yaw']  # radians

                    v_forward =  math.cos(yaw) * vx + math.sin(yaw) * vy
                    v_right   = -math.sin(yaw) * vx + math.cos(yaw) * vy
                    v_down    = vz

                # Yaw control
                if self.state == "SCAN":
                    yaw_rate = SCAN_YAW_RATE
                else:
                        yaw_rate = 0.0

                await self.interface.set_velocity_body(
                    v_forward,
                    v_right,
                    v_down,
                    yaw_rate
                )



                if self.loops < 30:
                    print(
                        f"[DBG {self.loops}] "
                        f"state={self.state} "
                        f"pos=({dpc_state['px']:.2f},{dpc_state['py']:.2f},{dpc_state['pz']:.2f}) "
                        f"vel_cmd=({vx:.2f},{vy:.2f},{vz:.2f})"
                    )


                # Logging
                if self.logger:
                    self.logger.log_step(dpc_state, self.dpc_target, action_out, self.state, timestamp=current_time)

                print("IMG:", img is None, "HEADLESS:", self.headless)


                # Visualization
                if img is not None and not self.headless:
                    disp = img.copy()
                    cv2.putText(disp, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(disp, f"Alt: {-dpc_state['pz']:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if self.state == "HOMING":
                        center, _, _ = self.detector.detect(img)
                        if center:
                            cv2.circle(disp, (int(center[0]), int(center[1])), 10, (0, 0, 255), 2)

                    cv2.imshow("The Show", disp)
                    cv2.waitKey(16)
                    print("next frame")

                self.loops += 1

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt in control_loop")
            pass
        finally:
            print("Stopping...")
            if not self.headless:
                cv2.destroyAllWindows()
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
            if current_alt >= TARGET_ALT - 0.2:
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
    parser.add_argument("--mode", choices=["real", "sim"], default="real", help="Run mode")
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmarking")
    parser.add_argument("--headless", action="store_true", help="Run without visualization window")
    parser.add_argument("--target-pos", nargs=3, type=float, default=[30.0, 30.0, 0.0], help="Target Position X Y Z")
    args = parser.parse_args()

    if _HAS_ROS:
        rclpy.init()

    # Shared Projector Config
    proj = Projector(width=1280, height=800, fov_deg=110.0, tilt_deg=30.0)

    if args.mode == "real":
        # Pass projector to Real interface for Synthetic Vision fallback
        interface = RealDroneInterface(projector=proj, target_pos=args.target_pos)
    else:
        interface = SimDroneInterface(proj)
        interface.target_pos = args.target_pos # Ensure Sim uses same target

    node = TheShow(interface, benchmark=args.benchmark, headless=args.headless)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(node.connect_drone())

    try:
        loop.run_until_complete(node.control_loop())
    except KeyboardInterrupt:
        print("Loop Interrupted")
    finally:
        if node.logger:
            node.logger.save_report()
        node.destroy_node()
        if _HAS_ROS:
            rclpy.shutdown()

if __name__ == "__main__":
    main()
