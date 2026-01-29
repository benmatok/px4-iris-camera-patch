import asyncio
import math
from dataclasses import dataclass

from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError

from pynput import keyboard


@dataclass
class Cmd:
    # Body frame velocities (m/s)
    fwd: float = 0.0   # + forward
    right: float = 0.0 # + right
    down: float = 0.0  # + down  (note: down is positive in NED)
    yaw_rate: float = 0.0  # deg/s


class KeyController:
    def __init__(self):
        self.cmd = Cmd()
        self._pressed = set()
        self._stop = False

        # Tuning
        self.v_xy = 1.0       # m/s
        self.v_z = 0.5        # m/s
        self.yaw_deg_s = 30.0 # deg/s

    def on_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = str(key)

        self._pressed.add(k)

        # Emergency/utility
        if k == 'q':
            self._stop = True
        if k == ' ':
            self.cmd = Cmd()

    def on_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = str(key)

        if k in self._pressed:
            self._pressed.remove(k)

    def update_cmd(self):
        # Reset each tick
        fwd = 0.0
        right = 0.0
        down = 0.0
        yaw = 0.0

        # Movement keys
        if 'w' in self._pressed:
            fwd += self.v_xy
        if 's' in self._pressed:
            fwd -= self.v_xy

        if 'd' in self._pressed:
            right += self.v_xy
        if 'a' in self._pressed:
            right -= self.v_xy

        # Altitude (I up, K down) => down is positive in NED
        if 'i' in self._pressed:
            down -= self.v_z
        if 'k' in self._pressed:
            down += self.v_z

        # Yaw
        if 'l' in self._pressed:
            yaw += self.yaw_deg_s
        if 'j' in self._pressed:
            yaw -= self.yaw_deg_s

        self.cmd = Cmd(fwd=fwd, right=right, down=down, yaw_rate=yaw)


async def wait_connected(drone: System):
    print("Waiting for PX4...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 connected")
            return


async def main():
    ctrl = KeyController()

    listener = keyboard.Listener(on_press=ctrl.on_press, on_release=ctrl.on_release)
    listener.start()

    drone = System()
    await drone.connect(system_address="udp://:14540")
    await wait_connected(drone)

    # Arm
    print("Arming...")
    await drone.action.arm()
    await asyncio.sleep(1)

    # Send an initial setpoint before starting Offboard
    print("Setting initial setpoint...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    print("Starting Offboard...")
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        print("Disarming...")
        await drone.action.disarm()
        return

    print("Keyboard control active:")
    print("  W/S: forward/back | A/D: left/right | I/K: up/down | J/L: yaw | Space: stop | Q: quit")

    try:
        rate_hz = 20.0
        dt = 1.0 / rate_hz

        while not ctrl._stop:
            ctrl.update_cmd()

            # Clamp just in case
            fwd = float(max(-3.0, min(3.0, ctrl.cmd.fwd)))
            right = float(max(-3.0, min(3.0, ctrl.cmd.right)))
            down = float(max(-2.0, min(2.0, ctrl.cmd.down)))
            yaw = float(max(-90.0, min(90.0, ctrl.cmd.yaw_rate)))

            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(fwd, right, down, yaw))
            await asyncio.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping Offboard and disarming...")
        try:
            await drone.offboard.stop()
        except Exception:
            pass
        try:
            await drone.action.disarm()
        except Exception:
            pass

        listener.stop()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())