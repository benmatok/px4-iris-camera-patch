from mavsdk.offboard import Attitude
import asyncio
from mavsdk import System


async def main():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for PX4...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 connected")
            break

    print("Arming...")
    await drone.action.arm()
    await asyncio.sleep(1)

    print("Sending zero attitude setpoint")
    await drone.offboard.set_attitude(
        Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, thrust_value=0.1)
    )

    print("Starting offboard")
    await drone.offboard.start()

    print("Motors should spin now (low thrust)")
    await asyncio.sleep(5)

    print("Stopping motors")
    await drone.offboard.stop()
    await drone.action.disarm()


asyncio.run(main())