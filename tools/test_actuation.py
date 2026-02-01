#!/usr/bin/env python3
import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, Attitude, VelocityBodyYawspeed)

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Setting initial setpoint")
    # We need to send setpoints before starting offboard
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    print("-- Roll +0.1 rad (~5.7 deg)")
    # set_attitude takes deg? No, usually Euler is deg in MAVSDK Attitude struct?
    # Checking docs: Attitude(roll_deg, pitch_deg, yaw_deg, thrust_value)
    # The plan says "Roll = +0.1 rad". 0.1 rad approx 5.7 deg.

    roll_deg = 5.7
    await drone.offboard.set_attitude(Attitude(roll_deg, 0.0, 0.0, 0.5)) # 0.5 Thrust to hover roughly
    await asyncio.sleep(5)

    print("-- Roll -0.1 rad (~-5.7 deg)")
    await drone.offboard.set_attitude(Attitude(-roll_deg, 0.0, 0.0, 0.5))
    await asyncio.sleep(5)

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: {error._result.result}")

    print("-- Landing")
    await drone.action.land()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
