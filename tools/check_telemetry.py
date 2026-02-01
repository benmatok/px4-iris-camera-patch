#!/usr/bin/env python3
import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    # Start tasks
    print("Fetching Telemetry...")

    async def print_position():
        async for pos in drone.telemetry.position():
            print(f"Position (LLA): Lat={pos.latitude_deg:.6f}, Lon={pos.longitude_deg:.6f}, Alt={pos.relative_altitude_m:.2f}m")
            # For brevity, only print once per second (logic handled by caller or sleep)
            await asyncio.sleep(1.0)

    async def print_attitude():
        async for att in drone.telemetry.attitude_euler():
            print(f"Attitude: Roll={att.roll_deg:.2f}, Pitch={att.pitch_deg:.2f}, Yaw={att.yaw_deg:.2f}")
            await asyncio.sleep(1.0)

    async def print_imu():
        async for imu in drone.telemetry.imu():
            print(f"IMU Accel: X={imu.acceleration_frd.forward_m_s2:.2f}, Y={imu.acceleration_frd.right_m_s2:.2f}, Z={imu.acceleration_frd.down_m_s2:.2f}")
            await asyncio.sleep(1.0)

    # Run them concurrently
    await asyncio.gather(print_position(), print_attitude(), print_imu())

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
