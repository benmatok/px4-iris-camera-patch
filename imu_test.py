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

    async for imu in drone.telemetry.imu():
        acc = imu.acceleration_frd
        gyro = imu.angular_velocity_frd

        print(
            f"ACC: [{acc.forward_m_s2:.2f}, {acc.right_m_s2:.2f}, {acc.down_m_s2:.2f}] | "
            f"GYRO: [{gyro.forward_rad_s:.2f}, {gyro.right_rad_s:.2f}, {gyro.down_rad_s:.2f}]"
        )


if __name__ == "__main__":
    asyncio.run(main())