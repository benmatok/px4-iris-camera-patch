import pytest
import asyncio
import sys
import os
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.append(os.getcwd())

# Mock modules
sys.modules['mavsdk'] = MagicMock()
sys.modules['mavsdk.offboard'] = MagicMock()
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()
sys.modules['cv_bridge'] = MagicMock()

# Mock cv2
mock_cv2 = MagicMock()
mock_cv2.WINDOW_NORMAL = 0
mock_cv2.FONT_HERSHEY_SIMPLEX = 0
sys.modules['cv2'] = mock_cv2
sys.modules['cv2.typing'] = MagicMock()

# Setup Module Mocks
mock_vision_detector = MagicMock()
mock_vision_projection = MagicMock()
mock_ghost_dpc = MagicMock()

# Configure RedObjectDetector class mock
mock_detector_instance = MagicMock()
mock_detector_instance.detect.return_value = ((320.0, 240.0), 100.0, (0,0,10,10))
# When the class is instantiated, return the instance
mock_vision_detector.RedObjectDetector.return_value = mock_detector_instance

# Configure Projector class mock
mock_projector_instance = MagicMock()
mock_projector_instance.pixel_to_world.return_value = (10.0, 5.0, 0.0)
mock_vision_projection.Projector.return_value = mock_projector_instance

# Configure Solver class mock
mock_solver_instance = MagicMock()
mock_solver_instance.solve.return_value = {
    'thrust': 0.8, 'roll_rate': 0.1, 'pitch_rate': -0.1, 'yaw_rate': 0.05
}
mock_ghost_dpc.PyDPCSolver.return_value = mock_solver_instance

# Pre-patch modules to allow import
with patch.dict(sys.modules, {
    'vision.detector': mock_vision_detector,
    'vision.projection': mock_vision_projection,
    'ghost_dpc.ghost_dpc': mock_ghost_dpc
}):
    import video_viewer
    from video_viewer import DroneController

# Configure Drone Mock
mock_drone = MagicMock(name="MockDrone")
mock_drone.connect = AsyncMock(name="connect")
mock_drone.action = MagicMock()
mock_drone.action.arm = AsyncMock(name="arm")
mock_drone.offboard = MagicMock()
mock_drone.offboard.start = AsyncMock(name="start")
mock_drone.offboard.set_attitude_rate = AsyncMock(name="set_attitude_rate")

# Telemetry
mock_drone.telemetry = MagicMock()
async def mock_pos_vel():
    mock_pv = MagicMock()
    mock_pv.position.north_m = 0.0
    mock_pv.position.east_m = 0.0
    mock_pv.position.down_m = -10.0
    mock_pv.velocity.north_m_s = 0.0
    mock_pv.velocity.east_m_s = 0.0
    mock_pv.velocity.down_m_s = 0.0
    while True:
        yield mock_pv
        await asyncio.sleep(0.1)

async def mock_att():
    mock_a = MagicMock()
    mock_a.roll_deg = 0.0
    mock_a.pitch_deg = 0.0
    mock_a.yaw_deg = 0.0
    while True:
        yield mock_a
        await asyncio.sleep(0.1)

mock_drone.telemetry.position_velocity_ned.return_value = mock_pos_vel()
mock_drone.telemetry.attitude_euler.return_value = mock_att()

# Set global system mock return value
sys.modules['mavsdk'].System.return_value = mock_drone

@pytest.mark.asyncio
async def test_drone_controller_track_logic():
    # Reset mocks
    mock_detector_instance.detect.reset_mock()
    mock_projector_instance.pixel_to_world.reset_mock()
    mock_solver_instance.solve.reset_mock()
    mock_drone.offboard.set_attitude_rate.reset_mock()

    controller = DroneController(mode="track")
    assert controller.drone is mock_drone

    await controller.connect()

    # State
    controller.pos_ned = MagicMock(north_m=0.0, east_m=0.0, down_m=-10.0)
    controller.vel_ned = MagicMock(north_m_s=0.0, east_m_s=0.0, down_m_s=0.0)
    controller.att_euler = MagicMock(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    controller.is_offboard = True

    mock_image = np.zeros((480, 640, 3), dtype=np.uint8)

    await controller.control_step(mock_image)

    mock_detector_instance.detect.assert_called_once()
    mock_projector_instance.pixel_to_world.assert_called_once()
    mock_solver_instance.solve.assert_called_once()
    mock_drone.offboard.set_attitude_rate.assert_called_once()
    print("Track Logic Passed")

@pytest.mark.asyncio
async def test_drone_controller_hover_logic():
    # Update solver mock return for hover (though it doesn't matter much for flow)
    mock_solver_instance.solve.return_value = {'thrust':0.6, 'roll_rate':0, 'pitch_rate':0, 'yaw_rate':0}
    mock_solver_instance.solve.reset_mock()
    mock_drone.offboard.set_attitude_rate.reset_mock()

    controller = DroneController(mode="hover")
    controller.pos_ned = MagicMock(north_m=0.0, east_m=0.0, down_m=-10.0)
    controller.vel_ned = MagicMock(north_m_s=0.0, east_m_s=0.0, down_m_s=0.0)
    controller.att_euler = MagicMock(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    controller.is_offboard = True

    await controller.control_step(None)

    mock_solver_instance.solve.assert_called_once()
    # Check target arg
    args = mock_solver_instance.solve.call_args
    target_arg = args[0][1]
    assert target_arg == [0.0, 0.0, -5.0]

    mock_drone.offboard.set_attitude_rate.assert_called_once()
    print("Hover Logic Passed")
