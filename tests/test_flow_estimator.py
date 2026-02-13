import sys
import os
import numpy as np
import pytest

# Add repo root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from vision.projection import Projector
from vision.flow_estimator import FlowVelocityEstimator

def get_rotation_matrix(roll, pitch, yaw):
    cphi, sphi = np.cos(roll), np.sin(roll)
    ctheta, stheta = np.cos(pitch), np.sin(pitch)
    cpsi, spsi = np.cos(yaw), np.sin(yaw)

    r11 = ctheta * cpsi
    r12 = cpsi * sphi * stheta - cphi * spsi
    r13 = sphi * spsi + cphi * cpsi * stheta

    r21 = ctheta * spsi
    r22 = cphi * cpsi + sphi * spsi * stheta
    r23 = cphi * spsi * stheta - cpsi * sphi

    r31 = -stheta
    r32 = ctheta * sphi
    r33 = cphi * ctheta

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

def test_flow_estimator_initialization():
    proj = Projector(width=640, height=480, fov_deg=60.0)
    est = FlowVelocityEstimator(proj, num_points=100)
    assert est is not None
    assert est.num_points == 100

def test_pure_forward_motion():
    # Camera tilted 0 deg
    proj = Projector(width=640, height=480, fov_deg=90.0, tilt_deg=0.0)
    est = FlowVelocityEstimator(proj, num_points=200)

    # Pitch down 20 deg to see ground
    pitch_rad = np.deg2rad(-20.0)

    state1 = {
        'px': 0.0, 'py': 0.0, 'pz': -100.0,
        'roll': 0.0, 'pitch': pitch_rad, 'yaw': 0.0
    }

    # Initialize
    # Try a few times to fill points
    for _ in range(5):
        est.update(state1, (0,0,0), 0.1)

    print(f"Points after init: {len(est.world_points)}")
    assert len(est.world_points) > 50

    # Move Forward in Body X
    # V_b = [1, 0, 0]
    R_b2w = get_rotation_matrix(0, pitch_rad, 0)
    move_ned = R_b2w @ np.array([1.0, 0, 0])

    state2 = state1.copy()
    state2['px'] += move_ned[0]
    state2['py'] += move_ned[1]
    state2['pz'] += move_ned[2]

    # No rotation
    foe = est.update(state2, (0,0,0), 0.1)

    assert foe is not None
    u, v = foe
    print(f"FOE (Tilt 0): {u}, {v}")

    # Expected FOE: Center (0,0) because Camera matches Body (Tilt 0) and motion is Body X
    assert abs(u) < 0.1
    assert abs(v) < 0.1

def test_tilted_motion():
    # Tilt 30 deg Up
    proj = Projector(width=640, height=480, fov_deg=90.0, tilt_deg=30.0)
    est = FlowVelocityEstimator(proj, num_points=200)

    # Pitch down 45 deg to see ground
    # Effective pitch = -45 + 30 = -15 deg (looking down)
    pitch_rad = np.deg2rad(-45.0)

    state1 = {
        'px': 0.0, 'py': 0.0, 'pz': -100.0,
        'roll': 0.0, 'pitch': pitch_rad, 'yaw': 0.0
    }

    # Initialize
    for _ in range(5):
        est.update(state1, (0,0,0), 0.1)

    print(f"Points after init (Tilted): {len(est.world_points)}")
    assert len(est.world_points) > 50

    # Move Forward in Body X
    R_b2w = get_rotation_matrix(0, pitch_rad, 0)
    move_ned = R_b2w @ np.array([1.0, 0, 0])

    state2 = state1.copy()
    state2['px'] += move_ned[0]
    state2['py'] += move_ned[1]
    state2['pz'] += move_ned[2]

    foe = est.update(state2, (0,0,0), 0.1)

    assert foe is not None
    u, v = foe

    # Expected FOE
    # V_b = [1, 0, 0]
    # Camera Tilt = 30 deg (Up)
    # R_c2b.T @ V_b = [0, sin30, cos30] = [0, 0.5, 0.866]
    # FOE = (0/0.866, 0.5/0.866) = (0, 0.577)

    expected_v = np.tan(np.deg2rad(30.0))
    print(f"Expected V: {expected_v}, Got V: {v}")

    assert abs(u) < 0.1
    assert abs(v - expected_v) < 0.1

def test_continuous_motion():
    # Test that points are replenished as we move
    proj = Projector(width=640, height=480, fov_deg=90.0, tilt_deg=0.0)
    est = FlowVelocityEstimator(proj, num_points=100)

    # Pitch down 45 deg
    pitch_rad = np.deg2rad(-45.0)
    state = {
        'px': 0.0, 'py': 0.0, 'pz': -100.0,
        'roll': 0.0, 'pitch': pitch_rad, 'yaw': 0.0
    }

    # Init
    est.update(state, (0,0,0), 0.1)

    # Move forward 50m
    R_b2w = get_rotation_matrix(0, pitch_rad, 0)
    vel_ned = R_b2w @ np.array([10.0, 0, 0])
    dt = 0.1

    for i in range(50):
        state['px'] += vel_ned[0] * dt
        state['py'] += vel_ned[1] * dt
        state['pz'] += vel_ned[2] * dt

        est.update(state, (0,0,0), dt)

        if i % 10 == 0:
            print(f"Step {i}: Points {len(est.world_points)}")
            assert len(est.world_points) > 20 # Should maintain count

if __name__ == "__main__":
    test_pure_forward_motion()
    test_tilted_motion()
    test_continuous_motion()
