import numpy as np
import sys
import os

# Add root to path
import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.projection import Projector

def test_inverse():
    # Setup
    width = 640
    height = 480
    proj = Projector(width=width, height=height, fov_deg=60.0, tilt_deg=30.0)

    # State: Drone at 50m alt, pitched down 45 deg to see ground
    # NED: z = -50. Pitch = -45 deg (-pi/4).
    state = {
        'px': 0.0, 'py': 0.0, 'pz': -50.0,
        'roll': 0.0,
        'pitch': np.deg2rad(-45.0),
        'yaw': 0.0
    }

    # Pick a pixel
    u = 320.0
    v = 240.0 # Center

    # 1. Pixel to World
    world_pt = proj.pixel_to_world(u, v, state)
    print(f"Pixel ({u}, {v}) -> World {world_pt}")

    if world_pt is None:
        print("Error: Ray did not intersect ground.")
        sys.exit(1)

    # 2. World to Pixel
    wx, wy, wz = world_pt
    uv_new = proj.world_to_pixel(wx, wy, wz, state)
    print(f"World {world_pt} -> Pixel {uv_new}")

    if uv_new is None:
        print("Error: Point projected behind camera?")
        sys.exit(1)

    u_new, v_new = uv_new

    # Check
    err_u = abs(u - u_new)
    err_v = abs(v - v_new)

    print(f"Errors: u={err_u:.4f}, v={err_v:.4f}")

    if err_u < 1.0 and err_v < 1.0:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)

def test_off_center():
    # Setup
    width = 640
    height = 480
    proj = Projector(width=width, height=height, fov_deg=60.0, tilt_deg=0.0)

    # State: Drone level at 10m
    state = {
        'px': 0.0, 'py': 0.0, 'pz': -10.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }

    # Target at (30, 5, 0) -> Angle down = atan(10/30) = 18.4 deg. Inside VFOV (23.4).
    target = (30.0, 5.0, 0.0)

    uv = proj.world_to_pixel(*target, state)
    print(f"Target {target} -> Pixel {uv}")

    if uv is None:
        print("FAIL: Should be visible")
        sys.exit(1)

    u, v = uv

    # Check bounds
    if 0 <= u <= width and 0 <= v <= height:
        print("PASS: In frame")
    else:
        print(f"FAIL: Out of frame {u}, {v}")
        sys.exit(1)

    # Round trip
    xyz_new = proj.pixel_to_world(u, v, state)
    print(f"Pixel {uv} -> World {xyz_new}")

    err_x = abs(target[0] - xyz_new[0])
    err_y = abs(target[1] - xyz_new[1])

    print(f"Errors: x={err_x:.4f}, y={err_y:.4f}")
    if err_x < 0.01 and err_y < 0.01:
        print("PASS Round Trip")
    else:
        print("FAIL Round Trip")
        sys.exit(1)

if __name__ == "__main__":
    test_inverse()
    test_off_center()
