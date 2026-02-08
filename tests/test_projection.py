import numpy as np
import sys
import os

import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision.projection import Projector

def test_projection():
    print("Testing Projector...")

    # Init Projector
    projector = Projector(width=640, height=480, fov_deg=60.0)

    # Case 1: Drone @ (0,0,-10) [Altitude 10m], Pitch=-90 (Nose Down)
    # Looking straight down. Center pixel should map to (0,0,0)
    state1 = {
        'px': 0.0, 'py': 0.0, 'pz': -10.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': -np.pi/2, 'yaw': 0.0
    }

    # Center Pixel
    u, v = 320.0, 240.0

    res1 = projector.pixel_to_world(u, v, state1)
    print(f"Case 1 (Straight Down): {res1}")

    if res1 is None:
        print("FAIL: No intersection for Case 1")
        sys.exit(1)

    if np.linalg.norm(np.array(res1)) > 0.01:
        print(f"FAIL: Expected (0,0,0), got {res1}")
        sys.exit(1)

    # Case 2: Drone @ (0,0,-10), Pitch=-45 (Nose Down 45)
    # Should map to (+X, 0, 0)
    # Expected X = 10.0
    state2 = state1.copy()
    state2['pitch'] = -np.pi/4

    res2 = projector.pixel_to_world(u, v, state2)
    print(f"Case 2 (45 deg Down): {res2}")

    if res2 is None:
        print("FAIL: No intersection for Case 2")
        sys.exit(1)

    expected_x = 10.0
    if abs(res2[0] - expected_x) > 0.1 or abs(res2[1]) > 0.1:
        print(f"FAIL: Expected ({expected_x}, 0, 0), got {res2}")
        sys.exit(1)

    # Case 3: Corner Case
    # Drone Level (Pitch=0). Camera Forward.
    # Pixel: Bottom Center (320, 480).
    # Camera FOV 60 Vertical (Assuming fx=fy, FOV_H=60, Aspect 4:3 -> FOV_V approx 45?)
    # Wait, FOV_H = 60.
    # fx = (640/2) / tan(30) = 554.25
    # cy = 240.
    # v = 480. y_c = (480-240)/554.25 = 240/554.25 = 0.433.
    # vec_c = [x, 0.433, 1].
    # vec_b = [1, x, 0.433]. (X_b = Z_c=1. Y_b = X_c. Z_b = Y_c=0.433).
    # Ray points Forward (1) and Down (0.433).
    # Drone @ (0,0,-10).
    # P(t) = (0,0,-10) + t(1, 0, 0.433).
    # Z(t) = -10 + 0.433 t = 0 => t = 10/0.433 = 23.09.
    # X(t) = 23.09.

    state3 = state1.copy()
    state3['pitch'] = 0.0
    u, v = 320.0, 480.0 # Bottom Center

    res3 = projector.pixel_to_world(u, v, state3)
    print(f"Case 3 (Level, Bottom Pixel): {res3}")

    if res3 is None:
        print("FAIL: No intersection for Case 3")
        sys.exit(1)

    if abs(res3[0] - 23.09) > 1.0:
         print(f"FAIL: Expected X ~ 23.09, got {res3[0]}")
         sys.exit(1)

    print("PASS: Projection logic verified.")

if __name__ == "__main__":
    test_projection()
