import unittest
import numpy as np
import cv2
import os

# Try importing the tracker. If compilation failed, this will fail.
try:
    from drone_env.tracker import TextureTracker
    TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    TRACKER_AVAILABLE = False
except Exception as e:
    print(f"Exception: {e}")
    TRACKER_AVAILABLE = False

class TestTextureTracker(unittest.TestCase):
    def setUp(self):
        if not TRACKER_AVAILABLE:
            self.skipTest("TextureTracker extension not available (needs compilation)")

        # Create a synthetic image sequence
        self.width = 640
        self.height = 480
        self.bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100

        # Draw a moving square
        self.target_size = 50
        self.start_x = 100
        self.start_y = 100
        self.velocity_x = 2
        self.velocity_y = 1

    def get_frame(self, frame_idx):
        img = self.bg.copy()
        x = int(self.start_x + self.velocity_x * frame_idx)
        y = int(self.start_y + self.velocity_y * frame_idx)

        # Draw target with texture
        cv2.rectangle(img, (x, y), (x+self.target_size, y+self.target_size), (200, 50, 50), -1)
        # Add some noise/texture inside
        noise = np.random.randint(0, 50, (self.target_size, self.target_size, 3), dtype=np.uint8)
        img[y:y+self.target_size, x:x+self.target_size] += noise

        return img, (x, y, self.target_size, self.target_size)

    def test_tracker_lifecycle(self):
        tracker = TextureTracker()

        # Init
        img0, bbox0 = self.get_frame(0)
        success = tracker.init(img0, bbox0)
        self.assertTrue(success, "Tracker init failed")

        # Update for 10 frames
        for i in range(1, 11):
            img, expected_bbox = self.get_frame(i)
            success, bbox = tracker.update(img)
            self.assertTrue(success, f"Tracker update failed at frame {i}")

            # Check accuracy roughly
            ex, ey, ew, eh = expected_bbox
            tx, ty, tw, th = bbox

            center_dist = np.sqrt((ex-tx)**2 + (ey-ty)**2)
            self.assertLess(center_dist, 10.0, f"Tracker drifted too much at frame {i}: dist={center_dist}")

if __name__ == '__main__':
    unittest.main()
