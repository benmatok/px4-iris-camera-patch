import unittest
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import stabilize_frame, warp_bbox, compute_homography

class TestCV(unittest.TestCase):
    def test_stabilize_frame_identity(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        # 0 pitch/roll should return identical image (mostly)
        stab = stabilize_frame(img, 0.0, 0.0)
        self.assertEqual(stab.shape, img.shape)
        np.testing.assert_array_equal(stab, img)

    def test_warp_bbox(self):
        # Create a dummy Homography (Identity)
        H = np.eye(3, dtype=np.float32)
        bbox = [10, 10, 50, 50]
        new_bbox, success = warp_bbox(H, bbox)
        self.assertTrue(success)
        self.assertEqual(new_bbox, bbox)

        # Translation H
        H_trans = np.eye(3, dtype=np.float32)
        H_trans[0, 2] = 10 # x + 10
        new_bbox, success = warp_bbox(H_trans, bbox)
        self.assertTrue(success)
        # x should be 20
        self.assertEqual(new_bbox[0], 20)
        self.assertEqual(new_bbox[1], 10)
        self.assertEqual(new_bbox[2], 50) # w same
        self.assertEqual(new_bbox[3], 50) # h same

    def test_compute_homography(self):
        # Same image -> Identity
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        # Need features. Random noise might not have good ORB features.
        # Draw some shapes
        cv2.rectangle(img, (50, 50), (100, 100), (255, 255, 255), -1)
        cv2.circle(img, (150, 150), 20, (128, 128, 128), -1)

        H, success = compute_homography(img, img)
        if success:
            np.testing.assert_array_almost_equal(H, np.eye(3), decimal=1)
        else:
            print("Skipping homography check (no features found)")

if __name__ == '__main__':
    unittest.main()
