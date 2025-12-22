import unittest
import numpy as np
import cv2
import time
import sys

# Try imports
try:
    from drone_env.tracker import TextureTracker
    TEXTURE_TRACKER_AVAILABLE = True
except ImportError:
    TEXTURE_TRACKER_AVAILABLE = False

CSRT_AVAILABLE = hasattr(cv2, 'TrackerCSRT_create')

class TestTrackerComparison(unittest.TestCase):
    def setUp(self):
        self.width = 640
        self.height = 480
        self.num_frames = 100
        self.target_size = 64
        self.bg_color = 100

        # Generate synthetic data
        self.frames = []
        self.ground_truth = []

        # Trajectory: Figure 8
        t = np.linspace(0, 2 * np.pi, self.num_frames)
        center_x = self.width // 2
        center_y = self.height // 2
        radius_x = 150
        radius_y = 100

        xs = center_x + radius_x * np.sin(t)
        ys = center_y + radius_y * np.sin(t) * np.cos(t)

        # Constant texture for the target
        self.target_texture = np.random.randint(0, 255, (self.target_size, self.target_size, 3), dtype=np.uint8)

        for i in range(self.num_frames):
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * self.bg_color

            x = int(xs[i]) - self.target_size // 2
            y = int(ys[i]) - self.target_size // 2

            # Ensure within bounds for simplicity of GT generation, though tracking should handle partial
            x = max(0, min(self.width - self.target_size, x))
            y = max(0, min(self.height - self.target_size, y))

            # Paste texture
            img[y:y+self.target_size, x:x+self.target_size] = self.target_texture

            # Add noise
            noise = np.random.normal(0, 10, img.shape).astype(np.int16)
            img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            self.frames.append(img_noisy)
            self.ground_truth.append((x, y, self.target_size, self.target_size))

    def calculate_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def run_tracker(self, tracker, name):
        ious = []
        center_errors = []

        start_time = time.time()

        # Init
        bbox = self.ground_truth[0]
        tracker.init(self.frames[0], bbox)

        for i in range(1, self.num_frames):
            success, bbox_pred = tracker.update(self.frames[i])

            if success:
                gt = self.ground_truth[i]
                iou = self.calculate_iou(bbox_pred, gt)
                ious.append(iou)

                center_pred = (bbox_pred[0] + bbox_pred[2]/2, bbox_pred[1] + bbox_pred[3]/2)
                center_gt = (gt[0] + gt[2]/2, gt[1] + gt[3]/2)
                dist = np.sqrt((center_pred[0] - center_gt[0])**2 + (center_pred[1] - center_gt[1])**2)
                center_errors.append(dist)
            else:
                ious.append(0.0)
                center_errors.append(-1.0) # Lost

        end_time = time.time()
        fps = (self.num_frames - 1) / (end_time - start_time)

        avg_iou = np.mean(ious)
        valid_errors = [e for e in center_errors if e >= 0]
        avg_error = np.mean(valid_errors) if valid_errors else float('inf')

        print(f"[{name}] FPS: {fps:.2f}, Avg IoU: {avg_iou:.4f}, Avg Center Error: {avg_error:.2f}")
        return avg_iou, avg_error, fps

    def test_compare_trackers(self):
        print("\n--- Tracker Comparison ---")

        if not TEXTURE_TRACKER_AVAILABLE:
            print("TextureTracker not available, skipping comparison.")
            return

        print("Running TextureTracker...")
        tt = TextureTracker()
        tt_iou, tt_err, tt_fps = self.run_tracker(tt, "TextureTracker")

        if CSRT_AVAILABLE:
            print("Running CSRT...")
            csrt = cv2.TrackerCSRT_create()
            csrt_iou, csrt_err, csrt_fps = self.run_tracker(csrt, "CSRT")

            # Assert that our tracker is somewhat competitive
            # Note: CSRT is very robust but slow. TextureTracker (MOSSE/KCF based) should be faster but maybe less accurate.
            # But here we just check it works reasonable well on synthetic simple data.

            self.assertGreater(tt_iou, 0.5, "TextureTracker IoU is too low")
            self.assertLess(tt_err, 20.0, "TextureTracker error is too high")

            print(f"Speedup vs CSRT: {tt_fps/csrt_fps:.2f}x")
        else:
            print("CSRT not available, skipping CSRT run.")
            self.assertGreater(tt_iou, 0.5)

if __name__ == '__main__':
    unittest.main()
