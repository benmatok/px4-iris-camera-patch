import cv2
import numpy as np

class RedObjectDetector:
    def __init__(self):
        # HSV ranges for Red
        # Lower Red: H=0-10
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])

        # Upper Red: H=170-180
        self.lower_red2 = np.array([170, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

    def detect(self, image):
        """
        Detects the largest red blob in the image.
        Args:
            image: BGR image (numpy array)
        Returns:
            (u, v): Center coordinates (float), or None if not found
            area: Area of the blob, or 0 if not found
            bbox: (x, y, w, h) bounding box, or None if not found
        """
        if image is None:
            return None, 0, None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0, None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 1: # Minimum area threshold
            return None, 0, None

        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate center
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + w/2, y + h/2

        return (cx, cy), area, (x, y, w, h)

if __name__ == "__main__":
    print("Testing RedObjectDetector...")

    # Create synthetic image (black background, red circle)
    height, width = 480, 640
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw red circle at (320, 240) radius 50
    # OpenCV BGR: Red is (0, 0, 255)
    cv2.circle(img, (320, 240), 50, (0, 0, 255), -1)

    detector = RedObjectDetector()
    center, area, bbox = detector.detect(img)

    print(f"Detected Center: {center}")
    print(f"Detected Area: {area}")
    print(f"Bounding Box: {bbox}")

    expected_center = (320, 240)
    expected_area = np.pi * 50**2

    if center is None:
        print("FAIL: No object detected.")
        exit(1)

    dist = np.sqrt((center[0]-expected_center[0])**2 + (center[1]-expected_center[1])**2)
    if dist > 5.0:
        print(f"FAIL: Center too far. Dist={dist}")
        exit(1)

    if abs(area - expected_area) > 0.1 * expected_area:
        print(f"FAIL: Area mismatch. Got {area}, expected {expected_area}")
        exit(1)

    print("PASS: Detector works on synthetic image.")
