import logging
from vision.detector import RedObjectDetector

logger = logging.getLogger(__name__)

class VisualTracker:
    def __init__(self, projector):
        self.projector = projector
        self.detector = RedObjectDetector()

    def process(self, image, drone_state_ned):
        """
        Processes the image to detect the target and computes its world position.

        Args:
            image: numpy array (BGR)
            drone_state_ned: dict with px, py, pz, roll, pitch, yaw (NED frame)

        Returns:
            center_pixel: (u, v) or None
            target_world_pos: [x, y, z] or None
            radius: float (pixels) or 0.0
        """
        if image is None:
            return None, None, 0.0

        center, area, _ = self.detector.detect(image)
        radius = 0.0
        if area > 0:
            import math
            radius = math.sqrt(area / math.pi)

        target_world_pos = None
        if center:
            # Project pixel to world (assuming target on ground/surface or ray intersection)
            # The original logic used projector.pixel_to_world
            target_world_pos = self.projector.pixel_to_world(center[0], center[1], drone_state_ned)

        return center, target_world_pos, radius
