import logging
from vision.detector import RedObjectDetector

logger = logging.getLogger(__name__)

class VisualTracker:
    def __init__(self, projector):
        self.projector = projector
        self.detector = RedObjectDetector()

    def process(self, image, drone_state_ned, ground_truth_target_pos=None):
        """
        Processes the image to detect the target and computes its world position.

        Args:
            image: numpy array (BGR)
            drone_state_ned: dict with px, py, pz, roll, pitch, yaw (NED frame)
            ground_truth_target_pos: [x, y, z] (World Frame) or None (Optional)

        Returns:
            center_pixel: (u, v) or None
            target_world_pos: [x, y, z] (Relative NED if px=0 in state) or None
            radius: float (pixels) or 0.0
        """
        import math

        if ground_truth_target_pos:
            # Bypass Detection: Use Perfect Projection (Oracle)
            tx, ty, tz = ground_truth_target_pos

            # 1. Project to Pixel
            logger.debug(f"Projecting GT Target: {tx}, {ty}, {tz} with Drone State: {drone_state_ned}")
            uv = self.projector.world_to_pixel(tx, ty, tz, drone_state_ned)
            logger.debug(f"Projected UV: {uv}")

            if uv:
                u, v = uv
                # Check if inside image bounds
                if 0 <= u < self.projector.width and 0 <= v < self.projector.height:
                    center = (u, v)

                    # 2. Compute Relative Target Position (NED)
                    # drone_state_ned typically has px=0, py=0, pz=-alt for relative tracking
                    # Target (NED) = [Tx, Ty, -Tz]
                    # Drone (NED) = [Dx, Dy, -Dz]
                    # Rel = Target - Drone

                    # Target Absolute NED
                    t_ned_x = tx
                    t_ned_y = ty
                    t_ned_z = -tz

                    # Drone Absolute NED
                    d_ned_x = drone_state_ned['px']
                    d_ned_y = drone_state_ned['py']
                    d_ned_z = drone_state_ned['pz']

                    target_world_pos = [
                        t_ned_x - d_ned_x,
                        t_ned_y - d_ned_y,
                        t_ned_z - d_ned_z
                    ]

                    # 3. Analytic Radius
                    # r = f * R_obj / Dist
                    # Dist = sqrt(dx^2 + dy^2 + dz^2)
                    dx = target_world_pos[0]
                    dy = target_world_pos[1]
                    dz = target_world_pos[2]
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                    # Focal Length (Pixels)
                    f = self.projector.fx

                    r_obj = 0.5 # 0.5m radius
                    if dist > 0.1:
                        radius = f * r_obj / dist
                    else:
                        radius = 100.0 # Huge

                    return center, target_world_pos, radius

            # If not in view, return None
            return None, None, 0.0

        # Fallback to Detector Logic
        if image is None:
            return None, None, 0.0

        center, area, _ = self.detector.detect(image)
        radius = 0.0
        if area > 0:
            radius = math.sqrt(area / math.pi)

        target_world_pos = None
        if center:
            # Project pixel to world (assuming target on ground/surface or ray intersection)
            # The original logic used projector.pixel_to_world
            target_world_pos = self.projector.pixel_to_world(center[0], center[1], drone_state_ned)

        return center, target_world_pos, radius
