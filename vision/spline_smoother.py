import numpy as np
from scipy.interpolate import UnivariateSpline
from collections import deque
import logging

logger = logging.getLogger(__name__)

class SplineSmoother:
    def __init__(self, window_size=20, smoothing_factor=None):
        """
        Spline Smoother for 3D Position.
        window_size: Number of points to keep in buffer (e.g. 20 points @ 20Hz = 1s)
        smoothing_factor: 's' parameter for UnivariateSpline. None = default (interpolating).
                          Larger s = more smoothing.
        """
        self.window_size = window_size
        self.s = smoothing_factor
        self.buffer_t = deque(maxlen=window_size)
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
        self.buffer_z = deque(maxlen=window_size)

    def reset(self):
        self.buffer_t.clear()
        self.buffer_x.clear()
        self.buffer_y.clear()
        self.buffer_z.clear()

    def update(self, t, pos):
        """
        Add a measurement.
        pos: [x, y, z]
        """
        self.buffer_t.append(t)
        self.buffer_x.append(pos[0])
        self.buffer_y.append(pos[1])
        self.buffer_z.append(pos[2])

    def get_state(self, t_eval):
        """
        Get smoothed Position and Velocity at t_eval.
        Returns:
            pos: [x, y, z]
            vel: [vx, vy, vz]
        """
        if len(self.buffer_t) < 4:
            # Not enough points for cubic spline (needs k+1=4 points)
            # Fallback to last measurement or linear
            if len(self.buffer_t) > 1:
                dt = self.buffer_t[-1] - self.buffer_t[-2]
                vx = (self.buffer_x[-1] - self.buffer_x[-2]) / dt if dt > 0 else 0.0
                vy = (self.buffer_y[-1] - self.buffer_y[-2]) / dt if dt > 0 else 0.0
                vz = (self.buffer_z[-1] - self.buffer_z[-2]) / dt if dt > 0 else 0.0
                return [self.buffer_x[-1], self.buffer_y[-1], self.buffer_z[-1]], [vx, vy, vz]
            elif len(self.buffer_t) == 1:
                return [self.buffer_x[-1], self.buffer_y[-1], self.buffer_z[-1]], [0.0, 0.0, 0.0]
            else:
                return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        t_arr = np.array(self.buffer_t)

        # Fit Splines
        try:
            # Use k=3 (Cubic)
            # w=None (Equal weights)
            spl_x = UnivariateSpline(t_arr, list(self.buffer_x), k=3, s=self.s)
            spl_y = UnivariateSpline(t_arr, list(self.buffer_y), k=3, s=self.s)
            spl_z = UnivariateSpline(t_arr, list(self.buffer_z), k=3, s=self.s)

            # Evaluate
            x = float(spl_x(t_eval))
            y = float(spl_y(t_eval))
            z = float(spl_z(t_eval))

            vx = float(spl_x.derivative()(t_eval))
            vy = float(spl_y.derivative()(t_eval))
            vz = float(spl_z.derivative()(t_eval))

            return [x, y, z], [vx, vy, vz]

        except Exception as e:
            logger.error(f"Spline fitting failed: {e}")
            # Fallback
            return [self.buffer_x[-1], self.buffer_y[-1], self.buffer_z[-1]], [0.0, 0.0, 0.0]
