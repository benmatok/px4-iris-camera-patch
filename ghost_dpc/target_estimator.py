import numpy as np

class TargetEstimator:
    def __init__(self, dt=0.05):
        self.dt = dt
        # State: Relative Position [rx, ry, rz] (Sim Frame: X-East, Y-North, Z-Up)
        # We assume the target is roughly stationary in world frame.
        # d(Rel)/dt = V_target - V_drone ~ -V_drone

        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 10.0 # Initial high uncertainty
        self.process_noise = np.eye(3) * 0.1 # Process noise for constant velocity model (unmodeled target motion)
        self.measure_noise = np.eye(3) * 0.5 # Measurement noise

        self.is_lost = True
        self.time_since_last_seen = 0.0

    def reset(self):
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 10.0
        self.is_lost = True
        self.time_since_last_seen = 0.0

    def predict(self, drone_vel):
        """
        Predicts the next state based on drone velocity (dead reckoning).
        Args:
            drone_vel: list or array [vx, vy, vz] (Sim Frame)
        """
        # x_k+1 = x_k - v_drone * dt
        self.state -= np.array(drone_vel) * self.dt

        # P_k+1 = F P_k F.T + Q
        # F is Identity
        self.covariance += self.process_noise

        self.time_since_last_seen += self.dt

        # Mark as lost if not seen for a while (e.g. 0.5s = 10 frames)
        if self.time_since_last_seen > 0.5:
            self.is_lost = True

    def update(self, measurement):
        """
        Updates the state with a measurement.
        Args:
            measurement: list or array [rx, ry, rz] (Relative Sim Frame)
        """
        # If valid measurement received, mark as found
        self.is_lost = False
        self.time_since_last_seen = 0.0

        # Kalman Update
        # z = H x + v
        # H is Identity
        H = np.eye(3)
        R = self.measure_noise
        P = self.covariance

        # Innovation Covariance S = H P H.T + R
        S = P + R

        # Kalman Gain K = P H.T S^-1
        K = P @ np.linalg.inv(S)

        # Innovation y = z - H x
        y = np.array(measurement) - self.state

        # State Update x = x + K y
        self.state += K @ y

        # Covariance Update P = (I - K H) P
        self.covariance = (np.eye(3) - K) @ P

    def get_estimate(self):
        """
        Returns the current estimated relative position [rx, ry, rz].
        """
        return self.state.copy()

    def get_uncertainty(self):
        """
        Returns a scalar metric of uncertainty (trace of covariance).
        """
        return np.trace(self.covariance)
