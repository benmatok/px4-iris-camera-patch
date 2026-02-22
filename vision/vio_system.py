from vision.feature_tracker import FeatureTracker
from vision.sliding_window_estimator import SlidingWindowEstimator
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VIOSystem:
    def __init__(self, projector, config=None):
        self.projector = projector
        self.config = config

        # Components
        self.tracker = FeatureTracker(projector, config)
        # Larger window for 5s horizon (15 frames: 3 dense + 12 sparse * 0.4s = 4.8s + buffer)
        self.estimator = SlidingWindowEstimator(window_size=15)

        self.initialized = False
        self.imu_buffer = [] # Buffer for current interval
        self.last_time = 0.0

        # Latest State Cache
        self.state_cache = {
            'px': 0.0, 'py': 0.0, 'pz': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0
        }

    def initialize(self, q, p, v):
        # Initial Frame
        self.estimator.add_frame(0.0, p, q, v, [], {})
        self.initialized = True
        self.state_cache.update({
            'px': p[0], 'py': p[1], 'pz': p[2],
            'vx': v[0], 'vy': v[1], 'vz': v[2]
        })
        logger.info("VIOSystem Initialized (BA Backend)")

    def propagate(self, gyro, accel, dt):
        """
        Buffer IMU data and propagate state cache.
        """
        if not self.initialized: return
        self.imu_buffer.append((dt, accel, gyro))

        # Simple Propagate State Cache for high-rate feedback
        # This is "blind" propagation until next optimization
        # NED Gravity
        g = np.array([0, 0, 9.81])

        from scipy.spatial.transform import Rotation as R
        if 'q' not in self.state_cache:
            last = self.estimator.get_latest_state()
            if last:
                self.state_cache['q'] = last['q']
            else:
                return

        # Propagate Cache
        q_curr = self.state_cache.get('q', np.array([0,0,0,1]))
        v_curr = np.array([self.state_cache['vx'], self.state_cache['vy'], self.state_cache['vz']])
        p_curr = np.array([self.state_cache['px'], self.state_cache['py'], self.state_cache['pz']])

        # Accel in world
        R_curr = R.from_quat(q_curr)
        acc_w = R_curr.apply(accel) + g

        v_next = v_curr + acc_w * dt
        p_next = p_curr + v_curr * dt + 0.5 * acc_w * dt**2

        dq = R.from_rotvec(gyro * dt)
        q_next = (R_curr * dq).as_quat()

        self.state_cache.update({
            'px': p_next[0], 'py': p_next[1], 'pz': p_next[2],
            'vx': v_next[0], 'vy': v_next[1], 'vz': v_next[2],
            'q': q_next
        })

    def track_features(self, state_ned, body_rates, dt):
        """
        Wrapper for tracker update.
        """
        if not self.initialized: return
        # Mock clone_idx for tracker (backend doesn't use it in this mode, but tracker expects it)
        clone_idx = 0
        self.tracker.update(state_ned, body_rates, dt, clone_idx)

        # Simple Propagate State Cache for high-rate feedback
        # This is "blind" propagation until next optimization
        # NED Gravity
        g = np.array([0, 0, 9.81])

        from scipy.spatial.transform import Rotation as R
        # We need current orientation. Where is it?
        # Get from last optimized + integration?
        # Or just integrate from cache?
        # Let's integrate from cache. We need to store 'q' in cache.
        if 'q' not in self.state_cache:
            # Need to fetch from estimator if init happened
            last = self.estimator.get_latest_state()
            if last:
                self.state_cache['q'] = last['q']
            else:
                return # Should be init

        # Propagate Cache
        q_curr = self.state_cache.get('q', np.array([0,0,0,1]))
        v_curr = np.array([self.state_cache['vx'], self.state_cache['vy'], self.state_cache['vz']])
        p_curr = np.array([self.state_cache['px'], self.state_cache['py'], self.state_cache['pz']])

        # Simple Propagate State Cache for high-rate feedback was corrupted here.
        # We should not propagate in track_features() unless we have accel.
        # track_features receives body_rates but NOT accel.
        # So we cannot propagate position/velocity here.
        # We should rely on propagate() for prediction.
        # And track_features just updates the tracker.
        pass

    def update_measurements(self, height_meas, vz_meas, tracks, velocity_prior=None):
        """
        In BA system, we treat this as "Keyframe Arrival".
        1. Process Tracks -> Observations
        2. Add Frame to Backend
        3. Solve
        """
        if not self.initialized: return

        # Convert tracks to {id: (u, v)}
        # Tracks format: list of dicts with 'obs'.
        # Wait, MSCKF expects 'finished_tracks'.
        # BA expects 'current_observations'.
        # We need the tracker to give us *all currently tracked points*, not just finished ones.
        # FeatureTracker.update returns finished_tracks.
        # We need access to `active_tracks` or `curr_projections` from tracker.

        # Hack: The tracker has `prev_projections` which are current valid points.
        # Let's use that.
        # Or modify tracker to return active tracks.
        # Let's use `prev_projections` (it's populated in update()).

        image_obs = self.tracker.prev_projections.copy()

        # Add Frame
        # Timestamp? Just increment or use accumulated dt?
        # self.estimator expects absolute t? Used for ordering.
        t_now = self.estimator.frames[-1]['t'] + sum(x[0] for x in self.imu_buffer)

        # Priors: We don't have external priors except VIO prop.
        # We pass None to let estimator propagate using IMU.

        # But we need to pass Barometer.
        # height_meas is NED Pz.
        # Baro is Altitude = -Pz.

        baro_val = None
        if height_meas is not None:
            baro_val = -height_meas

        # We let p_prior be None (Estimator propagates)
        # We pass baro_val explicitly

        self.estimator.add_frame(
            t_now,
            None, # p_prior (None -> propagate)
            None, # q_prior
            None, # v_prior
            self.imu_buffer,
            image_obs,
            vel_prior=velocity_prior,
            baro=baro_val
        )

        # Clear Buffer
        self.imu_buffer = []

        # Optimization
        self.estimator.solve()

        # Update Cache
        latest = self.estimator.get_latest_state()
        if latest:
            self.state_cache.update(latest)

    def is_reliable(self):
        # BA usually robust if enough points
        return self.initialized and len(self.estimator.points) > 5

    def get_state_dict(self):
        return self.state_cache.copy()

    # Bridge needed methods
    @property
    def cam_clones(self):
        # Dummy for FeatureTracker which expects clone_idx
        # FeatureTracker uses `cam_clones` to get ID?
        # Actually FeatureTracker just takes `clone_idx` as arg in `update`.
        # The external caller (theshow) passes `msckf.cam_clones[-1]['id']`.
        # We need to simulate this.
        # Return a list of dicts with 'id'.
        return [{'id': len(self.estimator.frames)}]

    @property
    def q(self):
        return self.state_cache.get('q', np.array([0,0,0,1]))
