import numpy as np
import drone_env.drone as drone

class DroneSim:
    """
    Wrapper around DroneEnv to provide a simplified single-agent interface
    for the Oracle/Training pipeline.
    """
    def __init__(self):
        # Initialize single agent environment
        self.env = drone.DroneEnv(num_agents=1, episode_length=1000) # Long episode for free simulation
        self.env.reset_all_envs()

        # Shortcuts to data buffers
        self.data = self.env.data_dictionary
        self.pos_x = self.data["pos_x"]
        self.pos_y = self.data["pos_y"]
        self.pos_z = self.data["pos_z"]
        self.vel_x = self.data["vel_x"]
        self.vel_y = self.data["vel_y"]
        self.vel_z = self.data["vel_z"]
        self.roll = self.data["roll"]
        self.pitch = self.data["pitch"]
        self.yaw = self.data["yaw"]

        # Step counter
        self.step_counts = self.data["step_counts"]

    def set_state(self, state):
        """
        Sets the state of the drone.
        State format: [px, py, pz, vx, vy, vz, r, p, y, wr, wp, wy]
        Note: The underlying environment does not explicitly store angular velocities (wr, wp, wy)
        as state variables for integration (it uses command history or assumes instantaneous rate tracking
        in the simplified model, but the scalar implementation updates r, p, y using commands).

        The provided core code expects 12 dims.
        The physics engine (drone_cython) integrates:
        r += roll_rate_cmd * dt

        So the "state" angular velocities are actually the previous commands?
        Or we just ignore them if the model assumes rate control?

        The Oracle seems to optimize for a state that includes angular velocities.
        However, our physics model is:
        Rates are INPUTS (Actions).
        State is Pos + Vel + Orientation.

        If the Oracle assumes 12-dim state including angular rates, we might need to mock them
        or map them to the previous action.

        Let's assume the 12-dim state is:
        [x, y, z, vx, vy, vz, r, p, y, wr, wp, wy]

        Our physics engine doesn't track angular acceleration, so wr, wp, wy are not state variables
        in the strict sense (they don't persist unless we hold the command).
        """
        self.pos_x[0] = state[0]
        self.pos_y[0] = state[1]
        self.pos_z[0] = state[2]
        self.vel_x[0] = state[3]
        self.vel_y[0] = state[4]
        self.vel_z[0] = state[5]
        self.roll[0] = state[6]
        self.pitch[0] = state[7]
        self.yaw[0] = state[8]
        # Ignore 9, 10, 11 (angular rates) as they are not state vars in this physics model

        self.step_counts[0] = 0

    def get_state(self):
        """
        Returns 12-dim state.
        For angular rates, we'll return 0 or the last command if we tracked it.
        For now, returning 0s for angular rates as they are instantaneous in this model.
        """
        s = np.zeros(12, dtype=np.float32)
        s[0] = self.pos_x[0]
        s[1] = self.pos_y[0]
        s[2] = self.pos_z[0]
        s[3] = self.vel_x[0]
        s[4] = self.vel_y[0]
        s[5] = self.vel_z[0]
        s[6] = self.roll[0]
        s[7] = self.pitch[0]
        s[8] = self.yaw[0]
        # s[9..11] remain 0
        return s

    def step(self, action):
        """
        Action: [thrust, roll_rate, pitch_rate, yaw_rate]
        """
        # Set action in buffer
        # Actions array is (num_agents * 4)
        self.data["actions"][:] = action.astype(np.float32)

        # Call step
        # We need to construct the kwargs for the step function
        kwargs = self.env.get_step_function_kwargs()
        args = {}
        for k, v in kwargs.items():
            if v in self.data:
                args[k] = self.data[v]
            elif k == "num_agents":
                args[k] = 1
            elif k == "episode_length":
                args[k] = 1000
            elif k == "env_ids": # Not strictly needed for cpu but kept for signature
                 args[k] = self.data.get("env_ids", np.zeros(1, dtype=np.int32))

        self.env.step_function(**args)

    def get_history_buffer(self):
        """
        Returns the history buffer in the format expected by ChebyshevHistoryEncoder.
        (Batch=1, Channels=12, Time=40)

        Our env stores history in `pos_history` (T, N, 3) which is just position.
        The `observations` array stores sliding window history of 10 features:
        [roll, pitch, yaw, z, thrust, roll_rate, pitch_rate, yaw_rate, u, v]
        Length 30 (3 seconds).

        The user's code expects 40 steps (4 seconds) and 12 channels.

        We need to bridge this.
        Option 1: Pad the existing history.
        Option 2: Use the existing 10 channels and adapt the model.

        The user specified `state_dim=12` in `JulesPolicy`.
        And `history_dim=12`.

        We should try to map our environment data to this 12-channel format.
        Env Channels (10): r, p, y, z, T, wr, wp, wy, u, v
        Missing: x, y (absolute position is usually not in history for RL, but relative might be).

        Let's construct a 12-channel history from the available data.
        If we lack data, we can pad with zeros.

        Proposed 12 channels for `JulesPolicy`:
        [x, y, z, vx, vy, vz, r, p, y, wr, wp, wy] (Full state history)

        Our `pos_history` has x,y,z.
        We can differentiate for vx, vy, vz.
        We have r, p, y in `observations`.
        We have wr, wp, wy in `observations` (as commands).

        However, extracting this from the packed `observations` buffer (flat) is tricky.
        The buffer structure is: 30 steps * 10 features.

        Let's simplify:
        We will manually buffer the 12-dim state in this wrapper class.
        """
        if not hasattr(self, '_history_buffer'):
            self._history_buffer = np.zeros((12, 40), dtype=np.float32)

        # Shift
        self._history_buffer = np.roll(self._history_buffer, -1, axis=1)

        # Update newest
        s = self.get_state()
        self._history_buffer[:, -1] = s

        return self._history_buffer[np.newaxis, :, :] # (1, 12, 40)
