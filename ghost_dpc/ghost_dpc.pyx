# distutils: language=c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "ghost_model.hpp":
    struct GhostModel:
        float mass
        float drag_coeff
        float thrust_coeff
        float wind_x
        float wind_y

    struct GhostState:
        float px, py, pz
        float vx, vy, vz
        float roll, pitch, yaw

    struct GhostAction:
        float thrust
        float roll_rate
        float pitch_rate
        float yaw_rate

    struct Jacobian:
        float data[36]
        float grad_mass[9]

    cdef cppclass GhostPhysics:
        @staticmethod
        GhostState step(GhostModel model, GhostState state, GhostAction action, float dt)
        @staticmethod
        Jacobian get_gradients(GhostModel model, GhostState state, GhostAction action, float dt)

cdef extern from "ghost_estimator.hpp":
    cdef cppclass GhostEstimator:
        vector[GhostModel] models
        vector[float] probabilities
        float lambda_param

        GhostEstimator()
        void initialize(vector[GhostModel] initial_models)
        void update(GhostState state, GhostAction action, float* measured_accel, float dt)
        int get_best_model_index()
        GhostModel get_weighted_model()

cdef extern from "dpc_solver.hpp":
    cdef cppclass DPCSolver:
        DPCSolver()
        GhostAction solve(GhostState initial_state,
                          float* target_pos,
                          GhostAction initial_guess,
                          vector[GhostModel] models,
                          vector[float] weights,
                          float dt)

def test_build():
    print("Ghost-DPC Extension Built Successfully")

cdef class PyGhostModel:
    cdef GhostModel c_model

    def __init__(self, mass, drag, thrust_coeff, wind_x=0.0, wind_y=0.0):
        self.c_model.mass = mass
        self.c_model.drag_coeff = drag
        self.c_model.thrust_coeff = thrust_coeff
        self.c_model.wind_x = wind_x
        self.c_model.wind_y = wind_y

    def step(self, state_dict, action_dict, dt):
        cdef GhostState s
        s.px = state_dict['px']; s.py = state_dict['py']; s.pz = state_dict['pz']
        s.vx = state_dict['vx']; s.vy = state_dict['vy']; s.vz = state_dict['vz']
        s.roll = state_dict['roll']; s.pitch = state_dict['pitch']; s.yaw = state_dict['yaw']

        cdef GhostAction a
        a.thrust = action_dict['thrust']
        a.roll_rate = action_dict['roll_rate']
        a.pitch_rate = action_dict['pitch_rate']
        a.yaw_rate = action_dict['yaw_rate']

        cdef GhostState next_s = GhostPhysics.step(self.c_model, s, a, dt)

        return {
            'px': next_s.px, 'py': next_s.py, 'pz': next_s.pz,
            'vx': next_s.vx, 'vy': next_s.vy, 'vz': next_s.vz,
            'roll': next_s.roll, 'pitch': next_s.pitch, 'yaw': next_s.yaw
        }

    def get_gradients(self, state_dict, action_dict, dt):
        cdef GhostState s
        s.px = state_dict['px']; s.py = state_dict['py']; s.pz = state_dict['pz']
        s.vx = state_dict['vx']; s.vy = state_dict['vy']; s.vz = state_dict['vz']
        s.roll = state_dict['roll']; s.pitch = state_dict['pitch']; s.yaw = state_dict['yaw']

        cdef GhostAction a
        a.thrust = action_dict['thrust']
        a.roll_rate = action_dict['roll_rate']
        a.pitch_rate = action_dict['pitch_rate']
        a.yaw_rate = action_dict['yaw_rate']

        cdef Jacobian J = GhostPhysics.get_gradients(self.c_model, s, a, dt)

        # Convert to numpy (9, 4)
        out = np.zeros((9, 4), dtype=np.float32)
        cdef int i, j
        for i in range(9):
            for j in range(4):
                out[i, j] = J.data[i * 4 + j]

        # Convert mass grad (9,)
        grad_m = np.zeros(9, dtype=np.float32)
        for i in range(9):
            grad_m[i] = J.grad_mass[i]

        return out, grad_m

cdef class PyGhostEstimator:
    cdef GhostEstimator* c_est

    def __init__(self, models_list):
        self.c_est = new GhostEstimator()
        cdef vector[GhostModel] vec_models
        cdef GhostModel m
        for md in models_list:
            m.mass = md['mass']
            m.drag_coeff = md['drag_coeff']
            m.thrust_coeff = md['thrust_coeff']
            m.wind_x = md.get('wind_x', 0.0)
            m.wind_y = md.get('wind_y', 0.0)
            vec_models.push_back(m)
        self.c_est.initialize(vec_models)

    def __dealloc__(self):
        del self.c_est

    def update(self, state_dict, action_dict, measured_accel_list, dt):
        cdef GhostState s
        s.px = state_dict['px']; s.py = state_dict['py']; s.pz = state_dict['pz']
        s.vx = state_dict['vx']; s.vy = state_dict['vy']; s.vz = state_dict['vz']
        s.roll = state_dict['roll']; s.pitch = state_dict['pitch']; s.yaw = state_dict['yaw']

        cdef GhostAction a
        a.thrust = action_dict['thrust']
        a.roll_rate = action_dict['roll_rate']
        a.pitch_rate = action_dict['pitch_rate']
        a.yaw_rate = action_dict['yaw_rate']

        cdef float acc[3]
        acc[0] = measured_accel_list[0]
        acc[1] = measured_accel_list[1]
        acc[2] = measured_accel_list[2]

        self.c_est.update(s, a, acc, dt)

    def get_probabilities(self):
        return np.array(self.c_est.probabilities, dtype=np.float32)

    def get_weighted_model(self):
        cdef GhostModel m = self.c_est.get_weighted_model()
        return {
            'mass': m.mass,
            'drag_coeff': m.drag_coeff,
            'thrust_coeff': m.thrust_coeff,
            'wind_x': m.wind_x,
            'wind_y': m.wind_y
        }

cdef class PyDPCSolver:
    cdef DPCSolver* c_solver

    def __init__(self):
        self.c_solver = new DPCSolver()

    def __dealloc__(self):
        del self.c_solver

    def solve(self, state_dict, target_pos, initial_action_dict, models_list, weights_list, dt):
        cdef GhostState s
        s.px = state_dict['px']; s.py = state_dict['py']; s.pz = state_dict['pz']
        s.vx = state_dict['vx']; s.vy = state_dict['vy']; s.vz = state_dict['vz']
        s.roll = state_dict['roll']; s.pitch = state_dict['pitch']; s.yaw = state_dict['yaw']

        cdef float t_pos[3]
        t_pos[0] = target_pos[0]
        t_pos[1] = target_pos[1]
        t_pos[2] = target_pos[2]

        cdef GhostAction a_init
        a_init.thrust = initial_action_dict['thrust']
        a_init.roll_rate = initial_action_dict['roll_rate']
        a_init.pitch_rate = initial_action_dict['pitch_rate']
        a_init.yaw_rate = initial_action_dict['yaw_rate']

        cdef vector[GhostModel] vec_models
        cdef GhostModel m
        for md in models_list:
            m.mass = md['mass']
            m.drag_coeff = md['drag_coeff']
            m.thrust_coeff = md['thrust_coeff']
            m.wind_x = md.get('wind_x', 0.0)
            m.wind_y = md.get('wind_y', 0.0)
            vec_models.push_back(m)

        cdef vector[float] vec_weights
        for w in weights_list:
            vec_weights.push_back(w)

        cdef GhostAction a_opt = self.c_solver.solve(s, t_pos, a_init, vec_models, vec_weights, dt)

        return {
            'thrust': a_opt.thrust,
            'roll_rate': a_opt.roll_rate,
            'pitch_rate': a_opt.pitch_rate,
            'yaw_rate': a_opt.yaw_rate
        }
