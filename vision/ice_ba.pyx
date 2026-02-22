# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "cpp/ice_ba.hpp":
    cdef cppclass IceBA:
        IceBA()
        void add_frame(int id, double t, double* p, double* q, double* v, double* bg, double* ba,
                       double dt_pre, double* dp_pre, double* dq_pre, double* dv_pre)
        void add_obs(int frame_id, int pt_id, double u, double v)
        void solve()
        void get_frame_state(int id, double* p, double* q, double* v, double* bg, double* ba)

cdef class PyIceBA:
    cdef IceBA* c_ba

    def __cinit__(self):
        self.c_ba = new IceBA()

    def __dealloc__(self):
        del self.c_ba

    def add_frame(self, int id, double t, list p, list q, list v, list bg, list ba,
                  double dt_pre, list dp_pre, list dq_pre, list dv_pre):
        cdef double c_p[3]
        cdef double c_q[4]
        cdef double c_v[3]
        cdef double c_bg[3]
        cdef double c_ba[3]
        cdef double c_dp[3]
        cdef double c_dq[4]
        cdef double c_dv[3]

        for i in range(3): c_p[i] = p[i]
        for i in range(4): c_q[i] = q[i]
        for i in range(3): c_v[i] = v[i]
        for i in range(3): c_bg[i] = bg[i]
        for i in range(3): c_ba[i] = ba[i]

        if dp_pre is not None:
            for i in range(3): c_dp[i] = dp_pre[i]
            for i in range(4): c_dq[i] = dq_pre[i]
            for i in range(3): c_dv[i] = dv_pre[i]
        else:
            dt_pre = 0.0 # Flag

        self.c_ba.add_frame(id, t, c_p, c_q, c_v, c_bg, c_ba, dt_pre, c_dp, c_dq, c_dv)

    def add_obs(self, int frame_id, int pt_id, double u, double v):
        self.c_ba.add_obs(frame_id, pt_id, u, v)

    def solve(self):
        self.c_ba.solve()

    def get_frame_state(self, int id):
        cdef double c_p[3]
        cdef double c_q[4]
        cdef double c_v[3]
        cdef double c_bg[3]
        cdef double c_ba[3]

        self.c_ba.get_frame_state(id, c_p, c_q, c_v, c_bg, c_ba)

        return {
            'p': [c_p[0], c_p[1], c_p[2]],
            'q': [c_q[0], c_q[1], c_q[2], c_q[3]],
            'v': [c_v[0], c_v[1], c_v[2]],
            'bg': [c_bg[0], c_bg[1], c_bg[2]],
            'ba': [c_ba[0], c_ba[1], c_ba[2]]
        }
