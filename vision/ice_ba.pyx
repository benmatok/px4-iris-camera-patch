# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "cpp/ice_ba.hpp":
    cdef struct Vec3:
        double data[3]

    cdef struct ImuMeas:
        double dt
        Vec3 acc
        Vec3 gyro

    cdef cppclass IceBA:
        IceBA()
        void add_frame(int id, double t, double* p, double* q, double* v, double* bg, double* ba,
                       const vector[ImuMeas]& imu_data, double baro_alt, bool has_baro, double* vel_prior, bool has_vel_prior)
        void add_obs(int frame_id, int pt_id, double u, double v)
        void solve()
        void get_frame_state(int id, double* p, double* q, double* v, double* bg, double* ba)
        void slide_window(int max_size)
        void get_costs(double* imu_cost, double* vis_cost)

cdef class PyIceBA:
    cdef IceBA* c_ba

    def __cinit__(self):
        self.c_ba = new IceBA()

    def __dealloc__(self):
        del self.c_ba

    def add_frame(self, int id, double t, list p, list q, list v, list bg, list ba, list imu_data, baro=None, vel_prior=None):
        cdef double c_p[3]
        cdef double c_q[4]
        cdef double c_v[3]
        cdef double c_bg[3]
        cdef double c_ba[3]

        for i in range(3): c_p[i] = p[i]
        for i in range(4): c_q[i] = q[i]
        for i in range(3): c_v[i] = v[i]
        for i in range(3): c_bg[i] = bg[i]
        for i in range(3): c_ba[i] = ba[i]

        cdef vector[ImuMeas] c_imu
        cdef ImuMeas m
        if imu_data is not None:
            for item in imu_data:
                # item: (dt, acc_np, gyro_np)
                m.dt = item[0]
                m.acc.data[0] = item[1][0]
                m.acc.data[1] = item[1][1]
                m.acc.data[2] = item[1][2]
                m.gyro.data[0] = item[2][0]
                m.gyro.data[1] = item[2][1]
                m.gyro.data[2] = item[2][2]
                c_imu.push_back(m)

        cdef double c_baro = 0.0
        cdef bool has_baro = False
        if baro is not None:
            c_baro = baro
            has_baro = True

        cdef double c_vp[3]
        cdef double* p_vp = NULL
        cdef bool has_vp = False
        if vel_prior is not None:
            c_vp[0] = vel_prior[0]
            c_vp[1] = vel_prior[1]
            c_vp[2] = vel_prior[2]
            p_vp = c_vp
            has_vp = True

        self.c_ba.add_frame(id, t, c_p, c_q, c_v, c_bg, c_ba, c_imu, c_baro, has_baro, p_vp, has_vp)

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

    def slide_window(self, int max_size):
        self.c_ba.slide_window(max_size)

    def get_costs(self):
        cdef double imu_cost
        cdef double vis_cost
        self.c_ba.get_costs(&imu_cost, &vis_cost)
        return imu_cost, vis_cost
