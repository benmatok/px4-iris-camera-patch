#ifndef ICE_BA_HPP
#define ICE_BA_HPP

#include "matrix_utils.hpp"
#include <vector>
#include <map>

struct Frame {
    int id;
    double t;
    Vec3 p, v, bg, ba;
    Quat q;
    // Preint
    double dt_pre;
    Vec3 dp_pre, dv_pre;
    Quat dq_pre;
    bool has_preint;
};

struct PointObs {
    int frame_id;
    double u, v;
};

struct Point {
    int id;
    Vec3 p_world;
    bool initialized;
    std::vector<PointObs> obs;
};

class IceBA {
public:
    IceBA();
    void add_frame(int id, double t, double* p, double* q, double* v, double* bg, double* ba,
                   double dt_pre, double* dp_pre, double* dq_pre, double* dv_pre);
    void add_obs(int frame_id, int pt_id, double u, double v);
    void solve();

    // Output
    void get_frame_state(int id, double* p, double* q, double* v, double* bg, double* ba);

private:
    std::vector<Frame> frames;
    std::map<int, Point> points;

    // Config
    double w_vis = 2.0;
    double w_imu = 10.0;

    // Helpers
    void triangulate();
    void optimize_points();
    void optimize(); // Gauss-Newton
};

#endif
