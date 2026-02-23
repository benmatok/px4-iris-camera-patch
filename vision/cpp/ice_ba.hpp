#ifndef ICE_BA_HPP
#define ICE_BA_HPP

#include "matrix_utils.hpp"
#include <vector>
#include <map>

// Struct for Raw IMU Measurement
struct ImuMeas {
    double dt;
    Vec3 acc;
    Vec3 gyro;
};

// Struct for Pre-integrated IMU Factor
struct IMUPreint {
    double sum_dt;
    Vec3 dp;
    Vec3 dv;
    Quat dq;

    // Jacobians w.r.t Bias (Approximation)
    // J_dp_dba, J_dp_dbg
    // J_dv_dba, J_dv_dbg
    // J_dq_dbg
    // Simplified: Store linearization point biases
    Vec3 lin_ba;
    Vec3 lin_bg;

    // Current integrated values
    Vec3 dp_curr, dv_curr;
    Quat dq_curr;

    std::vector<ImuMeas> data;

    IMUPreint();
    void integrate(const std::vector<ImuMeas>& data, const Vec3& ba, const Vec3& bg);
    // Re-integrate with new biases
    void repropagate(const Vec3& ba, const Vec3& bg);
};

struct Frame {
    int id;
    double t;
    Vec3 p, v, bg, ba;
    Quat q;

    IMUPreint preint; // Constraint from Previous to Current
    bool has_preint;

    // Priors
    double baro_alt;
    bool has_baro;

    Vec3 vel_prior;
    bool has_vel_prior;
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

    // Updated Interface: Pass Raw IMU
    void add_frame(int id, double t, double* p, double* q, double* v, double* bg, double* ba,
                   const std::vector<ImuMeas>& imu_data, double baro_alt, bool has_baro, double* vel_prior, bool has_vel_prior);

    void add_obs(int frame_id, int pt_id, double u, double v);
    void solve();

    // Output
    void get_frame_state(int id, double* p, double* q, double* v, double* bg, double* ba);

    // Maintenance
    void slide_window(int max_size);
    void get_costs(double* imu_cost, double* vis_cost);

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

    double last_imu_cost;
    double last_vis_cost;
};

#endif
