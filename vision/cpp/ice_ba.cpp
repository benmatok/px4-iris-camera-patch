#include "ice_ba.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

// --- Matrix Utils ---
void invert3x3(double A[3][3], double inv[3][3]) {
    double det = A[0][0]*(A[1][1]*A[2][2]-A[2][1]*A[1][2]) -
                 A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) +
                 A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if(std::abs(det) < 1e-9) det = 1e-9;
    double invDet = 1.0/det;

    inv[0][0] = (A[1][1]*A[2][2]-A[2][1]*A[1][2])*invDet;
    inv[0][1] = (A[0][2]*A[2][1]-A[0][1]*A[2][2])*invDet;
    inv[0][2] = (A[0][1]*A[1][2]-A[0][2]*A[1][1])*invDet;
    inv[1][0] = (A[1][2]*A[2][0]-A[1][0]*A[2][2])*invDet;
    inv[1][1] = (A[0][0]*A[2][2]-A[0][2]*A[2][0])*invDet;
    inv[1][2] = (A[1][0]*A[0][2]-A[0][0]*A[1][2])*invDet;
    inv[2][0] = (A[1][0]*A[2][1]-A[2][0]*A[1][1])*invDet;
    inv[2][1] = (A[2][0]*A[0][1]-A[0][0]*A[2][1])*invDet;
    inv[2][2] = (A[0][0]*A[1][1]-A[1][0]*A[0][1])*invDet;
}

// 15x15 Cholesky Solver (State: p, theta, v, bg, ba)
bool solve15x15(double H[15][15], double b[15], double x[15]) {
    double L[15][15] = {0};
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];
            if (i == j) {
                double val = H[i][i] - sum;
                if (val <= 0) return false;
                L[i][j] = sqrt(val);
            } else {
                L[i][j] = (H[i][j] - sum) / L[j][j];
            }
        }
    }
    double y[15];
    for (int i = 0; i < 15; i++) {
        double sum = 0;
        for (int k = 0; k < i; k++) sum += L[i][k] * y[k];
        y[i] = (b[i] - sum) / L[i][i];
    }
    for (int i = 14; i >= 0; i--) {
        double sum = 0;
        for (int k = i + 1; k < 15; k++) sum += L[k][i] * x[k];
        x[i] = (y[i] - sum) / L[i][i];
    }
    return true;
}

// --- IMU Preint Implementation ---

IMUPreint::IMUPreint() : sum_dt(0) {}

void IMUPreint::integrate(const std::vector<ImuMeas>& raw_data, const Vec3& ba, const Vec3& bg) {
    data = raw_data;
    lin_ba = ba;
    lin_bg = bg;
    repropagate(ba, bg);
}

void IMUPreint::repropagate(const Vec3& ba, const Vec3& bg) {
    dp_curr = Vec3(0,0,0);
    dv_curr = Vec3(0,0,0);
    dq_curr = Quat(0,0,0,1);
    sum_dt = 0;

    for(const auto& m : data) {
        double dt = m.dt;
        Vec3 acc = m.acc - ba;
        Vec3 gyr = m.gyro - bg;

        // Midpoint integration
        // Orientation
        // d_q = q_k * [w*dt/2]_exp
        // Just simple Euler for speed (small dt)
        Quat dq_inc(gyr[0]*dt*0.5, gyr[1]*dt*0.5, gyr[2]*dt*0.5, 1.0);
        double n = sqrt(dq_inc.data[0]*dq_inc.data[0] + dq_inc.data[1]*dq_inc.data[1] + dq_inc.data[2]*dq_inc.data[2] + dq_inc.data[3]*dq_inc.data[3]);
        dq_inc.data[0]/=n; dq_inc.data[1]/=n; dq_inc.data[2]/=n; dq_inc.data[3]/=n;

        // Acc in local frame k
        Vec3 acc_w = dq_curr.rotate(acc);

        dp_curr = dp_curr + dv_curr * dt + acc_w * (0.5 * dt * dt);
        dv_curr = dv_curr + acc_w * dt;
        dq_curr = dq_curr * dq_inc;

        sum_dt += dt;
    }
}

// --- IceBA Implementation ---

IceBA::IceBA() : last_imu_cost(0), last_vis_cost(0) {}

void IceBA::add_frame(int id, double t, double* p, double* q, double* v, double* bg, double* ba,
                      const std::vector<ImuMeas>& imu_data, double baro_alt, bool has_baro, double* vel_prior, bool has_vel_prior) {
    Frame f;
    f.id = id;
    f.t = t;
    f.p = Vec3(p[0], p[1], p[2]);
    f.q = Quat(q[0], q[1], q[2], q[3]);
    f.v = Vec3(v[0], v[1], v[2]);
    f.bg = Vec3(bg[0], bg[1], bg[2]);
    f.ba = Vec3(ba[0], ba[1], ba[2]);

    f.baro_alt = baro_alt;
    f.has_baro = has_baro;
    if(vel_prior) {
        f.vel_prior = Vec3(vel_prior[0], vel_prior[1], vel_prior[2]);
        f.has_vel_prior = has_vel_prior;
    } else {
        f.has_vel_prior = false;
    }

    // Integrate IMU from PREVIOUS frame to THIS frame
    // We assume imu_data is [t_prev, t_curr]
    if (!imu_data.empty() && !frames.empty()) {
        Frame& prev = frames.back();
        // Use Previous frame's biases for linearization point
        f.preint.integrate(imu_data, prev.ba, prev.bg);
        f.has_preint = true;
    } else {
        f.has_preint = false;
    }

    // Check update
    bool found = false;
    for(auto& fr : frames) {
        if(fr.id == id) {
            fr = f;
            found = true;
            break;
        }
    }
    if(!found) frames.push_back(f);
}

void IceBA::add_obs(int frame_id, int pt_id, double u, double v) {
    if(points.find(pt_id) == points.end()) {
        Point p;
        p.id = pt_id;
        p.initialized = false;
        points[pt_id] = p;
    }
    points[pt_id].obs.push_back({frame_id, u, v});
}

void IceBA::triangulate() {
    for(auto& kv : points) {
        Point& pt = kv.second;
        if(pt.obs.size() < 2) continue;
        if(pt.initialized) continue;

        int f_idx = -1;
        for(size_t i=0; i<frames.size(); ++i) if(frames[i].id == pt.obs[0].frame_id) f_idx = i;

        if(f_idx >= 0) {
            Frame& f = frames[f_idx];
            double u = pt.obs[0].u;
            double v = pt.obs[0].v;
            double depth = 10.0;
            Vec3 ray_c(u * depth, v * depth, depth);
            Vec3 ray_w = f.q.rotate(ray_c);
            pt.p_world = f.p + ray_w;
            pt.initialized = true;
        }
    }
}

void IceBA::optimize_points() {
    double w_vis = 100.0;

    for(auto& kv : points) {
        Point& pt = kv.second;
        if(!pt.initialized) continue;

        double H[3][3] = {0};
        double b[3] = {0};

        for(auto& obs : pt.obs) {
            Frame* f = nullptr;
            for(auto& fr : frames) if(fr.id == obs.frame_id) f = &fr;
            if(!f) continue;

            Vec3 Pc = f->q.conj().rotate(pt.p_world - f->p);
            if(Pc[2] < 0.1) continue;

            double u = Pc[0]/Pc[2];
            double v = Pc[1]/Pc[2];
            double r_u = u - obs.u;
            double r_v = v - obs.v;

            double z_inv = 1.0 / Pc[2];
            double z2_inv = z_inv * z_inv;

            // J_proj w.r.t Pc
            double J_proj[2][3];
            J_proj[0][0] = z_inv; J_proj[0][1] = 0; J_proj[0][2] = -Pc[0]*z2_inv;
            J_proj[1][0] = 0; J_proj[1][1] = z_inv; J_proj[1][2] = -Pc[1]*z2_inv;

            double R_T[3][3];
            f->q.conj().to_matrix(R_T);

            // J_pw = J_proj * R^T
            double J_pw[2][3];
            for(int r=0; r<2; ++r) {
                for(int c=0; c<3; ++c) {
                    J_pw[r][c] = 0;
                    for(int k=0; k<3; ++k) J_pw[r][c] += J_proj[r][k] * R_T[k][c];
                }
            }

            for(int r=0; r<2; ++r) {
                double res = (r==0) ? r_u : r_v;
                // Robust Kernel (Huber)
                double huber_k = 0.05; // 5% of image width approx
                double abs_res = std::abs(res);
                double weight = w_vis;
                if(abs_res > huber_k) weight *= (huber_k / abs_res);

                for(int k=0; k<3; ++k) {
                    for(int l=0; l<3; ++l) H[k][l] += weight * J_pw[r][k] * J_pw[r][l];
                    b[k] -= weight * J_pw[r][k] * res;
                }
            }
        }

        for(int k=0; k<3; ++k) H[k][k] += 10.0; // Strong Prior to stay close

        double H_inv[3][3];
        invert3x3(H, H_inv);

        Vec3 dp;
        for(int k=0; k<3; ++k) {
            dp[k] = 0;
            for(int l=0; l<3; ++l) dp[k] += H_inv[k][l] * b[l];
        }

        pt.p_world = pt.p_world + dp * 0.5;
    }
}

void IceBA::optimize() {
    triangulate();

    int max_iter = 10;
    Vec3 g_grav(0, 0, 9.81);
    double w_imu = 5.0; // Extremely weak IMU constraints
    double w_rot = 5000.0; // Strong Rotation to fix pitch divergence
    double w_vis = 100.0; // Moderate vision
    double w_bias = 5000.0; // Lock bias to prevent runaway
    double w_baro = 20.0; // Barometer constraint
    double w_vp = 10.0; // Velocity Prior constraint

    last_imu_cost = 0;
    last_vis_cost = 0;

    for(int iter=0; iter<max_iter; ++iter) {
        // Reset costs for this iteration (accumulation)
        double current_imu_cost = 0;
        double current_vis_cost = 0;

        // 1. Structure
        optimize_points();

        // 2. Motion (and Bias)
        for(size_t i=0; i<frames.size(); ++i) {
            Frame& f = frames[i];

            // Re-propagate IMU with current bias estimate to linearize
            if(f.has_preint && i > 0) {
                Frame& prev = frames[i-1];
                // Should use prev.bg/ba?
                // Yes, preintegration is function of bias_k.
                f.preint.repropagate(prev.ba, prev.bg);
            }

            if(i == 0) continue; // Fix Frame 0

            // 15x15 System
            double H[15][15] = {0};
            double b[15] = {0};

            // --- Factor 1: Backward IMU (i-1 -> i) ---
            if(i > 0) {
                Frame& prev = frames[i-1];
                if(f.has_preint) {
                    double dt = f.preint.sum_dt;

                    // Pred
                    Vec3 p_pred = prev.p + prev.v * dt + g_grav * (0.5 * dt * dt) + prev.q.rotate(f.preint.dp_curr);
                    Vec3 v_pred = prev.v + g_grav * dt + prev.q.rotate(f.preint.dv_curr);

                    Vec3 r_p = f.p - p_pred;
                    Vec3 r_v = f.v - v_pred;

                    // Accumulate Cost
                    for(int k=0; k<3; ++k) current_imu_cost += 0.5 * w_imu * (r_p[k]*r_p[k] + r_v[k]*r_v[k]);

                    // Jacobian w.r.t f (frame i)
                    // d(rp)/dp_i = I
                    // d(rv)/dv_i = I

                    // Add to H (Indices: 0-2 P, 6-8 V)
                    for(int k=0; k<3; ++k) {
                        H[k][k] += w_imu;
                        b[k] -= w_imu * r_p[k];
                        H[k+6][k+6] += w_imu;
                        b[k+6] -= w_imu * r_v[k];
                    }

                    // Bias Random Walk: b_i - b_{i-1} = 0
                    Vec3 r_bg = f.bg - prev.bg;
                    Vec3 r_ba = f.ba - prev.ba;

                    // Accumulate Cost
                    for(int k=0; k<3; ++k) current_imu_cost += 0.5 * w_bias * (r_bg[k]*r_bg[k] + r_ba[k]*r_ba[k]);

                    // d(r_bg)/dbg_i = I
                    for(int k=0; k<3; ++k) {
                        H[k+9][k+9] += w_bias;  // BG 9-11
                        b[k+9] -= w_bias * r_bg[k];

                        H[k+12][k+12] += w_bias; // BA 12-14
                        b[k+12] -= w_bias * r_ba[k];
                    }

                    // Rotation Constraint: f.q should match prev.q * dq
                    // q_pred = prev.q * dq
                    // err = q_pred.conj() * f.q (Identity expected)
                    Quat q_pred = prev.q * f.preint.dq_curr;
                    Quat q_err = q_pred.conj() * f.q;
                    Vec3 r_theta(2.0*q_err.data[0], 2.0*q_err.data[1], 2.0*q_err.data[2]);

                    // Accumulate Cost
                    for(int k=0; k<3; ++k) current_imu_cost += 0.5 * w_rot * r_theta[k]*r_theta[k];

                    // If scalar is last (data[3]), vector part is 0,1,2.
                    // Jacobian J = I
                    for(int k=0; k<3; ++k) {
                        H[k+3][k+3] += w_rot;
                        b[k+3] -= w_rot * r_theta[k];
                    }
                }
            }

            // --- Factor 2: Forward IMU (i -> i+1) ---
            // Constraints on i from i+1
            if(i < frames.size() - 1) {
                Frame& next = frames[i+1];
                if(next.has_preint) {
                    double dt = next.preint.sum_dt;

                    // Need to re-prop next's preint with f's current bias?
                    // Done at start of loop?
                    // Wait, we iterate i. If we change f.ba, next.preint changes.
                    // We re-propagated `f.preint` using `prev.ba`.
                    // We need `next.preint` using `f.ba`.
                    next.preint.repropagate(f.ba, f.bg);

                    Vec3 p_pred = f.p + f.v * dt + g_grav * (0.5 * dt * dt) + f.q.rotate(next.preint.dp_curr);
                    Vec3 v_pred = f.v + g_grav * dt + f.q.rotate(next.preint.dv_curr);

                    Vec3 r_p = next.p - p_pred;
                    Vec3 r_v = next.v - v_pred;

                    // Jacobians w.r.t f (frame i)
                    // d(rp)/dp = -I
                    // d(rp)/dv = -I*dt
                    // d(rp)/dtheta = R [dp]x
                    // d(rp)/dba = -R * d(dp)/dba (need preint jacobian) -> Simplified: Ignore or numerical diff?
                    // ICE-BA uses precomputed jacobians. We lack them.
                    // Numerical Jacobian for Bias?
                    // Let's assume bias effect is small for this step or handled by random walk weight.
                    // Actually, if we don't update bias based on IMU error, it won't converge.
                    // We MUST have d(preint)/db.
                    // Simple approximation: d(dp)/dba ~ -0.5 * dt^2. d(dv)/dba ~ -dt.

                    double J_dp_dba_approx = -0.5 * dt * dt;
                    double J_dv_dba_approx = -dt;

                    // J_p_ba = -R * J_dp_dba
                    // J_v_ba = -R * J_dv_dba

                    // Rotation
                    double R[3][3]; f.q.to_matrix(R);
                    Vec3 dp = next.preint.dp_curr;
                    double dp_x[3][3] = {{0, -dp[2], dp[1]}, {dp[2], 0, -dp[0]}, {-dp[1], dp[0], 0}};
                    double J_p_theta[3][3];
                    // J_p_theta = - R * [dp]x
                    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                        J_p_theta[r][c] = 0;
                        for(int k=0; k<3; ++k) J_p_theta[r][c] -= R[r][k] * dp_x[k][c]; // Corrected sign: -R*[dp]x
                    }

                    // Fill H/b
                    double J[3][15] = {0}; // 3x15 for P residual
                    for(int r=0; r<3; ++r) {
                        J[r][0+r] = -1.0; // dp
                        J[r][3+0] = J_p_theta[r][0]; J[r][3+1] = J_p_theta[r][1]; J[r][3+2] = J_p_theta[r][2]; // theta
                        J[r][6+r] = -dt; // dv

                        // Bias A (indices 12-14)
                        // -R * (-0.5*dt*dt) * I ?? No, rotated.
                        // d(rp)/dba = - R * d(dp)/dba
                        // If d(dp)/dba = diag(-0.5t2), then -R * diag * I
                        for(int c=0; c<3; ++c) J[r][12+c] = -R[r][c] * J_dp_dba_approx;
                    }

                    for(int r=0; r<3; ++r) {
                        double res = r_p[r];
                        for(int k=0; k<15; ++k) {
                            for(int l=0; l<15; ++l) H[k][l] += w_imu * J[r][k] * J[r][l];
                            b[k] -= w_imu * J[r][k] * res;
                        }
                    }

                    // Velocity Residual J
                    double Jv[3][15] = {0};
                    Vec3 dv = next.preint.dv_curr;
                    double dv_x[3][3] = {{0, -dv[2], dv[1]}, {dv[2], 0, -dv[0]}, {-dv[1], dv[0], 0}};
                    double J_v_theta[3][3];
                    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                        J_v_theta[r][c] = 0;
                        for(int k=0; k<3; ++k) J_v_theta[r][c] -= R[r][k] * dv_x[k][c]; // Corrected sign: -R*[dv]x
                    }

                    for(int r=0; r<3; ++r) {
                        Jv[r][6+r] = -1.0; // dv
                        Jv[r][3+0] = J_v_theta[r][0]; Jv[r][3+1] = J_v_theta[r][1]; Jv[r][3+2] = J_v_theta[r][2];
                        // Bias A
                        for(int c=0; c<3; ++c) Jv[r][12+c] = -R[r][c] * J_dv_dba_approx;
                    }

                     for(int r=0; r<3; ++r) {
                        double res = r_v[r];
                        for(int k=0; k<15; ++k) {
                            for(int l=0; l<15; ++l) H[k][l] += w_imu * Jv[r][k] * Jv[r][l];
                            b[k] -= w_imu * Jv[r][k] * res;
                        }
                    }

                    // Rotation Constraint Forward (i -> i+1)
                    // q_pred = f.q * dq
                    // err = q_pred.conj() * next.q  (should be I)
                    Quat q_pred = f.q * next.preint.dq_curr;
                    Quat q_err = q_pred.conj() * next.q;
                    Vec3 r_theta(2.0*q_err.data[0], 2.0*q_err.data[1], 2.0*q_err.data[2]);

                    // Accumulate Cost (Forward Rotation)
                    for(int k=0; k<3; ++k) current_imu_cost += 0.5 * w_rot * r_theta[k]*r_theta[k];

                    // Jacobian J = - R_{dq}^T
                    // R_{dq} is rotation of dq
                    double R_dq[3][3];
                    next.preint.dq_curr.to_matrix(R_dq);
                    double J_rot[3][3];
                    // Transpose and Negate
                    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_rot[r][c] = -R_dq[c][r];

                    for(int r=0; r<3; ++r) {
                        double res = r_theta[r];
                        double J_row[15] = {0};
                        J_row[3+0] = J_rot[r][0]; J_row[3+1] = J_rot[r][1]; J_row[3+2] = J_rot[r][2];

                        for(int k=0; k<15; ++k) {
                             for(int l=0; l<15; ++l) H[k][l] += w_rot * J_row[k] * J_row[l];
                             b[k] -= w_rot * J_row[k] * res;
                        }
                    }

                    // Bias RW Forward (i -> i+1)
                    // r_b = next.b - f.b.
                    // d(r_b)/db_i = -I
                     for(int k=0; k<3; ++k) {
                        H[k+9][k+9] += w_bias;
                        b[k+9] -= w_bias * (next.bg[k] - f.bg[k]);

                        H[k+12][k+12] += w_bias;
                        b[k+12] -= w_bias * (next.ba[k] - f.ba[k]);
                    }
                }
            }

            // --- Vision ---
            for(auto& kv : points) {
                Point& pt = kv.second;
                if(!pt.initialized) continue;
                for(auto& obs : pt.obs) {
                    if(obs.frame_id == f.id) {
                         Vec3 Pc = f.q.conj().rotate(pt.p_world - f.p);
                        if(Pc[2] < 0.1) continue;
                        double u = Pc[0]/Pc[2];
                        double v = Pc[1]/Pc[2];
                        double r_u = u - obs.u;
                        double r_v = v - obs.v;

                        double z_inv = 1.0 / Pc[2];
                        double z2_inv = z_inv * z_inv;
                        double J_proj_u[3] = {z_inv, 0, -Pc[0]*z2_inv};
                        double J_proj_v[3] = {0, z_inv, -Pc[1]*z2_inv};

                        double R_T[3][3];
                        f.q.conj().to_matrix(R_T);

                        // J_pos = J_proj * (-R^T)
                        double J_pos[2][3];
                         for(int r=0; r<2; ++r) for(int c=0; c<3; ++c) {
                            J_pos[r][c] = 0;
                            for(int k=0; k<3; ++k) J_pos[r][c] += (r==0?J_proj_u[k]:J_proj_v[k]) * (-R_T[k][c]);
                        }

                        // J_theta
                         double J_theta[2][3];
                        // u
                        J_theta[0][0] = J_proj_u[1]*Pc[2] - J_proj_u[2]*Pc[1];
                        J_theta[0][1] = J_proj_u[2]*Pc[0] - J_proj_u[0]*Pc[2];
                        J_theta[0][2] = J_proj_u[0]*Pc[1] - J_proj_u[1]*Pc[0];
                        // v
                        J_theta[1][0] = J_proj_v[1]*Pc[2] - J_proj_v[2]*Pc[1];
                        J_theta[1][1] = J_proj_v[2]*Pc[0] - J_proj_v[0]*Pc[2];
                        J_theta[1][2] = J_proj_v[0]*Pc[1] - J_proj_v[1]*Pc[0];

                         for(int r=0; r<2; ++r) {
                            double resid = (r==0) ? r_u : r_v;
                             // Robust Kernel
                            double huber_k = 0.05;
                            double abs_res = std::abs(resid);
                            double weight = w_vis;
                            if(abs_res > huber_k) weight *= (huber_k / abs_res);

                            // Accumulate Cost (Vision)
                            current_vis_cost += 0.5 * weight * resid * resid;

                            double J_row[15] = {0};
                            for(int k=0; k<3; ++k) J_row[k] = J_pos[r][k];
                            for(int k=0; k<3; ++k) J_row[3+k] = J_theta[r][k];

                            for(int k=0; k<15; ++k) {
                                for(int l=0; l<15; ++l) H[k][l] += weight * J_row[k] * J_row[l];
                                b[k] -= weight * J_row[k] * resid;
                            }
                        }
                    }
                }
            }

            // --- Priors ---
            // Barometer (Z constraint)
            if(f.has_baro) {
                // pz + baro = 0 => pz = -baro
                double res = f.p[2] + f.baro_alt; // NED Pz is negative altitude

                // Accumulate Cost
                current_imu_cost += 0.5 * w_baro * res * res;

                // J_pz = 1. Index 2.
                H[2][2] += w_baro;
                b[2] -= w_baro * res;
            }

            // Velocity Prior
            if(f.has_vel_prior) {
                Vec3 r_vp = f.v - f.vel_prior;

                // Accumulate Cost
                for(int k=0; k<3; ++k) current_imu_cost += 0.5 * w_vp * r_vp[k]*r_vp[k];

                // J_v = I. Indices 6-8.
                for(int k=0; k<3; ++k) {
                    H[6+k][6+k] += w_vp;
                    b[6+k] -= w_vp * r_vp[k];
                }
            }

            // Damp
            for(int k=0; k<15; ++k) H[k][k] += 1.0;

            // Solve
            double dx[15];
            if(solve15x15(H, b, dx)) {
                double step = 0.5;
                for(int k=0; k<3; ++k) f.p[k] += dx[k] * step;

                Quat dq(dx[3]*step*0.5, dx[4]*step*0.5, dx[5]*step*0.5, 1.0);
                double nq = std::sqrt(dq.data[0]*dq.data[0] + dq.data[1]*dq.data[1] + dq.data[2]*dq.data[2] + dq.data[3]*dq.data[3]);
                dq.data[0]/=nq; dq.data[1]/=nq; dq.data[2]/=nq; dq.data[3]/=nq;
                f.q = f.q * dq;

                for(int k=0; k<3; ++k) f.v[k] += dx[6+k] * step;

                // Bias update
                for(int k=0; k<3; ++k) f.bg[k] += dx[9+k] * 0.1; // Slower learn rate for bias
                for(int k=0; k<3; ++k) f.ba[k] += dx[12+k] * 0.1;
            }
        }

        // Update total costs
        last_imu_cost = current_imu_cost;
        last_vis_cost = current_vis_cost;
    }
}

void IceBA::slide_window(int max_size) {
    while(frames.size() > (size_t)max_size) {
        int rm_id = frames[0].id;
        frames.erase(frames.begin());

        // Remove obs
        for(auto it = points.begin(); it != points.end(); ) {
            auto& pt = it->second;
            // Filter obs
            auto& obs = pt.obs;
            obs.erase(std::remove_if(obs.begin(), obs.end(),
                [rm_id](const PointObs& o){ return o.frame_id == rm_id; }), obs.end());

            if(obs.empty()) {
                it = points.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void IceBA::get_costs(double* imu_cost, double* vis_cost) {
    *imu_cost = last_imu_cost;
    *vis_cost = last_vis_cost;
}

void IceBA::solve() {
    optimize();
}

void IceBA::get_frame_state(int id, double* p, double* q, double* v, double* bg, double* ba) {
    for(auto& f : frames) {
        if(f.id == id) {
            p[0]=f.p[0]; p[1]=f.p[1]; p[2]=f.p[2];
            q[0]=f.q.data[0]; q[1]=f.q.data[1]; q[2]=f.q.data[2]; q[3]=f.q.data[3];
            v[0]=f.v[0]; v[1]=f.v[1]; v[2]=f.v[2];
            bg[0]=f.bg[0]; bg[1]=f.bg[1]; bg[2]=f.bg[2];
            ba[0]=f.ba[0]; ba[1]=f.ba[1]; ba[2]=f.ba[2];
            return;
        }
    }
}
