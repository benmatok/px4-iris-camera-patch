#include "ice_ba.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

IceBA::IceBA() {}

void IceBA::add_frame(int id, double t, double* p, double* q, double* v, double* bg, double* ba,
               double dt_pre, double* dp_pre, double* dq_pre, double* dv_pre) {
    Frame f;
    f.id = id;
    f.t = t;
    f.p = Vec3(p[0], p[1], p[2]);
    f.q = Quat(q[0], q[1], q[2], q[3]);
    f.v = Vec3(v[0], v[1], v[2]);
    f.bg = Vec3(bg[0], bg[1], bg[2]);
    f.ba = Vec3(ba[0], ba[1], ba[2]);

    f.dt_pre = dt_pre;
    if (dt_pre > 0) {
        f.has_preint = true;
        f.dp_pre = Vec3(dp_pre[0], dp_pre[1], dp_pre[2]);
        f.dq_pre = Quat(dq_pre[0], dq_pre[1], dq_pre[2], dq_pre[3]);
        f.dv_pre = Vec3(dv_pre[0], dv_pre[1], dv_pre[2]);
    } else {
        f.has_preint = false;
    }

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
            double depth = 5.0;
            Vec3 ray_c(u * depth, v * depth, depth);
            Vec3 ray_w = f.q.rotate(ray_c);
            pt.p_world = f.p + ray_w;
            pt.initialized = true;
        }
    }
}

// Minimal solver for 9x9 symmetric positive definite system Hx = b
// Uses Cholesky LDLT or Gaussian
bool solve9x9(double H[9][9], double b[9], double x[9]) {
    // Cholesky L * L^T = H
    double L[9][9] = {0};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];

            if (i == j) {
                double val = H[i][i] - sum;
                if (val <= 0) return false; // Not PD
                L[i][j] = sqrt(val);
            } else {
                L[i][j] = (H[i][j] - sum) / L[j][j];
            }
        }
    }

    // Forward substitution L * y = b
    double y[9];
    for (int i = 0; i < 9; i++) {
        double sum = 0;
        for (int k = 0; k < i; k++) sum += L[i][k] * y[k];
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Backward substitution L^T * x = y
    for (int i = 8; i >= 0; i--) {
        double sum = 0;
        for (int k = i + 1; k < 9; k++) sum += L[k][i] * x[k];
        x[i] = (y[i] - sum) / L[i][i];
    }
    return true;
}

void IceBA::optimize() {
    triangulate();

    int max_iter = 10;
    Vec3 g_grav(0, 0, 9.81);

    for(int iter=0; iter<max_iter; ++iter) {

        for(size_t i=0; i<frames.size(); ++i) {
            Frame& f = frames[i];
            if(i == 0) continue;

            // Build Normal Equation for Frame i: H (9x9), b (9)
            // State order: dp (0-2), dtheta (3-5), dv (6-8)
            double H[9][9] = {0};
            double b[9] = {0};

            // 1. IMU Factor
            if(i > 0) {
                Frame& prev = frames[i-1];
                if(f.has_preint) {
                    double dt = f.dt_pre;

                    // Residuals
                    Vec3 p_pred = prev.p + prev.v * dt + g_grav * (0.5 * dt * dt) + prev.q.rotate(f.dp_pre);
                    Vec3 v_pred = prev.v + g_grav * dt + prev.q.rotate(f.dv_pre);

                    Vec3 r_p = f.p - p_pred;
                    Vec3 r_v = f.v - v_pred;

                    // TODO: Orientation residual
                    // R_err = (R_pred^T * R_curr)
                    // R_pred = R_prev * R_preint

                    double w_imu = 100.0;

                    // Pos/Vel Diagonal fill
                    for(int k=0; k<3; ++k) {
                        H[k][k] += w_imu;
                        b[k] -= w_imu * r_p[k];

                        H[k+6][k+6] += w_imu;
                        b[k+6] -= w_imu * r_v[k];
                    }
                }
            }

            // 2. Vision Factors
            double w_vis = 50.0;

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

                        // Jacobians
                        double z_inv = 1.0 / Pc[2];
                        double z2_inv = z_inv * z_inv;

                        // d(uv)/dPc
                        double J_proj[2][3];
                        J_proj[0][0] = z_inv; J_proj[0][1] = 0; J_proj[0][2] = -Pc[0]*z2_inv;
                        J_proj[1][0] = 0; J_proj[1][1] = z_inv; J_proj[1][2] = -Pc[1]*z2_inv;

                        // dPc/dp = -R^T
                        // dPc/dtheta = [Pc]x

                        double R_T[3][3];
                        f.q.conj().to_matrix(R_T); // This is R^T

                        // J_pos = J_proj * (-R^T) (2x3)
                        double J_pos[2][3];
                        for(int r=0; r<2; ++r) {
                            for(int c=0; c<3; ++c) {
                                J_pos[r][c] = 0;
                                for(int k=0; k<3; ++k) {
                                    J_pos[r][c] += J_proj[r][k] * (-R_T[k][c]);
                                }
                            }
                        }

                        // J_theta = J_proj * [Pc]x (2x3)
                        // [Pc]x = [ 0  -z   y]
                        //         [ z   0  -x]
                        //         [-y   x   0]
                        double J_theta[2][3];
                        // u row
                        J_theta[0][0] = J_proj[0][1]*Pc[2] - J_proj[0][2]*Pc[1];
                        J_theta[0][1] = J_proj[0][2]*Pc[0] - J_proj[0][0]*Pc[2];
                        J_theta[0][2] = J_proj[0][0]*Pc[1] - J_proj[0][1]*Pc[0];
                        // v row
                        J_theta[1][0] = J_proj[1][1]*Pc[2] - J_proj[1][2]*Pc[1];
                        J_theta[1][1] = J_proj[1][2]*Pc[0] - J_proj[1][0]*Pc[2];
                        J_theta[1][2] = J_proj[1][0]*Pc[1] - J_proj[1][1]*Pc[0];

                        // Accumulate H, b
                        // J = [J_pos, J_theta, 0] (2x9)
                        // r = [r_u, r_v]
                        // H += J^T * W * J
                        // b -= J^T * W * r

                        for(int r=0; r<2; ++r) { // u, v
                            double weight = w_vis;
                            double resid = (r==0) ? r_u : r_v;

                            // Fill J_full (length 9)
                            double J_row[9];
                            for(int k=0; k<3; ++k) J_row[k] = J_pos[r][k];
                            for(int k=0; k<3; ++k) J_row[3+k] = J_theta[r][k];
                            for(int k=0; k<3; ++k) J_row[6+k] = 0.0;

                            for(int k=0; k<9; ++k) {
                                for(int l=0; l<9; ++l) {
                                    H[k][l] += weight * J_row[k] * J_row[l];
                                }
                                b[k] -= weight * J_row[k] * resid;
                            }
                        }
                    }
                }
            }

            // Regularization (Levenberg-Marquardt damping)
            for(int k=0; k<9; ++k) H[k][k] += 1e-3;

            // Solve H dx = b
            double dx[9];
            if(solve9x9(H, b, dx)) {
                // Update State
                // Pos
                for(int k=0; k<3; ++k) f.p[k] += dx[k];

                // Theta (Exp map)
                Quat dq(dx[3]*0.5, dx[4]*0.5, dx[5]*0.5, 1.0);
                double nq = std::sqrt(dq.data[0]*dq.data[0] + dq.data[1]*dq.data[1] + dq.data[2]*dq.data[2] + dq.data[3]*dq.data[3]);
                dq.data[0]/=nq; dq.data[1]/=nq; dq.data[2]/=nq; dq.data[3]/=nq;
                f.q = f.q * dq;

                // Vel
                for(int k=0; k<3; ++k) f.v[k] += dx[6+k];
            }
        }
    }
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
