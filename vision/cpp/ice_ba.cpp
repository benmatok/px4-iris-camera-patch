#include "ice_ba.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

// Utils
// 3x3 Inv
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

// 9x9 Cholesky Solver
bool solve9x9(double H[9][9], double b[9], double x[9]) {
    double L[9][9] = {0};
    for (int i = 0; i < 9; i++) {
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
    double y[9];
    for (int i = 0; i < 9; i++) {
        double sum = 0;
        for (int k = 0; k < i; k++) sum += L[i][k] * y[k];
        y[i] = (b[i] - sum) / L[i][i];
    }
    for (int i = 8; i >= 0; i--) {
        double sum = 0;
        for (int k = i + 1; k < 9; k++) sum += L[k][i] * x[k];
        x[i] = (y[i] - sum) / L[i][i];
    }
    return true;
}

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
            double depth = 10.0; // Heuristic depth
            Vec3 ray_c(u * depth, v * depth, depth);
            Vec3 ray_w = f.q.rotate(ray_c);
            pt.p_world = f.p + ray_w;
            pt.initialized = true;
        }
    }
}

void IceBA::optimize_points() {
    double w_vis = 50.0;

    for(auto& kv : points) {
        Point& pt = kv.second;
        if(!pt.initialized) continue;

        double H[3][3] = {0};
        double b[3] = {0};

        for(auto& obs : pt.obs) {
            Frame* f = nullptr;
            for(auto& fr : frames) if(fr.id == obs.frame_id) f = &fr;
            if(!f) continue;

            // P_c = R^T (P_w - p)
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

            // dPc/dP_w = R^T
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

            // Accumulate
            for(int r=0; r<2; ++r) {
                double res = (r==0) ? r_u : r_v;
                for(int k=0; k<3; ++k) {
                    for(int l=0; l<3; ++l) H[k][l] += w_vis * J_pw[r][k] * J_pw[r][l];
                    b[k] -= w_vis * J_pw[r][k] * res;
                }
            }
        }

        // Damp
        for(int k=0; k<3; ++k) H[k][k] += 1.0;

        // Solve 3x3
        double H_inv[3][3];
        invert3x3(H, H_inv);

        Vec3 dp;
        for(int k=0; k<3; ++k) {
            dp[k] = 0;
            for(int l=0; l<3; ++l) dp[k] += H_inv[k][l] * b[l];
        }

        pt.p_world = pt.p_world + dp * 0.5; // Step
    }
}

void IceBA::optimize() {
    triangulate();

    int max_iter = 10;
    Vec3 g_grav(0, 0, 9.81);
    double w_imu = 100.0;
    double w_vis = 50.0;

    for(int iter=0; iter<max_iter; ++iter) {

        // 1. Optimize Structure
        optimize_points();

        // 2. Optimize Motion
        for(size_t i=0; i<frames.size(); ++i) {
            Frame& f = frames[i];
            if(i == 0) continue;

            double H[9][9] = {0};
            double b[9] = {0};

            // --- Factor 1: Backward IMU (i-1 -> i) ---
            if(i > 0) {
                Frame& prev = frames[i-1];
                if(f.has_preint) {
                    double dt = f.dt_pre;

                    // Pred
                    Vec3 p_pred = prev.p + prev.v * dt + g_grav * (0.5 * dt * dt) + prev.q.rotate(f.dp_pre);
                    Vec3 v_pred = prev.v + g_grav * dt + prev.q.rotate(f.dv_pre);

                    Vec3 r_p = f.p - p_pred;
                    Vec3 r_v = f.v - v_pred;

                    // d(res)/dx_i.
                    // res_p = p_i - ... => J_p = I
                    // res_v = v_i - ... => J_v = I

                    for(int k=0; k<3; ++k) {
                        H[k][k] += w_imu;
                        b[k] -= w_imu * r_p[k];
                        H[k+6][k+6] += w_imu;
                        b[k+6] -= w_imu * r_v[k];
                    }

                    // Rotation residual? Skipped for simplicity, assumes orientation is good
                }
            }

            // --- Factor 2: Forward IMU (i -> i+1) ---
            if(i < frames.size() - 1) {
                Frame& next = frames[i+1];
                if(next.has_preint) {
                    double dt = next.dt_pre;

                    // Residual definition: r = next.p - (f.p + f.v*dt + ... + f.q.rotate(dp))
                    Vec3 p_pred = f.p + f.v * dt + g_grav * (0.5 * dt * dt) + f.q.rotate(next.dp_pre);
                    Vec3 v_pred = f.v + g_grav * dt + f.q.rotate(next.dv_pre);

                    Vec3 r_p = next.p - p_pred;
                    Vec3 r_v = next.v - v_pred;

                    // J w.r.t f (frame i)
                    // d(r_p)/dp_i = -I
                    // d(r_p)/dv_i = -I * dt
                    // d(r_p)/dtheta_i = R_i [dp_pre]x (derived previously)

                    // [dp_pre]x
                    Vec3 dp = next.dp_pre;
                    double dp_x[3][3] = {{0, -dp[2], dp[1]}, {dp[2], 0, -dp[0]}, {-dp[1], dp[0], 0}};

                    // R_i * [dp_pre]x
                    double R[3][3];
                    f.q.to_matrix(R);
                    double J_p_theta[3][3]; // 3x3
                    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                        J_p_theta[r][c] = 0;
                        for(int k=0; k<3; ++k) J_p_theta[r][c] += R[r][k] * dp_x[k][c];
                    }

                    // Accumulate for P residual
                    // Rows 0-2 of H/b correspond to p_i
                    // But here we are adding to H based on jacobians.
                    // J = [-I,  J_p_theta,  -I*dt] (size 3x9)

                    // Simplified accumulation loop
                    double J_row[3][9]; // 3 residuals (x,y,z), 9 vars
                    for(int r=0; r<3; ++r) {
                        // dp
                        J_row[r][0] = (r==0?-1:0); J_row[r][1] = (r==1?-1:0); J_row[r][2] = (r==2?-1:0);
                        // dtheta
                        J_row[r][3] = J_p_theta[r][0]; J_row[r][4] = J_p_theta[r][1]; J_row[r][5] = J_p_theta[r][2];
                        // dv
                        J_row[r][6] = (r==0?-dt:0); J_row[r][7] = (r==1?-dt:0); J_row[r][8] = (r==2?-dt:0);
                    }

                    for(int r=0; r<3; ++r) {
                        double res = r_p[r];
                        for(int k=0; k<9; ++k) {
                            for(int l=0; l<9; ++l) H[k][l] += w_imu * J_row[r][k] * J_row[r][l];
                            b[k] -= w_imu * J_row[r][k] * res;
                        }
                    }

                    // Velocity Residual J
                    // d(r_v)/dv_i = -I
                    // d(r_v)/dtheta_i = R_i [dv_pre]x
                    Vec3 dv_v = next.dv_pre;
                    double dv_x[3][3] = {{0, -dv_v[2], dv_v[1]}, {dv_v[2], 0, -dv_v[0]}, {-dv_v[1], dv_v[0], 0}};
                    double J_v_theta[3][3];
                    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                        J_v_theta[r][c] = 0;
                        for(int k=0; k<3; ++k) J_v_theta[r][c] += R[r][k] * dv_x[k][c];
                    }

                    for(int r=0; r<3; ++r) {
                        // Fill J (1x9)
                        double J_row_v[9] = {0};
                        // dp = 0
                        // dtheta
                        J_row_v[3] = J_v_theta[r][0]; J_row_v[4] = J_v_theta[r][1]; J_row_v[5] = J_v_theta[r][2];
                        // dv
                        J_row_v[6] = (r==0?-1:0); J_row_v[7] = (r==1?-1:0); J_row_v[8] = (r==2?-1:0);

                        double res = r_v[r];
                        for(int k=0; k<9; ++k) {
                            for(int l=0; l<9; ++l) H[k][l] += w_imu * J_row_v[k] * J_row_v[l];
                            b[k] -= w_imu * J_row_v[k] * res;
                        }
                    }
                }
            }

            // --- Factor 3: Vision ---
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

                        double J_pos[2][3];
                        for(int r=0; r<2; ++r) for(int c=0; c<3; ++c) {
                            J_pos[r][c] = 0;
                            for(int k=0; k<3; ++k) J_pos[r][c] += J_proj_u[k] * (r==0?1:0) * (-R_T[k][c]) + J_proj_v[k] * (r==1?1:0) * (-R_T[k][c]);
                        }
                        // Corrected use of J_proj_u/v
                        // Re-defining J_proj for clarity above

                        double J_theta[2][3];
                        // J_theta = J_proj * [Pc]x
                        J_theta[0][0] = J_proj_u[1]*Pc[2] - J_proj_u[2]*Pc[1];
                        J_theta[0][1] = J_proj_u[2]*Pc[0] - J_proj_u[0]*Pc[2];
                        J_theta[0][2] = J_proj_u[0]*Pc[1] - J_proj_u[1]*Pc[0];
                        J_theta[1][0] = J_proj_v[1]*Pc[2] - J_proj_v[2]*Pc[1];
                        J_theta[1][1] = J_proj_v[2]*Pc[0] - J_proj_v[0]*Pc[2];
                        J_theta[1][2] = J_proj_v[0]*Pc[1] - J_proj_v[1]*Pc[0];

                        for(int r=0; r<2; ++r) {
                            double resid = (r==0) ? r_u : r_v;
                            double J_row[9];
                            for(int k=0; k<3; ++k) J_row[k] = J_pos[r][k];
                            for(int k=0; k<3; ++k) J_row[3+k] = J_theta[r][k];
                            for(int k=0; k<3; ++k) J_row[6+k] = 0.0;

                            for(int k=0; k<9; ++k) {
                                for(int l=0; l<9; ++l) H[k][l] += w_vis * J_row[k] * J_row[l];
                                b[k] -= w_vis * J_row[k] * resid;
                            }
                        }
                    }
                }
            }

            // Regularization
            for(int k=0; k<9; ++k) H[k][k] += 1.0;

            // Solve
            double dx[9];
            if(solve9x9(H, b, dx)) {
                double step = 0.5; // Conservative step
                for(int k=0; k<3; ++k) f.p[k] += dx[k] * step;

                Quat dq(dx[3]*step*0.5, dx[4]*step*0.5, dx[5]*step*0.5, 1.0);
                double nq = std::sqrt(dq.data[0]*dq.data[0] + dq.data[1]*dq.data[1] + dq.data[2]*dq.data[2] + dq.data[3]*dq.data[3]);
                dq.data[0]/=nq; dq.data[1]/=nq; dq.data[2]/=nq; dq.data[3]/=nq;
                f.q = f.q * dq;

                for(int k=0; k<3; ++k) f.v[k] += dx[6+k] * step;
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
