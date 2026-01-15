#ifndef PHYSICS_AVX_HPP
#define PHYSICS_AVX_HPP

#include <immintrin.h>
#include <cmath>
#include <cstring>
#include "avx_mathfun.h"
#include "avx_mathfun_lut.h"

// Constants
static const float DT = 0.05f;
static const float GRAVITY = 9.81f;
static const int SUBSTEPS = 2;

// Helper: Custom Memmove using AVX for new obs size (274)
// Shift 0..261 <- 9..270
// 261 floats shift.
inline void shift_observations_avx(float* observations, int i) {
    for (int k = 0; k < 8; k++) {
        float* ptr = &observations[(i + k) * 274];
        float* src = ptr + 9; // Shift by 9 (one step history)
        float* dst = ptr;

        // Copy 261 floats
        // 261 / 8 = 32 blocks + 5 remainder
        for (int j = 0; j < 32; j++) {
            __m256 v = _mm256_loadu_ps(src + j * 8);
            _mm256_storeu_ps(dst + j * 8, v);
        }
        dst[256] = src[256];
        dst[257] = src[257];
        dst[258] = src[258];
        dst[259] = src[259];
        dst[260] = src[260];
    }
}

// AVX2 Implementation for a block of 8 agents
inline void step_agents_avx2(
    int i,
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float* roll, float* pitch, float* yaw,
    float* masses, float* drag_coeffs, float* thrust_coeffs,
    float* target_vx, float* target_vy, float* target_vz, float* target_yaw_rate,
    float* vt_x, float* vt_y, float* vt_z, // Virtual Target Position (Output)
    float* target_trajectory, // Precomputed Trajectory: Shape (episode_length+1, num_agents, 3)
    float* pos_history, // Shape (episode_length, num_agents, 3)
    float* observations, // stride 274
    float* rewards,
    float* reward_components, // New: stride 8 (num_agents, 8)
    float* done_flags,
    float* actions, // stride 4
    int episode_length,
    int t,
    int num_agents
) {
    // Check if we can process 8 agents
    if (i + 8 > num_agents) return;

    // Load State
    __m256 px = _mm256_loadu_ps(&pos_x[i]);
    __m256 py = _mm256_loadu_ps(&pos_y[i]);
    __m256 pz = _mm256_loadu_ps(&pos_z[i]);
    __m256 vx = _mm256_loadu_ps(&vel_x[i]);
    __m256 vy = _mm256_loadu_ps(&vel_y[i]);
    __m256 vz = _mm256_loadu_ps(&vel_z[i]);
    __m256 r  = _mm256_loadu_ps(&roll[i]);
    __m256 p  = _mm256_loadu_ps(&pitch[i]);
    __m256 y  = _mm256_loadu_ps(&yaw[i]);

    __m256 mass = _mm256_loadu_ps(&masses[i]);
    __m256 drag = _mm256_loadu_ps(&drag_coeffs[i]);
    __m256 t_coeff = _mm256_loadu_ps(&thrust_coeffs[i]);

    // Actions
    __m256i idx_base = _mm256_set_epi32(7*4, 6*4, 5*4, 4*4, 3*4, 2*4, 1*4, 0);
    __m256i idx_0 = _mm256_add_epi32(idx_base, _mm256_set1_epi32(i*4));
    __m256i idx_1 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(1));
    __m256i idx_2 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(2));
    __m256i idx_3 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(3));

    __m256 thrust_cmd = _mm256_i32gather_ps(actions, idx_0, 4);
    __m256 roll_rate_cmd = _mm256_i32gather_ps(actions, idx_1, 4);
    __m256 pitch_rate_cmd = _mm256_i32gather_ps(actions, idx_2, 4);
    __m256 yaw_rate_cmd = _mm256_i32gather_ps(actions, idx_3, 4);

    // Constants
    __m256 dt_v = _mm256_set1_ps(DT);
    __m256 g_v = _mm256_set1_ps(9.81f);
    __m256 c20 = _mm256_set1_ps(20.0f);
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c5 = _mm256_set1_ps(5.0f);
    __m256 c01 = _mm256_set1_ps(0.1f);
    __m256 c10 = _mm256_set1_ps(10.0f);
    __m256 c05 = _mm256_set1_ps(0.5f);
    __m256 c_fov = _mm256_set1_ps(1.732f); // tan(60) for 120 deg FOV

    // ------------------------------------------------------------------------
    // Lookup Virtual Target Position from Precomputed Trajectory
    // ------------------------------------------------------------------------
    int step_idx = t;
    if (step_idx > episode_length) step_idx = episode_length; // Clamp

    float* traj_base = &target_trajectory[step_idx * num_agents * 3];

    __m256i idx_x = _mm256_set_epi32(7*3, 6*3, 5*3, 4*3, 3*3, 2*3, 1*3, 0);
    __m256i idx_y = _mm256_add_epi32(idx_x, _mm256_set1_epi32(1));
    __m256i idx_z = _mm256_add_epi32(idx_x, _mm256_set1_epi32(2));

    __m256 vtx = _mm256_i32gather_ps(&traj_base[i*3], idx_x, 4);
    __m256 vty = _mm256_i32gather_ps(&traj_base[i*3], idx_y, 4);
    __m256 vtz = _mm256_i32gather_ps(&traj_base[i*3], idx_z, 4);

    int next_step = step_idx + 1;
    if (next_step > episode_length) next_step = episode_length;
    float* traj_next = &target_trajectory[next_step * num_agents * 3];

    __m256 vtx_n = _mm256_i32gather_ps(&traj_next[i*3], idx_x, 4);
    __m256 vty_n = _mm256_i32gather_ps(&traj_next[i*3], idx_y, 4);
    __m256 vtz_n = _mm256_i32gather_ps(&traj_next[i*3], idx_z, 4);

    __m256 inv_dt = _mm256_set1_ps(1.0f / DT);

    __m256 vtvx = _mm256_mul_ps(_mm256_sub_ps(vtx_n, vtx), inv_dt);
    __m256 vtvy = _mm256_mul_ps(_mm256_sub_ps(vty_n, vty), inv_dt);
    __m256 vtvz = _mm256_mul_ps(_mm256_sub_ps(vtz_n, vtz), inv_dt);

    _mm256_storeu_ps(&vt_x[i], vtx);
    _mm256_storeu_ps(&vt_y[i], vty);
    _mm256_storeu_ps(&vt_z[i], vtz);

    // ------------------------------------------------------------------------
    // Tracker / UV features needed for History
    // ------------------------------------------------------------------------
    shift_observations_avx(observations, i);

    // Substeps
    for (int s = 0; s < SUBSTEPS; s++) {
        // Dynamics
        r = _mm256_add_ps(r, _mm256_mul_ps(roll_rate_cmd, dt_v));
        p = _mm256_add_ps(p, _mm256_mul_ps(pitch_rate_cmd, dt_v));
        y = _mm256_add_ps(y, _mm256_mul_ps(yaw_rate_cmd, dt_v));

        __m256 max_thrust = _mm256_mul_ps(c20, t_coeff);
        __m256 thrust_force = _mm256_mul_ps(thrust_cmd, max_thrust);

        // Dynamics Sincos (LUT)
        __m256 sr = lut_sin256_ps(r);
        __m256 cr = lut_cos256_ps(r);
        __m256 sp = lut_sin256_ps(p);
        __m256 cp = lut_cos256_ps(p);
        __m256 sy = lut_sin256_ps(y);
        __m256 cy = lut_cos256_ps(y);

        __m256 term1_x = _mm256_mul_ps(_mm256_mul_ps(cy, sp), cr);
        __m256 term2_x = _mm256_mul_ps(sy, sr);
        __m256 ax_thrust = _mm256_div_ps(_mm256_mul_ps(thrust_force, _mm256_add_ps(term1_x, term2_x)), mass);

        __m256 term1_y = _mm256_mul_ps(_mm256_mul_ps(sy, sp), cr);
        __m256 term2_y = _mm256_mul_ps(cy, sr);
        __m256 ay_thrust = _mm256_div_ps(_mm256_mul_ps(thrust_force, _mm256_sub_ps(term1_y, term2_y)), mass);

        __m256 az_thrust = _mm256_div_ps(_mm256_mul_ps(thrust_force, _mm256_mul_ps(cp, cr)), mass);
        __m256 az_gravity = _mm256_sub_ps(c0, g_v);

        __m256 ax_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), vx);
        __m256 ay_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), vy);
        __m256 az_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), vz);

        __m256 ax = _mm256_add_ps(ax_thrust, ax_drag);
        __m256 ay = _mm256_add_ps(ay_thrust, ay_drag);
        __m256 az = _mm256_add_ps(az_thrust, _mm256_add_ps(az_gravity, az_drag));

        vx = _mm256_add_ps(vx, _mm256_mul_ps(ax, dt_v));
        vy = _mm256_add_ps(vy, _mm256_mul_ps(ay, dt_v));
        vz = _mm256_add_ps(vz, _mm256_mul_ps(az, dt_v));

        px = _mm256_add_ps(px, _mm256_mul_ps(vx, dt_v));
        py = _mm256_add_ps(py, _mm256_mul_ps(vy, dt_v));
        pz = _mm256_add_ps(pz, _mm256_mul_ps(vz, dt_v));

        // Terrain Sincos (LUT)
        __m256 sin_px = lut_sin256_ps(_mm256_mul_ps(c01, px));
        __m256 cos_py = lut_cos256_ps(_mm256_mul_ps(c01, py));
        __m256 terr_z = _mm256_mul_ps(c5, _mm256_mul_ps(sin_px, cos_py));

        __m256 mask_under = _mm256_cmp_ps(pz, terr_z, _CMP_LT_OQ);
        pz = _mm256_blendv_ps(pz, terr_z, mask_under);
        vx = _mm256_blendv_ps(vx, c0, mask_under);
        vy = _mm256_blendv_ps(vy, c0, mask_under);
        vz = _mm256_blendv_ps(vz, c0, mask_under);
    }

    _mm256_storeu_ps(&pos_x[i], px);
    _mm256_storeu_ps(&pos_y[i], py);
    _mm256_storeu_ps(&pos_z[i], pz);
    _mm256_storeu_ps(&vel_x[i], vx);
    _mm256_storeu_ps(&vel_y[i], vy);
    _mm256_storeu_ps(&vel_z[i], vz);
    _mm256_storeu_ps(&roll[i], r);
    _mm256_storeu_ps(&pitch[i], p);
    _mm256_storeu_ps(&yaw[i], y);

    // Final terrain check with LUT
    __m256 sin_px_f = lut_sin256_ps(_mm256_mul_ps(c01, px));
    __m256 cos_py_f = lut_cos256_ps(_mm256_mul_ps(c01, py));
    __m256 terr_z_final = _mm256_mul_ps(c5, _mm256_mul_ps(sin_px_f, cos_py_f));

    __m256 mask_coll = _mm256_cmp_ps(pz, terr_z_final, _CMP_LT_OQ);

    if (t <= episode_length) {
        int base_idx = (t-1) * num_agents * 3 + i * 3;
        float tmp_px[8], tmp_py[8], tmp_pz[8];
        _mm256_storeu_ps(tmp_px, px);
        _mm256_storeu_ps(tmp_py, py);
        _mm256_storeu_ps(tmp_pz, pz);

        for(int k=0; k<8; k++) {
            pos_history[base_idx + k*3 + 0] = tmp_px[k];
            pos_history[base_idx + k*3 + 1] = tmp_py[k];
            pos_history[base_idx + k*3 + 2] = tmp_pz[k];
        }
    }

    // ------------------------------------------------------------------------
    // Calculate Augmented Features & Update History
    // ------------------------------------------------------------------------
    __m256 dx_w = _mm256_sub_ps(vtx, px);
    __m256 dy_w = _mm256_sub_ps(vty, py);
    __m256 dz_w = _mm256_sub_ps(vtz, pz);

    // Rotation Matrix (LUT)
    __m256 sr = lut_sin256_ps(r);
    __m256 cr = lut_cos256_ps(r);
    __m256 sp = lut_sin256_ps(p);
    __m256 cp = lut_cos256_ps(p);
    __m256 sy = lut_sin256_ps(y);
    __m256 cy = lut_cos256_ps(y);

    __m256 r11 = _mm256_mul_ps(cy, cp);
    __m256 r12 = _mm256_mul_ps(sy, cp);
    __m256 r13 = _mm256_sub_ps(c0, sp);
    __m256 r21 = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(cy, sp), sr), _mm256_mul_ps(sy, cr));
    __m256 r22 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(sy, sp), sr), _mm256_mul_ps(cy, cr));
    __m256 r23 = _mm256_mul_ps(cp, sr);
    __m256 r31 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(cy, sp), cr), _mm256_mul_ps(sy, sr));
    __m256 r32 = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(sy, sp), cr), _mm256_mul_ps(cy, sr));
    __m256 r33 = _mm256_mul_ps(cp, cr);

    __m256 xb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, dx_w), _mm256_mul_ps(r12, dy_w)), _mm256_mul_ps(r13, dz_w));
    __m256 yb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r21, dx_w), _mm256_mul_ps(r22, dy_w)), _mm256_mul_ps(r23, dz_w));
    __m256 zb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r31, dx_w), _mm256_mul_ps(r32, dy_w)), _mm256_mul_ps(r33, dz_w));

    __m256 s30 = c05;
    __m256 c30 = _mm256_set1_ps(0.866025f);

    __m256 xc = yb;
    // Camera Up 30 deg:
    // yc = -s30 * xb + c30 * zb
    // zc = c30 * xb + s30 * zb
    __m256 yc = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(c0, s30), xb), _mm256_mul_ps(c30, zb));
    __m256 zc = _mm256_add_ps(_mm256_mul_ps(c30, xb), _mm256_mul_ps(s30, zb));

    __m256 zc_safe = _mm256_max_ps(zc, c01);
    __m256 u = _mm256_div_ps(xc, zc_safe);
    __m256 v = _mm256_div_ps(yc, zc_safe);

    __m256 c_fov_neg = _mm256_sub_ps(c0, c_fov);
    u = _mm256_max_ps(c_fov_neg, _mm256_min_ps(u, c_fov));
    v = _mm256_max_ps(c_fov_neg, _mm256_min_ps(v, c_fov));

    __m256 rel_size = _mm256_div_ps(c10, _mm256_add_ps(_mm256_mul_ps(zc, zc), c1));

    __m256 w2 = _mm256_add_ps(
        _mm256_mul_ps(roll_rate_cmd, roll_rate_cmd),
        _mm256_add_ps(
            _mm256_mul_ps(pitch_rate_cmd, pitch_rate_cmd),
            _mm256_mul_ps(yaw_rate_cmd, yaw_rate_cmd)
        )
    );
    __m256 conf = exp256_ps(_mm256_mul_ps(_mm256_set1_ps(-0.1f), w2));
    // Behind camera if zc < 0 (using new zc)
    __m256 mask_behind = _mm256_cmp_ps(zc, c0, _CMP_LT_OQ);
    conf = _mm256_blendv_ps(conf, c0, mask_behind);

    // Update History (last 9 slots)
    // Features: Thrust, Rates(3), Yaw, Pitch, Roll, u, v
    float tmp_r[8], tmp_p[8], tmp_y[8];
    float tmp_th[8], tmp_rr[8], tmp_pr[8], tmp_yr[8];
    float tmp_u[8], tmp_v[8];

    _mm256_storeu_ps(tmp_r, r);
    _mm256_storeu_ps(tmp_p, p);
    _mm256_storeu_ps(tmp_y, y);
    _mm256_storeu_ps(tmp_th, thrust_cmd);
    _mm256_storeu_ps(tmp_rr, roll_rate_cmd);
    _mm256_storeu_ps(tmp_pr, pitch_rate_cmd);
    _mm256_storeu_ps(tmp_yr, yaw_rate_cmd);
    _mm256_storeu_ps(tmp_u, u);
    _mm256_storeu_ps(tmp_v, v);

    for (int k=0; k<8; k++) {
        int agent_idx = i+k;
        int off = agent_idx*274 + 261; // Last 9 of 270
        observations[off+0] = tmp_th[k];
        observations[off+1] = tmp_rr[k];
        observations[off+2] = tmp_pr[k];
        observations[off+3] = tmp_yr[k];
        observations[off+4] = tmp_y[k];
        observations[off+5] = tmp_p[k];
        observations[off+6] = tmp_r[k];
        observations[off+7] = tmp_u[k];
        observations[off+8] = tmp_v[k];
    }

    // Update Aux Features (270-273)
    // Rel Vel REMOVED. Only tracker.

    // We still need Rel Vel for REWARDS.
    __m256 rvx = _mm256_sub_ps(vtvx, vx);
    __m256 rvy = _mm256_sub_ps(vtvy, vy);
    __m256 rvz = _mm256_sub_ps(vtvz, vz);

    // Body rel vel (for reward only)
    // __m256 rvx_b = ... (unused)
    // __m256 rvy_b = ... (not needed for reward in this master equation formulation actually? no, rvel_sq uses global)

    __m256 rx = _mm256_sub_ps(vtx, px);
    __m256 ry = _mm256_sub_ps(vty, py);
    __m256 rz = _mm256_sub_ps(vtz, pz);
    __m256 dist_sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry)), _mm256_mul_ps(rz, rz));
    __m256 dist = _mm256_sqrt_ps(dist_sq);
    __m256 dist_safe = _mm256_max_ps(dist, c01);

    float tmp_size[8], tmp_conf[8];
    _mm256_storeu_ps(tmp_size, rel_size);
    _mm256_storeu_ps(tmp_conf, conf);

    for(int k=0; k<8; k++) {
        int agent_idx = i+k;
        int off = agent_idx*274 + 270;
        observations[off+0] = tmp_u[k];
        observations[off+1] = tmp_v[k];
        observations[off+2] = tmp_size[k];
        observations[off+3] = tmp_conf[k];
    }

    // Rewards with Optimized Division (Newton-Raphson)
    __m256 rvel_sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rvx, rvx), _mm256_mul_ps(rvy, rvy)), _mm256_mul_ps(rvz, rvz));
    __m256 r_dot_v = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rx, rvx), _mm256_mul_ps(ry, rvy)), _mm256_mul_ps(rz, rvz));
    __m256 dist_sq_safe = _mm256_max_ps(dist_sq, _mm256_set1_ps(0.01f));

    // rcp_dist_sq = 1.0 / dist_sq_safe (Approx)
    __m256 rcp_dist_sq = _mm256_rcp_ps(dist_sq_safe);
    // Refinement: y = y * (2 - x * y)
    rcp_dist_sq = _mm256_mul_ps(rcp_dist_sq, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(dist_sq_safe, rcp_dist_sq)));

    // term1 = rvel_sq / dist_sq_safe
    __m256 term1 = _mm256_mul_ps(rvel_sq, rcp_dist_sq);

    // term2 = (r_dot_v / dist_sq_safe)^2
    __m256 r_dot_v_scaled = _mm256_mul_ps(r_dot_v, rcp_dist_sq);
    __m256 term2 = _mm256_mul_ps(r_dot_v_scaled, r_dot_v_scaled); // Slightly different algebraic expansion, but equivalent: (r.v)^2 / d^4 = (r.v/d^2)^2. Correct.

    __m256 omega_sq = _mm256_sub_ps(term1, term2);
    omega_sq = _mm256_max_ps(omega_sq, c0);
    __m256 rew_pn = _mm256_mul_ps(_mm256_set1_ps(-2.0f), omega_sq);

    __m256 vd_dot_r = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(vx, rx), _mm256_mul_ps(vy, ry)), _mm256_mul_ps(vz, rz));
    __m256 rcp_dist = _mm256_rcp_ps(dist_safe);
    rcp_dist = _mm256_mul_ps(rcp_dist, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(dist_safe, rcp_dist)));
    __m256 closing = _mm256_mul_ps(vd_dot_r, rcp_dist);
    __m256 rew_closing = _mm256_mul_ps(_mm256_set1_ps(0.5f), closing);

    __m256 vx_b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, vx), _mm256_mul_ps(r12, vy)), _mm256_mul_ps(r13, vz));
    __m256 v_ideal = _mm256_mul_ps(_mm256_set1_ps(0.1f), vx_b);
    __m256 v_err = _mm256_sub_ps(v, v_ideal);
    __m256 gaze_err = _mm256_add_ps(_mm256_mul_ps(u, u), _mm256_mul_ps(v_err, v_err));
    __m256 rew_gaze = _mm256_mul_ps(_mm256_set1_ps(-0.01f), gaze_err);

    __m256 d_plus_1 = _mm256_add_ps(dist, c1);
    __m256 funnel = _mm256_rcp_ps(d_plus_1);
    funnel = _mm256_mul_ps(funnel, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(d_plus_1, funnel)));
    __m256 rew_guidance = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(rew_pn, rew_gaze), rew_closing), funnel);

    __m256 rew_rate = _mm256_mul_ps(_mm256_set1_ps(-1.0f), w2);
    __m256 upright_err = _mm256_sub_ps(c1, r33);
    __m256 rew_upright = _mm256_mul_ps(_mm256_set1_ps(-5.0f), _mm256_mul_ps(upright_err, upright_err));

    __m256 diff_thrust = _mm256_sub_ps(_mm256_set1_ps(0.4f), thrust_cmd);
    diff_thrust = _mm256_max_ps(diff_thrust, c0);
    __m256 rew_eff = _mm256_mul_ps(_mm256_set1_ps(-10.0f), diff_thrust);

    __m256 rew = _mm256_add_ps(rew_guidance, _mm256_add_ps(rew_rate, _mm256_add_ps(rew_upright, rew_eff)));

    __m256 bonus_val = _mm256_set1_ps(10.0f);
    __m256 mask_success = _mm256_cmp_ps(dist, _mm256_set1_ps(0.4f), _CMP_LT_OQ);
    __m256 bonus = _mm256_and_ps(mask_success, bonus_val);
    rew = _mm256_add_ps(rew, bonus);

    __m256 mask_tilt = _mm256_cmp_ps(r33, c05, _CMP_LT_OQ);
    __m256 penalty = c0;
    penalty = _mm256_add_ps(penalty, _mm256_and_ps(mask_tilt, _mm256_set1_ps(10.0f)));
    penalty = _mm256_add_ps(penalty, _mm256_and_ps(mask_coll, _mm256_set1_ps(10.0f)));
    rew = _mm256_sub_ps(rew, penalty);

    _mm256_storeu_ps(&rewards[i], rew);

    float t_pn[8], t_cl[8], t_gz[8], t_rt[8], t_up[8], t_ef[8], t_pe[8], t_bo[8];
    _mm256_storeu_ps(t_pn, rew_pn);
    _mm256_storeu_ps(t_cl, rew_closing);
    _mm256_storeu_ps(t_gz, rew_gaze);
    _mm256_storeu_ps(t_rt, rew_rate);
    _mm256_storeu_ps(t_up, rew_upright);
    _mm256_storeu_ps(t_ef, rew_eff);
    _mm256_storeu_ps(t_pe, penalty);
    _mm256_storeu_ps(t_bo, bonus);

    for(int k=0; k<8; k++) {
        int agent_idx = i+k;
        int off = agent_idx*8;
        reward_components[off+0] = t_pn[k];
        reward_components[off+1] = t_cl[k];
        reward_components[off+2] = t_gz[k];
        reward_components[off+3] = t_rt[k];
        reward_components[off+4] = t_up[k];
        reward_components[off+5] = t_ef[k];
        reward_components[off+6] = -t_pe[k];
        reward_components[off+7] = t_bo[k];
    }

    __m256 mask_done = _mm256_or_ps(mask_success, _mm256_or_ps(mask_tilt, mask_coll));
    __m256 mask_timeout = c0;
    if (t >= episode_length) mask_timeout = c1;
    mask_done = _mm256_or_ps(mask_done, mask_timeout);
    __m256 done_val = _mm256_and_ps(mask_done, c1);
    _mm256_storeu_ps(&done_flags[i], done_val);
}

#endif
