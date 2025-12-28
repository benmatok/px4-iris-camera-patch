#ifndef PHYSICS_AVX_HPP
#define PHYSICS_AVX_HPP

#include <immintrin.h>
#include <cmath>
#include <cstring>
#include "avx_mathfun.h"

// Constants
static const float DT = 0.01f;
static const float GRAVITY = 9.81f;
static const int SUBSTEPS = 10;

// AVX2 Implementation for a block of 8 agents
inline void step_agents_avx2(
    int i,
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float* roll, float* pitch, float* yaw,
    float* masses, float* drag_coeffs, float* thrust_coeffs,
    float* target_vx, float* target_vy, float* target_vz, float* target_yaw_rate,
    float* vt_x, float* vt_y, float* vt_z, // Virtual Target Position
    float* traj_params, // Trajectory Parameters. Shape (10, num_agents).
    float* pos_history, // Shape (episode_length, num_agents, 3)
    float* observations, // stride 608
    float* rewards,
    float* done_flags,
    float* actions, // stride 4
    int episode_length,
    int t,
    int num_agents
) {
    // Check if we can process 8 agents
    if (i + 8 > num_agents) return;

    // Load State into Registers (Structure of Arrays)
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
    // Noise scaling: 0.04 (range +/- 0.02)
    // __m256 c_noise_scale = _mm256_set1_ps(0.04f);
    // __m256 c_noise_offset = _mm256_set1_ps(0.5f);

    // ------------------------------------------------------------------------
    // Update Virtual Target Position (Complex Trajectory)
    // ------------------------------------------------------------------------
    // traj_params: (10, num_agents).
    // Row 0: Ax, Row 1: Fx, ...
    // To access parameter X for agents i..i+7, we access traj_params[row * num_agents + i]
    // This is contiguous!

    int offset_i = i;
    int stride = num_agents;

    // Ax (0)
    __m256 ax = _mm256_loadu_ps(&traj_params[0 * stride + offset_i]);
    // Fx (1)
    __m256 fx = _mm256_loadu_ps(&traj_params[1 * stride + offset_i]);
    // Px (2)
    __m256 px_ph = _mm256_loadu_ps(&traj_params[2 * stride + offset_i]);

    // Ay (3)
    __m256 ay = _mm256_loadu_ps(&traj_params[3 * stride + offset_i]);
    // Fy (4)
    __m256 fy = _mm256_loadu_ps(&traj_params[4 * stride + offset_i]);
    // Py (5)
    __m256 py_ph = _mm256_loadu_ps(&traj_params[5 * stride + offset_i]);

    // Az (6)
    __m256 az = _mm256_loadu_ps(&traj_params[6 * stride + offset_i]);
    // Fz (7)
    __m256 fz = _mm256_loadu_ps(&traj_params[7 * stride + offset_i]);
    // Pz (8)
    __m256 pz_ph = _mm256_loadu_ps(&traj_params[8 * stride + offset_i]);
    // Oz (9)
    __m256 oz = _mm256_loadu_ps(&traj_params[9 * stride + offset_i]);

    float t_f = (float)t;
    __m256 t_v = _mm256_set1_ps(t_f);

    // x = Ax * sin(Fx * t + Px)
    __m256 arg_x = _mm256_add_ps(_mm256_mul_ps(fx, t_v), px_ph);
    __m256 sx, cx; sincos256_ps(arg_x, &sx, &cx);
    __m256 vtx = _mm256_mul_ps(ax, sx);
    __m256 vtvx = _mm256_mul_ps(ax, _mm256_mul_ps(fx, cx));

    // y = Ay * sin(Fy * t + Py)
    __m256 arg_y = _mm256_add_ps(_mm256_mul_ps(fy, t_v), py_ph);
    __m256 sy_t, cy_t; sincos256_ps(arg_y, &sy_t, &cy_t);
    __m256 vty = _mm256_mul_ps(ay, sy_t);
    __m256 vtvy = _mm256_mul_ps(ay, _mm256_mul_ps(fy, cy_t));

    // z = Oz + Az * sin(Fz * t + Pz)
    __m256 arg_z = _mm256_add_ps(_mm256_mul_ps(fz, t_v), pz_ph);
    __m256 sz, cz; sincos256_ps(arg_z, &sz, &cz);
    __m256 vtz = _mm256_add_ps(oz, _mm256_mul_ps(az, sz));
    __m256 vtvz = _mm256_mul_ps(az, _mm256_mul_ps(fz, cz));

    _mm256_storeu_ps(&vt_x[i], vtx);
    _mm256_storeu_ps(&vt_y[i], vty);
    _mm256_storeu_ps(&vt_z[i], vtz);

    // Shift Observations
    // New Size 608. Shift 6..600 to 0..594
    // 6 floats = 2 samples * 3 floats
    for (int k = 0; k < 8; k++) {
        int agent_idx = i + k;
        memmove(&observations[agent_idx * 608], &observations[agent_idx * 608 + 6], 594 * sizeof(float));
    }

    // Substeps
    for (int s = 0; s < SUBSTEPS; s++) {
        // Dynamics
        r = _mm256_add_ps(r, _mm256_mul_ps(roll_rate_cmd, dt_v));
        p = _mm256_add_ps(p, _mm256_mul_ps(pitch_rate_cmd, dt_v));
        y = _mm256_add_ps(y, _mm256_mul_ps(yaw_rate_cmd, dt_v));

        __m256 max_thrust = _mm256_mul_ps(c20, t_coeff);
        __m256 thrust_force = _mm256_mul_ps(thrust_cmd, max_thrust);

        __m256 sr, cr, sp, cp, sy, cy;
        sincos256_ps(r, &sr, &cr);
        sincos256_ps(p, &sp, &cp);
        sincos256_ps(y, &sy, &cy);

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

        __m256 terr_z = _mm256_mul_ps(c5, _mm256_mul_ps(
            sin256_ps(_mm256_mul_ps(c01, px)),
            cos256_ps(_mm256_mul_ps(c01, py))
        ));

        __m256 mask_under = _mm256_cmp_ps(pz, terr_z, _CMP_LT_OQ);
        pz = _mm256_blendv_ps(pz, terr_z, mask_under);
        vx = _mm256_blendv_ps(vx, c0, mask_under);
        vy = _mm256_blendv_ps(vy, c0, mask_under);
        vz = _mm256_blendv_ps(vz, c0, mask_under);

        // Capture Samples at s=4 and s=9
        if (s == 4 || s == 9) {
            int buffer_slot = (s == 4) ? 0 : 1;

            // Noise (scalar loop approx or implement AVX rand)
            // Scalar for simplicity (performance hit is minor compared to logic)
            float nr[8], np[8], ny[8];
            for (int k=0; k<8; k++) {
                nr[k] = (((float)rand() / (float)RAND_MAX) - 0.5f) * 0.04f;
                np[k] = (((float)rand() / (float)RAND_MAX) - 0.5f) * 0.04f;
                ny[k] = (((float)rand() / (float)RAND_MAX) - 0.5f) * 0.04f;
            }
            __m256 vnr = _mm256_loadu_ps(nr);
            __m256 vnp = _mm256_loadu_ps(np);
            __m256 vny = _mm256_loadu_ps(ny);

            __m256 r_noisy = _mm256_add_ps(r, vnr);
            __m256 p_noisy = _mm256_add_ps(p, vnp);
            __m256 y_noisy = _mm256_add_ps(y, vny);

            float tmp_r[8], tmp_p[8], tmp_y[8];
            _mm256_storeu_ps(tmp_r, r_noisy);
            _mm256_storeu_ps(tmp_p, p_noisy);
            _mm256_storeu_ps(tmp_y, y_noisy);

            for(int k=0; k<8; k++) {
                int agent_idx = i+k;
                int offset = agent_idx * 608 + 594 + buffer_slot * 3;
                observations[offset + 0] = tmp_r[k];
                observations[offset + 1] = tmp_p[k];
                observations[offset + 2] = tmp_y[k];
            }
        }
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

    __m256 terr_z_final = _mm256_mul_ps(c5, _mm256_mul_ps(
            sin256_ps(_mm256_mul_ps(c01, px)),
            cos256_ps(_mm256_mul_ps(c01, py))
    ));
    __m256 mask_coll = _mm256_cmp_ps(pz, terr_z_final, _CMP_LT_OQ);

    if (t <= episode_length) {
        // Optimized History Write: pos_history[t, i, 0..2]
        // This is a contiguous block for all agents at time t.
        // Wait, pos_history layout: (episode_length, num_agents, 3).
        // Memory offset: (t-1) * num_agents * 3 + i * 3
        // So for agents i..i+7, the writes are:
        // [i*3, i*3+1, i*3+2], [(i+1)*3...], ...
        // This is contiguous block of 8*3 = 24 floats.

        int base_idx = (t-1) * num_agents * 3 + i * 3;

        // We have px, py, pz registers.
        // We need to interleave them: x0, y0, z0, x1, y1, z1...
        // AVX doesn't have easy interleave for 3 streams.
        // So we just store to temp and memcpy or scalar assign.
        // Scalar assign is fine because it's writing to L1 cache (contiguous block).

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
    // Calculate Augmented Features (Tracker Simulation)
    // ------------------------------------------------------------------------
    __m256 dx_w = _mm256_sub_ps(vtx, px);
    __m256 dy_w = _mm256_sub_ps(vty, py);
    __m256 dz_w = _mm256_sub_ps(vtz, pz);

    // Recompute R matrix elements (r,p,y are updated)
    __m256 sr, cr, sp, cp, sy, cy;
    sincos256_ps(r, &sr, &cr);
    sincos256_ps(p, &sp, &cp);
    sincos256_ps(y, &sy, &cy);

    __m256 r11 = _mm256_mul_ps(cy, cp);
    __m256 r12 = _mm256_mul_ps(sy, cp);
    __m256 r13 = _mm256_sub_ps(c0, sp);
    __m256 r21 = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(cy, sp), sr), _mm256_mul_ps(sy, cr));
    __m256 r22 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(sy, sp), sr), _mm256_mul_ps(cy, cr));
    __m256 r23 = _mm256_mul_ps(cp, sr);
    __m256 r31 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(cy, sp), cr), _mm256_mul_ps(sy, sr));
    __m256 r32 = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(sy, sp), cr), _mm256_mul_ps(cy, sr));
    __m256 r33 = _mm256_mul_ps(cp, cr);

    // Transform to Body Frame: P_b = R^T * P_w
    __m256 xb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, dx_w), _mm256_mul_ps(r12, dy_w)), _mm256_mul_ps(r13, dz_w));
    __m256 yb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r21, dx_w), _mm256_mul_ps(r22, dy_w)), _mm256_mul_ps(r23, dz_w));
    __m256 zb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r31, dx_w), _mm256_mul_ps(r32, dy_w)), _mm256_mul_ps(r33, dz_w));

    // Transform to Camera Frame (Pitch Up 30 deg)
    __m256 s30 = c05;
    __m256 c30 = _mm256_set1_ps(0.866025f);

    __m256 xc = yb;
    __m256 yc = _mm256_add_ps(_mm256_mul_ps(s30, xb), _mm256_mul_ps(c30, zb));
    __m256 zc = _mm256_sub_ps(_mm256_mul_ps(c30, xb), _mm256_mul_ps(s30, zb));

    // Project (u = xc/zc, v = yc/zc)
    __m256 zc_safe = _mm256_max_ps(zc, c01); // min distance 0.1
    __m256 u = _mm256_div_ps(xc, zc_safe);
    __m256 v = _mm256_div_ps(yc, zc_safe);

    __m256 rel_size = _mm256_div_ps(c10, _mm256_add_ps(_mm256_mul_ps(zc, zc), c1));

    __m256 w2 = _mm256_add_ps(
        _mm256_mul_ps(roll_rate_cmd, roll_rate_cmd),
        _mm256_add_ps(
            _mm256_mul_ps(pitch_rate_cmd, pitch_rate_cmd),
            _mm256_mul_ps(yaw_rate_cmd, yaw_rate_cmd)
        )
    );
    __m256 conf = exp256_ps(_mm256_mul_ps(_mm256_set1_ps(-0.1f), w2));

    __m256 mask_behind = _mm256_cmp_ps(zc, c0, _CMP_LT_OQ);
    conf = _mm256_blendv_ps(conf, c0, mask_behind);

    // Store Obs
    float tmp_u[8], tmp_v[8], tmp_size[8], tmp_conf[8];
    _mm256_storeu_ps(tmp_u, u);
    _mm256_storeu_ps(tmp_v, v);
    _mm256_storeu_ps(tmp_size, rel_size);
    _mm256_storeu_ps(tmp_conf, conf);

    // Calculate Relative Velocity (World)
    __m256 rvx = _mm256_sub_ps(vtvx, vx);
    __m256 rvy = _mm256_sub_ps(vtvy, vy);
    __m256 rvz = _mm256_sub_ps(vtvz, vz);

    // Body Frame Relative Velocity
    __m256 rvx_b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, rvx), _mm256_mul_ps(r12, rvy)), _mm256_mul_ps(r13, rvz));
    __m256 rvy_b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r21, rvx), _mm256_mul_ps(r22, rvy)), _mm256_mul_ps(r23, rvz));
    __m256 rvz_b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r31, rvx), _mm256_mul_ps(r32, rvy)), _mm256_mul_ps(r33, rvz));

    // Distance
    __m256 rx = _mm256_sub_ps(vtx, px);
    __m256 ry = _mm256_sub_ps(vty, py);
    __m256 rz = _mm256_sub_ps(vtz, pz);
    __m256 dist_sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry)), _mm256_mul_ps(rz, rz));
    __m256 dist = _mm256_sqrt_ps(dist_sq);
    __m256 dist_safe = _mm256_max_ps(dist, c01);

    // Store State: Replace TargetCmds (600-603) with RelVel (600-602) and Distance (603)
    float tmp_rvx[8], tmp_rvy[8], tmp_rvz[8], tmp_dist[8];
    _mm256_storeu_ps(tmp_rvx, rvx_b);
    _mm256_storeu_ps(tmp_rvy, rvy_b);
    _mm256_storeu_ps(tmp_rvz, rvz_b);
    _mm256_storeu_ps(tmp_dist, dist);

    for(int k=0; k<8; k++) {
        int agent_idx = i+k;
        int off = agent_idx*608 + 600;
        observations[off] = tmp_rvx[k];
        observations[off+1] = tmp_rvy[k];
        observations[off+2] = tmp_rvz[k];
        observations[off+3] = tmp_dist[k];

        // 604-607 are tracker features
        observations[off+4] = tmp_u[k];
        observations[off+5] = tmp_v[k];
        observations[off+6] = tmp_size[k];
        observations[off+7] = tmp_conf[k];
    }

    // ------------------------------------------------------------------------
    // Homing Reward Logic (Master Equation)
    // ------------------------------------------------------------------------

    // 1. Guidance (PN)
    // |Omega|^2 = (v_rel^2 / d^2) - ((r . v_rel)^2 / d^4)
    __m256 rvel_sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rvx, rvx), _mm256_mul_ps(rvy, rvy)), _mm256_mul_ps(rvz, rvz));
    __m256 r_dot_v = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rx, rvx), _mm256_mul_ps(ry, rvy)), _mm256_mul_ps(rz, rvz));

    __m256 term1 = _mm256_div_ps(rvel_sq, dist_sq);
    __m256 term2_num = _mm256_mul_ps(r_dot_v, r_dot_v);
    __m256 term2_den = _mm256_mul_ps(dist_sq, dist_sq);
    __m256 term2 = _mm256_div_ps(term2_num, term2_den);

    __m256 omega_sq = _mm256_sub_ps(term1, term2);
    omega_sq = _mm256_max_ps(omega_sq, c0); // Clip

    __m256 rew_pn = _mm256_mul_ps(_mm256_set1_ps(-1.0f), omega_sq); // k1=1.0

    // 2. Closing Speed
    // V_drone . r_hat.
    // r_hat = r / d.
    __m256 vd_dot_r = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(vx, rx), _mm256_mul_ps(vy, ry)), _mm256_mul_ps(vz, rz));
    __m256 closing = _mm256_div_ps(vd_dot_r, dist_safe);
    __m256 rew_closing = _mm256_mul_ps(_mm256_set1_ps(0.5f), closing); // k2=0.5

    // 3. Vision (Gaze)
    // v_ideal = -0.1 * vx_b (heuristic: pitch forward -> vx_b > 0 -> target moves up -> v decreases?)
    // Actually, if nose down (pitch positive in this specific physics?), ax > 0.
    // Cam pitch up 30 deg.
    // We want target in center.
    // Velocity compensation: if flying fast, we are pitched down.
    // Target appears higher in frame (negative v?).
    // So ideal v is negative?
    __m256 vx_b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, vx), _mm256_mul_ps(r12, vy)), _mm256_mul_ps(r13, vz));
    __m256 v_ideal = _mm256_mul_ps(_mm256_set1_ps(0.1f), vx_b); // Tuning direction
    __m256 v_err = _mm256_sub_ps(v, v_ideal);
    __m256 gaze_err = _mm256_add_ps(_mm256_mul_ps(u, u), _mm256_mul_ps(v_err, v_err));
    __m256 rew_gaze = _mm256_mul_ps(_mm256_set1_ps(-1.0f), gaze_err); // k3=1.0

    // Funnel Scaling: 1 / (d + 1.0)
    __m256 funnel = _mm256_div_ps(c1, _mm256_add_ps(dist, c1));

    // Total Guidance + Gaze
    __m256 rew_guidance = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(rew_pn, rew_gaze), rew_closing), funnel);

    // 4. Stability
    // Rate damping
    __m256 rew_rate = _mm256_mul_ps(_mm256_set1_ps(-0.1f), w2);
    // Upright: (1 - r33)^2. r33 = cp*cr
    __m256 upright_err = _mm256_sub_ps(c1, r33);
    __m256 rew_upright = _mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_mul_ps(upright_err, upright_err));
    // Efficiency
    __m256 rew_eff = _mm256_mul_ps(_mm256_set1_ps(-0.01f), _mm256_mul_ps(thrust_cmd, thrust_cmd));

    __m256 rew = _mm256_add_ps(rew_guidance, _mm256_add_ps(rew_rate, _mm256_add_ps(rew_upright, rew_eff)));

    // Terminations
    // Success: dist < 0.2
    __m256 mask_success = _mm256_cmp_ps(dist, _mm256_set1_ps(0.2f), _CMP_LT_OQ);
    rew = _mm256_add_ps(rew, _mm256_and_ps(mask_success, _mm256_set1_ps(10.0f)));

    // Fail: Tilt > 60 deg (r33 < 0.5)
    __m256 mask_tilt = _mm256_cmp_ps(r33, c05, _CMP_LT_OQ);

    // Penalties
    rew = _mm256_sub_ps(rew, _mm256_and_ps(mask_tilt, _mm256_set1_ps(10.0f)));
    rew = _mm256_sub_ps(rew, _mm256_and_ps(mask_coll, _mm256_set1_ps(10.0f)));

    _mm256_storeu_ps(&rewards[i], rew);

    __m256 mask_done = _mm256_or_ps(mask_success, _mm256_or_ps(mask_tilt, mask_coll));

    // Handle episode length done
    __m256 mask_timeout = c0;
    if (t >= episode_length) mask_timeout = c1;
    mask_done = _mm256_or_ps(mask_done, mask_timeout);

    _mm256_storeu_ps(&done_flags[i], mask_done);
}

#endif
