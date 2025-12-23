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
    float* pos_history,
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
    __m256 c_noise_scale = _mm256_set1_ps(0.04f);
    __m256 c_noise_offset = _mm256_set1_ps(0.5f);

    // Update Virtual Target Position
    float t_f = (float)t;
    __m256 vtx = _mm256_set1_ps(5.0f * std::sin(0.05f * t_f));
    __m256 vty = _mm256_set1_ps(5.0f * std::cos(0.05f * t_f));
    __m256 vtz = _mm256_set1_ps(10.0f + 2.0f * std::sin(0.1f * t_f));

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
        float tmp_px[8], tmp_py[8], tmp_pz[8];
        _mm256_storeu_ps(tmp_px, px);
        _mm256_storeu_ps(tmp_py, py);
        _mm256_storeu_ps(tmp_pz, pz);
        for(int k=0; k<8; k++) {
            int agent_idx = i+k;
            int idx = agent_idx * episode_length * 3 + (t-1)*3;
            pos_history[idx+0] = tmp_px[k];
            pos_history[idx+1] = tmp_py[k];
            pos_history[idx+2] = tmp_pz[k];
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

    __m256 tvx = _mm256_loadu_ps(&target_vx[i]);
    __m256 tvy = _mm256_loadu_ps(&target_vy[i]);
    __m256 tvz = _mm256_loadu_ps(&target_vz[i]);
    __m256 tyr = _mm256_loadu_ps(&target_yaw_rate[i]);

    float tmp_tvx[8], tmp_tvy[8], tmp_tvz[8], tmp_tyr[8];
    _mm256_storeu_ps(tmp_tvx, tvx);
    _mm256_storeu_ps(tmp_tvy, tvy);
    _mm256_storeu_ps(tmp_tvz, tvz);
    _mm256_storeu_ps(tmp_tyr, tyr);

    for(int k=0; k<8; k++) {
        int agent_idx = i+k;
        int off = agent_idx*608 + 600; // 600-603 commands
        observations[off] = tmp_tvx[k];
        observations[off+1] = tmp_tvy[k];
        observations[off+2] = tmp_tvz[k];
        observations[off+3] = tmp_tyr[k];

        // 604-607 are tracker features
        observations[off+4] = tmp_u[k];
        observations[off+5] = tmp_v[k];
        observations[off+6] = tmp_size[k];
        observations[off+7] = tmp_conf[k];
    }

    // Reward Logic (Reuse previous logic)
    __m256 vxb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, vx), _mm256_mul_ps(r12, vy)), _mm256_mul_ps(r13, vz));
    __m256 vyb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r21, vx), _mm256_mul_ps(r22, vy)), _mm256_mul_ps(r23, vz));
    __m256 vzb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r31, vx), _mm256_mul_ps(r32, vy)), _mm256_mul_ps(r33, vz));

    __m256 dx = _mm256_sub_ps(vxb, tvx);
    __m256 dy = _mm256_sub_ps(vyb, tvy);
    __m256 dz = _mm256_sub_ps(vzb, tvz);
    __m256 v_err_sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz));

    __m256 dyr = _mm256_sub_ps(yaw_rate_cmd, tyr);
    __m256 y_err_sq = _mm256_mul_ps(dyr, dyr);

    __m256 rew = _mm256_add_ps(
        exp256_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), v_err_sq)),
        _mm256_mul_ps(c05, exp256_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), y_err_sq)))
    );

    rew = _mm256_sub_ps(rew, _mm256_mul_ps(c01, _mm256_mul_ps(c01, _mm256_add_ps(_mm256_mul_ps(r, r), _mm256_mul_ps(p, p)))));

    __m256 m_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 ar = _mm256_and_ps(r, m_abs);
    __m256 ap = _mm256_and_ps(p, m_abs);
    __m256 mask_unst = _mm256_or_ps(
        _mm256_cmp_ps(ar, c1, _CMP_GT_OQ),
        _mm256_cmp_ps(ap, c1, _CMP_GT_OQ)
    );

    rew = _mm256_sub_ps(rew, _mm256_and_ps(mask_unst, c01));
    rew = _mm256_sub_ps(rew, _mm256_and_ps(mask_coll, c10));
    rew = _mm256_add_ps(rew, c01);

    _mm256_storeu_ps(&rewards[i], rew);

    if (t >= episode_length) {
        _mm256_storeu_ps(&done_flags[i], c1);
    } else {
        _mm256_storeu_ps(&done_flags[i], c0);
    }
}

#endif
