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
    float* pos_history,
    float* observations, // stride 1804
    float* rewards,
    float* done_flags,
    float* actions, // stride 4
    int episode_length,
    int t,
    int num_agents
) {
    // Check if we can process 8 agents
    if (i + 8 > num_agents) return; // Should be handled by caller loop

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

    // Actions are interleaved: [thrust, roll_rate, pitch_rate, yaw_rate] per agent
    __m256i idx_base = _mm256_set_epi32(7*4, 6*4, 5*4, 4*4, 3*4, 2*4, 1*4, 0);
    __m256i idx_0 = _mm256_add_epi32(idx_base, _mm256_set1_epi32(i*4));
    __m256i idx_1 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(1));
    __m256i idx_2 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(2));
    __m256i idx_3 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(3));

    __m256 thrust_cmd = _mm256_i32gather_ps(actions, idx_0, 4);
    __m256 roll_rate_cmd = _mm256_i32gather_ps(actions, idx_1, 4);
    __m256 pitch_rate_cmd = _mm256_i32gather_ps(actions, idx_2, 4);
    __m256 yaw_rate_cmd = _mm256_i32gather_ps(actions, idx_3, 4);

    // Constants vectors
    __m256 dt_v = _mm256_set1_ps(DT);
    __m256 g_v = _mm256_set1_ps(9.81f);
    __m256 c20 = _mm256_set1_ps(20.0f);
    __m256 c0 = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 c5 = _mm256_set1_ps(5.0f);
    __m256 c01 = _mm256_set1_ps(0.1f);

    // Shift Observations
    for (int k = 0; k < 8; k++) {
        int agent_idx = i + k;
        memmove(&observations[agent_idx * 1804], &observations[agent_idx * 1804 + 60], 1740 * sizeof(float));
    }

    // Cached values from last substep
    __m256 final_sr = c0, final_cr = one, final_sp = c0, final_cp = one, final_sy = c0, final_cy = one;
    __m256 mask_coll_final = c0;

    // Substeps
    for (int s = 0; s < SUBSTEPS; s++) {
        // Dynamics
        r = _mm256_add_ps(r, _mm256_mul_ps(roll_rate_cmd, dt_v));
        p = _mm256_add_ps(p, _mm256_mul_ps(pitch_rate_cmd, dt_v));
        y = _mm256_add_ps(y, _mm256_mul_ps(yaw_rate_cmd, dt_v));

        // Thrust
        __m256 max_thrust = _mm256_mul_ps(c20, t_coeff);
        __m256 thrust_force = _mm256_mul_ps(thrust_cmd, max_thrust);

        // Trig
        __m256 sr, cr, sp, cp, sy, cy;
        sincos256_ps(r, &sr, &cr);
        sincos256_ps(p, &sp, &cp);
        sincos256_ps(y, &sy, &cy);

        // Cache if last step
        if (s == SUBSTEPS - 1) {
            final_sr = sr; final_cr = cr;
            final_sp = sp; final_cp = cp;
            final_sy = sy; final_cy = cy;
        }

        // R Matrix Components
        __m256 term1_x = _mm256_mul_ps(_mm256_mul_ps(cy, sp), cr);
        __m256 term2_x = _mm256_mul_ps(sy, sr);
        __m256 ax_thrust = _mm256_div_ps(_mm256_mul_ps(thrust_force, _mm256_add_ps(term1_x, term2_x)), mass);

        __m256 term1_y = _mm256_mul_ps(_mm256_mul_ps(sy, sp), cr);
        __m256 term2_y = _mm256_mul_ps(cy, sr);
        __m256 ay_thrust = _mm256_div_ps(_mm256_mul_ps(thrust_force, _mm256_sub_ps(term1_y, term2_y)), mass);

        __m256 az_thrust = _mm256_div_ps(_mm256_mul_ps(thrust_force, _mm256_mul_ps(cp, cr)), mass);

        // Gravity
        __m256 az_gravity = _mm256_sub_ps(c0, g_v);

        // Drag
        __m256 ax_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), vx);
        __m256 ay_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), vy);
        __m256 az_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), vz);

        // Total Acc
        __m256 ax = _mm256_add_ps(ax_thrust, ax_drag);
        __m256 ay = _mm256_add_ps(ay_thrust, ay_drag);
        __m256 az = _mm256_add_ps(az_thrust, _mm256_add_ps(az_gravity, az_drag));

        // Integration
        vx = _mm256_add_ps(vx, _mm256_mul_ps(ax, dt_v));
        vy = _mm256_add_ps(vy, _mm256_mul_ps(ay, dt_v));
        vz = _mm256_add_ps(vz, _mm256_mul_ps(az, dt_v));

        px = _mm256_add_ps(px, _mm256_mul_ps(vx, dt_v));
        py = _mm256_add_ps(py, _mm256_mul_ps(vy, dt_v));
        pz = _mm256_add_ps(pz, _mm256_mul_ps(vz, dt_v));

        // Terrain
        __m256 terr_z = _mm256_mul_ps(c5, _mm256_mul_ps(
            sin256_ps(_mm256_mul_ps(c01, px)),
            cos256_ps(_mm256_mul_ps(c01, py))
        ));

        // Collision Check
        __m256 mask_under = _mm256_cmp_ps(pz, terr_z, _CMP_LT_OQ);

        // Handle Collision
        pz = _mm256_blendv_ps(pz, terr_z, mask_under);
        vx = _mm256_blendv_ps(vx, c0, mask_under);
        vy = _mm256_blendv_ps(vy, c0, mask_under);
        vz = _mm256_blendv_ps(vz, c0, mask_under);

        // Cache if last step
        if (s == SUBSTEPS - 1) {
            mask_coll_final = mask_under;
        }

        // IMU Capture
        __m256 r11 = _mm256_mul_ps(cy, cp);
        __m256 r12 = _mm256_mul_ps(sy, cp);
        __m256 r13 = _mm256_sub_ps(c0, sp);

        __m256 r21 = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(cy, sp), sr), _mm256_mul_ps(sy, cr));
        __m256 r22 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(sy, sp), sr), _mm256_mul_ps(cy, cr));
        __m256 r23 = _mm256_mul_ps(cp, sr);

        __m256 r31 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(cy, sp), cr), _mm256_mul_ps(sy, sr));
        __m256 r32 = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(sy, sp), cr), _mm256_mul_ps(cy, sr));
        __m256 r33 = _mm256_mul_ps(cp, cr);

        __m256 awx = _mm256_add_ps(ax_thrust, ax_drag);
        __m256 awy = _mm256_add_ps(ay_thrust, ay_drag);
        __m256 awz = _mm256_add_ps(az_thrust, az_drag);

        __m256 abx = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, awx), _mm256_mul_ps(r12, awy)), _mm256_mul_ps(r13, awz));
        __m256 aby = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r21, awx), _mm256_mul_ps(r22, awy)), _mm256_mul_ps(r23, awz));
        __m256 abz = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r31, awx), _mm256_mul_ps(r32, awy)), _mm256_mul_ps(r33, awz));

        // Store to observations buffer
        float tmp_abx[8], tmp_aby[8], tmp_abz[8];
        _mm256_storeu_ps(tmp_abx, abx);
        _mm256_storeu_ps(tmp_aby, aby);
        _mm256_storeu_ps(tmp_abz, abz);

        float tmp_rr[8], tmp_pr[8], tmp_yr[8];
        _mm256_storeu_ps(tmp_rr, roll_rate_cmd);
        _mm256_storeu_ps(tmp_pr, pitch_rate_cmd);
        _mm256_storeu_ps(tmp_yr, yaw_rate_cmd);

        for (int k=0; k<8; k++) {
            int agent_idx = i + k;
            int offset = agent_idx * 1804 + 1740 + s * 6;
            observations[offset + 0] = tmp_abx[k];
            observations[offset + 1] = tmp_aby[k];
            observations[offset + 2] = tmp_abz[k];
            observations[offset + 3] = tmp_rr[k];
            observations[offset + 4] = tmp_pr[k];
            observations[offset + 5] = tmp_yr[k];
        }
    }

    // Store State Back
    _mm256_storeu_ps(&pos_x[i], px);
    _mm256_storeu_ps(&pos_y[i], py);
    _mm256_storeu_ps(&pos_z[i], pz);
    _mm256_storeu_ps(&vel_x[i], vx);
    _mm256_storeu_ps(&vel_y[i], vy);
    _mm256_storeu_ps(&vel_z[i], vz);
    _mm256_storeu_ps(&roll[i], r);
    _mm256_storeu_ps(&pitch[i], p);
    _mm256_storeu_ps(&yaw[i], y);

    // Store Pos History
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

    // Load Targets
    __m256 tvx = _mm256_loadu_ps(&target_vx[i]);
    __m256 tvy = _mm256_loadu_ps(&target_vy[i]);
    __m256 tvz = _mm256_loadu_ps(&target_vz[i]);
    __m256 tyr = _mm256_loadu_ps(&target_yaw_rate[i]);

    // Update Observations with targets
    float tmp_tvx[8], tmp_tvy[8], tmp_tvz[8], tmp_tyr[8];
    _mm256_storeu_ps(tmp_tvx, tvx);
    _mm256_storeu_ps(tmp_tvy, tvy);
    _mm256_storeu_ps(tmp_tvz, tvz);
    _mm256_storeu_ps(tmp_tyr, tyr);
    for(int k=0; k<8; k++) {
        int agent_idx = i+k;
        observations[agent_idx*1804 + 1800] = tmp_tvx[k];
        observations[agent_idx*1804 + 1801] = tmp_tvy[k];
        observations[agent_idx*1804 + 1802] = tmp_tvz[k];
        observations[agent_idx*1804 + 1803] = tmp_tyr[k];
    }

    // Reward math
    // Reuse cached trig values
    __m256 sr = final_sr; __m256 cr = final_cr;
    __m256 sp = final_sp; __m256 cp = final_cp;
    __m256 sy = final_sy; __m256 cy = final_cy;

    // Body Velocity
    __m256 r11 = _mm256_mul_ps(cy, cp);
    __m256 r12 = _mm256_mul_ps(sy, cp);
    __m256 r13 = _mm256_sub_ps(c0, sp);
    __m256 r21 = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(cy, sp), sr), _mm256_mul_ps(sy, cr));
    __m256 r22 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(sy, sp), sr), _mm256_mul_ps(cy, cr));
    __m256 r23 = _mm256_mul_ps(cp, sr);
    __m256 r31 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(cy, sp), cr), _mm256_mul_ps(sy, sr));
    __m256 r32 = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(sy, sp), cr), _mm256_mul_ps(cy, sr));
    __m256 r33 = _mm256_mul_ps(cp, cr);

    __m256 vxb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, vx), _mm256_mul_ps(r12, vy)), _mm256_mul_ps(r13, vz));
    __m256 vyb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r21, vx), _mm256_mul_ps(r22, vy)), _mm256_mul_ps(r23, vz));
    __m256 vzb = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r31, vx), _mm256_mul_ps(r32, vy)), _mm256_mul_ps(r33, vz));

    __m256 dx = _mm256_sub_ps(vxb, tvx);
    __m256 dy = _mm256_sub_ps(vyb, tvy);
    __m256 dz = _mm256_sub_ps(vzb, tvz);
    __m256 v_err_sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz));

    __m256 dyr = _mm256_sub_ps(yaw_rate_cmd, tyr);
    __m256 y_err_sq = _mm256_mul_ps(dyr, dyr);

    // exp(-2 * err)
    __m256 rew = _mm256_add_ps(
        exp256_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), v_err_sq)),
        _mm256_mul_ps(_mm256_set1_ps(0.5f), exp256_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), y_err_sq)))
    );

    // Penalty: -0.01 * (r^2 + p^2)
    rew = _mm256_sub_ps(rew, _mm256_mul_ps(c01, _mm256_mul_ps(c01, _mm256_add_ps(_mm256_mul_ps(r, r), _mm256_mul_ps(p, p)))));

    // Unstable penalty
    // __m256 one = _mm256_set1_ps(1.0f); // Already defined
    __m256 m_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 ar = _mm256_and_ps(r, m_abs);
    __m256 ap = _mm256_and_ps(p, m_abs);
    __m256 mask_unst = _mm256_or_ps(
        _mm256_cmp_ps(ar, one, _CMP_GT_OQ),
        _mm256_cmp_ps(ap, one, _CMP_GT_OQ)
    );

    rew = _mm256_sub_ps(rew, _mm256_and_ps(mask_unst, c01));

    // Collision penalty 10.0 using cached mask
    __m256 c10 = _mm256_set1_ps(10.0f);
    rew = _mm256_sub_ps(rew, _mm256_and_ps(mask_coll_final, c10));

    // Survival +0.1
    rew = _mm256_add_ps(rew, c01);

    _mm256_storeu_ps(&rewards[i], rew);

    // Done flags
    if (t >= episode_length) {
        _mm256_storeu_ps(&done_flags[i], one);
    } else {
        _mm256_storeu_ps(&done_flags[i], c0);
    }
}

#endif
