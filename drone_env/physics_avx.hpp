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

// LCG RNG Helper
// Updates state and returns uniform float [0, 1]
inline __m256 avx_rng_float(__m256i& state) {
    __m256i A = _mm256_set1_epi32(1103515245);
    __m256i C = _mm256_set1_epi32(12345);
    state = _mm256_add_epi32(_mm256_mullo_epi32(state, A), C);

    // Use high bits for better randomness? Standard LCG usually takes bits 30..16
    // But float conversion usually wants 0..INT_MAX
    __m256i mask = _mm256_set1_epi32(0x7FFFFFFF);
    __m256i val_int = _mm256_and_si256(state, mask);
    __m256 val_f = _mm256_cvtepi32_ps(val_int);
    return _mm256_mul_ps(val_f, _mm256_set1_ps(4.65661287e-10f)); // 1.0 / 2147483647
}

// Approximate Normal Distribution (Mean 0, Std 1) using Sum of 4 Uniforms
// Variance of U[0,1] is 1/12. Sum of 4 has variance 4/12 = 1/3.
// We want variance 1.
// Let S = U1 + U2 + U3 + U4 - 2.0. Mean 0. Var 1/3.
// Multiply by sqrt(3) ~= 1.732 to get Std 1.
inline __m256 avx_rng_normal(__m256i& state) {
    __m256 u1 = avx_rng_float(state);
    __m256 u2 = avx_rng_float(state);
    __m256 u3 = avx_rng_float(state);
    __m256 u4 = avx_rng_float(state);

    __m256 sum = _mm256_add_ps(_mm256_add_ps(u1, u2), _mm256_add_ps(u3, u4));
    __m256 centered = _mm256_sub_ps(sum, _mm256_set1_ps(2.0f));
    return _mm256_mul_ps(centered, _mm256_set1_ps(1.73205f));
}

// Helper: Custom Memmove using AVX for new obs size (308)
inline void shift_observations_avx(float* observations, int i) {
    for (int k = 0; k < 8; k++) {
        float* ptr = &observations[(i + k) * 308];
        float* src = ptr + 10;
        float* dst = ptr;
        // Copy 290 floats
        for (int j = 0; j < 36; j++) {
            __m256 v = _mm256_loadu_ps(src + j * 8);
            _mm256_storeu_ps(dst + j * 8, v);
        }
        dst[288] = src[288];
        dst[289] = src[289];
    }
}

// AVX2 Implementation for a block of 8 agents
inline void step_agents_avx2(
    int i,
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float* roll, float* pitch, float* yaw,
    float* masses, float* drag_coeffs, float* thrust_coeffs,
    float* wind_x, float* wind_y, float* wind_z, // New
    float* target_vx, float* target_vy, float* target_vz, float* target_yaw_rate,
    float* vt_x, float* vt_y, float* vt_z,
    float* target_trajectory,
    float* pos_history,
    float* observations,
    float* rewards,
    float* reward_components,
    float* done_flags,
    float* actions,
    float* action_buffer, // stride 44 (11*4)
    int* delays,          // stride 1
    int* rng_states,      // stride 1
    int episode_length,
    int t,
    int num_agents
) {
    if (i + 8 > num_agents) return;

    // Load RNG State
    __m256i rng_state = _mm256_loadu_si256((__m256i*)&rng_states[i]);

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

    __m256 wx = _mm256_loadu_ps(&wind_x[i]);
    __m256 wy = _mm256_loadu_ps(&wind_y[i]);
    __m256 wz = _mm256_loadu_ps(&wind_z[i]);

    // ------------------------------------------------------------------------
    // Action Buffer & Delay Logic
    // ------------------------------------------------------------------------
    // 1. Shift buffer for each agent (Manual loop as it's scattered in memory)
    // 2. Insert new action at index 0

    // Gather current actions first
    __m256i idx_base = _mm256_set_epi32(7*4, 6*4, 5*4, 4*4, 3*4, 2*4, 1*4, 0);
    __m256i idx_0 = _mm256_add_epi32(idx_base, _mm256_set1_epi32(i*4));
    __m256i idx_1 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(1));
    __m256i idx_2 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(2));
    __m256i idx_3 = _mm256_add_epi32(idx_0, _mm256_set1_epi32(3));

    __m256 act_0 = _mm256_i32gather_ps(actions, idx_0, 4);
    __m256 act_1 = _mm256_i32gather_ps(actions, idx_1, 4);
    __m256 act_2 = _mm256_i32gather_ps(actions, idx_2, 4);
    __m256 act_3 = _mm256_i32gather_ps(actions, idx_3, 4);

    // Store back to temporary arrays to handle shifting
    float tmp_act[4][8];
    _mm256_storeu_ps(tmp_act[0], act_0);
    _mm256_storeu_ps(tmp_act[1], act_1);
    _mm256_storeu_ps(tmp_act[2], act_2);
    _mm256_storeu_ps(tmp_act[3], act_3);

    for(int k=0; k<8; k++) {
        int agent_idx = i + k;
        float* buf = &action_buffer[agent_idx * 44];
        // Shift 0..9 (40 floats) to 1..10
        // Use memmove for safety (overlapping)
        std::memmove(buf + 4, buf, 40 * sizeof(float));
        // Insert new
        buf[0] = tmp_act[0][k];
        buf[1] = tmp_act[1][k];
        buf[2] = tmp_act[2][k];
        buf[3] = tmp_act[3][k];
    }

    // Load Delays
    __m256i delay_vals = _mm256_loadu_si256((__m256i*)&delays[i]);
    // Clamp delays to [0, 10]
    delay_vals = _mm256_max_epi32(delay_vals, _mm256_setzero_si256());
    delay_vals = _mm256_min_epi32(delay_vals, _mm256_set1_epi32(10));

    // Compute gather offsets for Action Buffer
    // Base ptr: action_buffer
    // For agent k (0..7): Offset = (i+k)*44 + delay[k]*4 + comp

    // We construct the indices manually
    __m256i stride_44 = _mm256_set1_epi32(44);
    __m256i stride_4 = _mm256_set1_epi32(4);

    __m256i agent_indices = _mm256_set_epi32(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i);
    __m256i base_offsets = _mm256_mullo_epi32(agent_indices, stride_44);
    __m256i delay_offsets = _mm256_mullo_epi32(delay_vals, stride_4);

    __m256i total_offset_base = _mm256_add_epi32(base_offsets, delay_offsets);

    __m256 thrust_cmd = _mm256_i32gather_ps(action_buffer, total_offset_base, 4);
    __m256 roll_rate_cmd = _mm256_i32gather_ps(action_buffer, _mm256_add_epi32(total_offset_base, _mm256_set1_epi32(1)), 4);
    __m256 pitch_rate_cmd = _mm256_i32gather_ps(action_buffer, _mm256_add_epi32(total_offset_base, _mm256_set1_epi32(2)), 4);
    __m256 yaw_rate_cmd = _mm256_i32gather_ps(action_buffer, _mm256_add_epi32(total_offset_base, _mm256_set1_epi32(3)), 4);

    // ------------------------------------------------------------------------
    // Wind Update
    // ------------------------------------------------------------------------
    // Removed wind noise as per instruction.
    // Wind remains constant (0.0 initialized).

    _mm256_storeu_ps(&wind_x[i], wx);
    _mm256_storeu_ps(&wind_y[i], wy);
    _mm256_storeu_ps(&wind_z[i], wz);

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
    __m256 c_fov = _mm256_set1_ps(1.732f);

    // Lookup Virtual Target
    int step_idx = t;
    if (step_idx > episode_length) step_idx = episode_length;

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

    shift_observations_avx(observations, i);

    // Substeps
    for (int s = 0; s < SUBSTEPS; s++) {
        r = _mm256_add_ps(r, _mm256_mul_ps(roll_rate_cmd, dt_v));
        p = _mm256_add_ps(p, _mm256_mul_ps(pitch_rate_cmd, dt_v));
        y = _mm256_add_ps(y, _mm256_mul_ps(yaw_rate_cmd, dt_v));

        __m256 max_thrust = _mm256_mul_ps(c20, t_coeff);
        __m256 thrust_force = _mm256_mul_ps(thrust_cmd, max_thrust);

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

        // Relative Velocity for Drag
        __m256 rvx_a = _mm256_sub_ps(vx, wx);
        __m256 rvy_a = _mm256_sub_ps(vy, wy);
        __m256 rvz_a = _mm256_sub_ps(vz, wz);

        __m256 ax_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), rvx_a);
        __m256 ay_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), rvy_a);
        __m256 az_drag = _mm256_mul_ps(_mm256_sub_ps(c0, drag), rvz_a);

        __m256 ax = _mm256_add_ps(ax_thrust, ax_drag);
        __m256 ay = _mm256_add_ps(ay_thrust, ay_drag);
        __m256 az = _mm256_add_ps(az_thrust, _mm256_add_ps(az_gravity, az_drag));

        vx = _mm256_add_ps(vx, _mm256_mul_ps(ax, dt_v));
        vy = _mm256_add_ps(vy, _mm256_mul_ps(ay, dt_v));
        vz = _mm256_add_ps(vz, _mm256_mul_ps(az, dt_v));

        px = _mm256_add_ps(px, _mm256_mul_ps(vx, dt_v));
        py = _mm256_add_ps(py, _mm256_mul_ps(vy, dt_v));
        pz = _mm256_add_ps(pz, _mm256_mul_ps(vz, dt_v));

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

    // Augmented Features
    __m256 dx_w = _mm256_sub_ps(vtx, px);
    __m256 dy_w = _mm256_sub_ps(vty, py);
    __m256 dz_w = _mm256_sub_ps(vtz, pz);

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
    __m256 yc = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(c0, s30), xb), _mm256_mul_ps(c30, zb));
    __m256 zc = _mm256_add_ps(_mm256_mul_ps(c30, xb), _mm256_mul_ps(s30, zb));

    __m256 zc_safe = _mm256_max_ps(zc, c01);
    __m256 u = _mm256_div_ps(xc, zc_safe);
    __m256 v = _mm256_div_ps(yc, zc_safe);

    // ------------------------------------------------------------------------
    // Noise on Tracking (u, v)
    // ------------------------------------------------------------------------
    // Add noise: ~5%? or fixed std dev?
    // Using Normal Distribution approx std=0.05
    __m256 u_noise = avx_rng_normal(rng_state);
    __m256 v_noise = avx_rng_normal(rng_state);
    __m256 c_noise_scale = _mm256_set1_ps(0.05f);

    u = _mm256_add_ps(u, _mm256_mul_ps(u_noise, c_noise_scale));
    v = _mm256_add_ps(v, _mm256_mul_ps(v_noise, c_noise_scale));

    __m256 c_fov_neg = _mm256_sub_ps(c0, c_fov);
    u = _mm256_max_ps(c_fov_neg, _mm256_min_ps(u, c_fov));
    v = _mm256_max_ps(c_fov_neg, _mm256_min_ps(v, c_fov));

    __m256 rel_size = _mm256_div_ps(c10, _mm256_add_ps(_mm256_mul_ps(zc, zc), c1));
    // Noise on size
    __m256 s_noise = avx_rng_normal(rng_state);
    rel_size = _mm256_add_ps(rel_size, _mm256_mul_ps(s_noise, _mm256_set1_ps(0.01f)));

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

    // Save RNG State
    _mm256_storeu_si256((__m256i*)&rng_states[i], rng_state);

    float tmp_r[8], tmp_p[8], tmp_y[8], tmp_pz[8];
    float tmp_th[8], tmp_rr[8], tmp_pr[8], tmp_yr[8];
    float tmp_u[8], tmp_v[8];

    _mm256_storeu_ps(tmp_r, r);
    _mm256_storeu_ps(tmp_p, p);
    _mm256_storeu_ps(tmp_y, y);
    _mm256_storeu_ps(tmp_pz, pz);
    _mm256_storeu_ps(tmp_th, thrust_cmd);
    _mm256_storeu_ps(tmp_rr, roll_rate_cmd);
    _mm256_storeu_ps(tmp_pr, pitch_rate_cmd);
    _mm256_storeu_ps(tmp_yr, yaw_rate_cmd);
    _mm256_storeu_ps(tmp_u, u);
    _mm256_storeu_ps(tmp_v, v);

    for (int k=0; k<8; k++) {
        int agent_idx = i+k;
        int off = agent_idx*308 + 290;
        observations[off+0] = tmp_r[k];
        observations[off+1] = tmp_p[k];
        observations[off+2] = tmp_y[k];
        observations[off+3] = tmp_pz[k];
        observations[off+4] = tmp_th[k];
        observations[off+5] = tmp_rr[k];
        observations[off+6] = tmp_pr[k];
        observations[off+7] = tmp_yr[k];
        observations[off+8] = tmp_u[k];
        observations[off+9] = tmp_v[k];
    }

    __m256 rvx = _mm256_sub_ps(vtvx, vx);
    __m256 rvy = _mm256_sub_ps(vtvy, vy);
    __m256 rvz = _mm256_sub_ps(vtvz, vz);

    __m256 rvx_b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r11, rvx), _mm256_mul_ps(r12, rvy)), _mm256_mul_ps(r13, rvz));
    __m256 rvy_b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r21, rvx), _mm256_mul_ps(r22, rvy)), _mm256_mul_ps(r23, rvz));
    __m256 rvz_b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r31, rvx), _mm256_mul_ps(r32, rvy)), _mm256_mul_ps(r33, rvz));

    __m256 rx = _mm256_sub_ps(vtx, px);
    __m256 ry = _mm256_sub_ps(vty, py);
    __m256 rz = _mm256_sub_ps(vtz, pz);
    __m256 dist_sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry)), _mm256_mul_ps(rz, rz));
    __m256 dist = _mm256_sqrt_ps(dist_sq);
    __m256 dist_safe = _mm256_max_ps(dist, c01);

    float tmp_rvx[8], tmp_rvy[8], tmp_rvz[8], tmp_dist[8];
    float tmp_size[8], tmp_conf[8];
    _mm256_storeu_ps(tmp_rvx, rvx_b);
    _mm256_storeu_ps(tmp_rvy, rvy_b);
    _mm256_storeu_ps(tmp_rvz, rvz_b);
    _mm256_storeu_ps(tmp_dist, dist);
    _mm256_storeu_ps(tmp_size, rel_size);
    _mm256_storeu_ps(tmp_conf, conf);

    for(int k=0; k<8; k++) {
        int agent_idx = i+k;
        int off = agent_idx*308 + 300;
        observations[off] = tmp_rvx[k];
        observations[off+1] = tmp_rvy[k];
        observations[off+2] = tmp_rvz[k];
        observations[off+3] = tmp_dist[k];
        observations[off+4] = tmp_u[k];
        observations[off+5] = tmp_v[k];
        observations[off+6] = tmp_size[k];
        observations[off+7] = tmp_conf[k];
    }

    // Rewards
    __m256 rvel_sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rvx, rvx), _mm256_mul_ps(rvy, rvy)), _mm256_mul_ps(rvz, rvz));
    __m256 r_dot_v = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rx, rvx), _mm256_mul_ps(ry, rvy)), _mm256_mul_ps(rz, rvz));
    __m256 dist_sq_safe = _mm256_max_ps(dist_sq, _mm256_set1_ps(0.01f));

    __m256 rcp_dist_sq = _mm256_rcp_ps(dist_sq_safe);
    rcp_dist_sq = _mm256_mul_ps(rcp_dist_sq, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(dist_sq_safe, rcp_dist_sq)));

    __m256 term1 = _mm256_mul_ps(rvel_sq, rcp_dist_sq);
    __m256 r_dot_v_scaled = _mm256_mul_ps(r_dot_v, rcp_dist_sq);
    __m256 term2 = _mm256_mul_ps(r_dot_v_scaled, r_dot_v_scaled);

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
    __m256 mask_success = _mm256_cmp_ps(dist, _mm256_set1_ps(0.2f), _CMP_LT_OQ);
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
