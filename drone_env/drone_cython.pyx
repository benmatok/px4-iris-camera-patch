# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, exp, fabs, sqrt, M_PI
# sincosf is not in standard libc.math pxd, need extern
cdef extern from "math.h" nogil:
    void sincosf(float x, float *sin, float *cos)
    float atan2f(float y, float x)

from libc.stdlib cimport rand, RAND_MAX
from libc.string cimport memmove, memset
from cython.parallel import prange

# Define float32 type for numpy
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t DTYPE_INT_t

cdef extern from "physics_avx.hpp":
    void step_agents_avx2(
        int i,
        float* pos_x, float* pos_y, float* pos_z,
        float* vel_x, float* vel_y, float* vel_z,
        float* roll, float* pitch, float* yaw,
        float* masses, float* drag_coeffs, float* thrust_coeffs,
        float* wind_x, float* wind_y, float* wind_z, # New
        float* target_vx, float* target_vy, float* target_vz, float* target_yaw_rate,
        float* vt_x, float* vt_y, float* vt_z, # Virtual Target Position
        float* target_trajectory, # Precomputed Trajectory
        float* pos_history,
        float* observations,
        float* rewards,
        float* reward_components,
        float* done_flags,
        float* actions,
        float* action_buffer, # New: stride 11*4 = 44
        int* delays,          # New: stride 1
        int* rng_states,      # New: stride 1
        int episode_length,
        int t,
        int num_agents
    ) nogil

cdef inline float terrain_height(float x, float y) nogil:
    return 5.0 * sin(0.1 * x) * cos(0.1 * y)

cdef inline float rand_float() nogil:
    return <float>rand() / <float>RAND_MAX

# Scalar Fallback Helper function for single agent step
cdef void _step_agent_scalar(
    int i,
    float[:] pos_x, float[:] pos_y, float[:] pos_z,
    float[:] vel_x, float[:] vel_y, float[:] vel_z,
    float[:] roll, float[:] pitch, float[:] yaw,
    float[:] masses, float[:] drag_coeffs, float[:] thrust_coeffs,
    float[:] wind_x, float[:] wind_y, float[:] wind_z,
    float[:] target_vx, float[:] target_vy, float[:] target_vz, float[:] target_yaw_rate,
    float[:] vt_x, float[:] vt_y, float[:] vt_z,
    float[:, :, :] target_trajectory,
    float[:, :, :] pos_history,
    float[:, :] observations,
    float[:] rewards,
    float[:, :] reward_components,
    float[:] done_flags,
    float[:] actions,
    float[:, :, :] action_buffer, # (num_agents, 11, 4)
    int[:] delays,
    int[:] rng_states,
    int episode_length,
    int t
) noexcept nogil:
    cdef int s, k
    cdef int substeps = 2
    cdef float dt = 0.05
    cdef float g = 9.81

    cdef float px, py, pz, vx, vy, vz, r, p, y_ang
    cdef float thrust_cmd, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd
    cdef float mass, drag, thrust_coeff

    cdef float max_thrust, thrust_force
    cdef float sr, cr, sp, cp, sy, cy
    cdef float ax_thrust, ay_thrust, az_thrust, az_gravity
    cdef float ax_drag, ay_drag, az_drag
    cdef float ax, ay, az
    cdef float terr_z

    # Pre-declare loop variables
    cdef float rvx_a, rvy_a, rvz_a
    cdef float wx, wy, wz

    cdef float r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef float vx_b, vy_b, vz_b
    cdef int collision

    # Load state
    px = pos_x[i]
    py = pos_y[i]
    pz = pos_z[i]
    vx = vel_x[i]
    vy = vel_y[i]
    vz = vel_z[i]
    r = roll[i]
    p = pitch[i]
    y_ang = yaw[i]

    mass = masses[i]
    drag = drag_coeffs[i]
    thrust_coeff = thrust_coeffs[i]

    # -------------------------------------------------------------------------
    # Delay Logic
    # -------------------------------------------------------------------------
    # Shift Action Buffer: 0..9 -> 1..10
    # action_buffer is (num_agents, 11, 4)
    # Move memory for this agent (stride 4 floats)
    # &action_buffer[i, 1, 0] <-- &action_buffer[i, 0, 0], len 10*4 floats
    cdef float* ab_ptr = &action_buffer[i, 0, 0]
    memmove(ab_ptr + 4, ab_ptr, 10 * 4 * sizeof(float))

    # Insert new action
    ab_ptr[0] = actions[i * 4 + 0]
    ab_ptr[1] = actions[i * 4 + 1]
    ab_ptr[2] = actions[i * 4 + 2]
    ab_ptr[3] = actions[i * 4 + 3]

    cdef int d = delays[i]
    if d > 10: d = 10
    if d < 0: d = 0

    thrust_cmd = action_buffer[i, d, 0]
    roll_rate_cmd = action_buffer[i, d, 1]
    pitch_rate_cmd = action_buffer[i, d, 2]
    yaw_rate_cmd = action_buffer[i, d, 3]

    # -------------------------------------------------------------------------
    # Wind Dynamics (Scalar RNG)
    # -------------------------------------------------------------------------
    wx = wind_x[i]
    wy = wind_y[i]
    wz = wind_z[i]

    wx += (rand_float() - 0.5) * 0.2
    wy += (rand_float() - 0.5) * 0.2
    wz += (rand_float() - 0.5) * 0.2

    # Clamp
    if wx > 10.0: wx = 10.0
    if wx < -10.0: wx = -10.0
    if wy > 10.0: wy = 10.0
    if wy < -10.0: wy = -10.0
    if wz > 5.0: wz = 5.0
    if wz < -5.0: wz = -5.0

    wind_x[i] = wx
    wind_y[i] = wy
    wind_z[i] = wz

    # Update Virtual Target
    cdef int step_idx = t
    if step_idx > episode_length: step_idx = episode_length
    cdef float vtx_val = target_trajectory[step_idx, i, 0]
    cdef float vty_val = target_trajectory[step_idx, i, 1]
    cdef float vtz_val = target_trajectory[step_idx, i, 2]

    # Target Velocity
    cdef int next_step = step_idx + 1
    if next_step > episode_length: next_step = episode_length
    cdef float vtx_n = target_trajectory[next_step, i, 0]
    cdef float vty_n = target_trajectory[next_step, i, 1]
    cdef float vtz_n = target_trajectory[next_step, i, 2]

    cdef float inv_dt = 1.0 / dt
    cdef float vtvx_val = (vtx_n - vtx_val) * inv_dt
    cdef float vtvy_val = (vty_n - vty_val) * inv_dt
    cdef float vtvz_val = (vtz_n - vtz_val) * inv_dt

    vt_x[i] = vtx_val
    vt_y[i] = vty_val
    vt_z[i] = vtz_val

    # Shift Observations
    memmove(&observations[i, 0], &observations[i, 10], 290 * 4)

    # Substeps
    for s in range(substeps):
        # 1. Dynamics
        r += roll_rate_cmd * dt
        p += pitch_rate_cmd * dt
        y_ang += yaw_rate_cmd * dt

        max_thrust = 20.0 * thrust_coeff
        thrust_force = thrust_cmd * max_thrust

        sincosf(r, &sr, &cr)
        sincosf(p, &sp, &cp)
        sincosf(y_ang, &sy, &cy)

        ax_thrust = thrust_force * (cy * sp * cr + sy * sr) / mass
        ay_thrust = thrust_force * (sy * sp * cr - cy * sr) / mass
        az_thrust = thrust_force * (cp * cr) / mass

        az_gravity = -g

        # Relative Velocity for Drag
        rvx_a = vx - wx
        rvy_a = vy - wy
        rvz_a = vz - wz

        ax_drag = -drag * rvx_a
        ay_drag = -drag * rvy_a
        az_drag = -drag * rvz_a

        ax = ax_thrust + ax_drag
        ay = ay_thrust + ay_drag
        az = az_thrust + az_gravity + az_drag

        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        px += vx * dt
        py += vy * dt
        pz += vz * dt

        # Terrain Collision
        terr_z = terrain_height(px, py)
        if pz < terr_z:
            pz = terr_z
            vx = 0.0
            vy = 0.0
            vz = 0.0

    # Final terrain check
    terr_z = terrain_height(px, py)
    collision = 0
    if pz < terr_z:
        pz = terr_z
        collision = 1

    # Store State
    pos_x[i] = px
    pos_y[i] = py
    pos_z[i] = pz
    vel_x[i] = vx
    vel_y[i] = vy
    vel_z[i] = vz
    roll[i] = r
    pitch[i] = p
    yaw[i] = y_ang

    # Pos History
    if t <= episode_length:
        pos_history[t-1, i, 0] = px
        pos_history[t-1, i, 1] = py
        pos_history[t-1, i, 2] = pz

    # Update History with NEW state
    cdef float dx_w, dy_w, dz_w
    dx_w = vtx_val - px
    dy_w = vty_val - py
    dz_w = vtz_val - pz

    sincosf(r, &sr, &cr)
    sincosf(p, &sp, &cp)
    sincosf(y_ang, &sy, &cy)

    r11 = cy * cp
    r12 = sy * cp
    r13 = -sp
    r21 = cy * sp * sr - sy * cr
    r22 = sy * sp * sr + cy * cr
    r23 = cp * sr
    r31 = cy * sp * cr + sy * sr
    r32 = sy * sp * cr - cy * sr
    r33 = cp * cr

    cdef float xb, yb, zb
    xb = r11 * dx_w + r12 * dy_w + r13 * dz_w
    yb = r21 * dx_w + r22 * dy_w + r23 * dz_w
    zb = r31 * dx_w + r32 * dy_w + r33 * dz_w

    cdef float xc, yc, zc
    cdef float s30 = 0.5
    cdef float c30 = 0.866025
    xc = yb
    yc = -s30 * xb + c30 * zb
    zc = c30 * xb + s30 * zb

    cdef float u, v
    if zc < 0.1: zc = 0.1
    u = xc / zc
    v = yc / zc

    # -------------------------------------------------------------------------
    # Noise on Tracking
    # -------------------------------------------------------------------------
    u += (rand_float() - 0.5) * 0.1
    v += (rand_float() - 0.5) * 0.1

    if u > 1.732: u = 1.732
    if u < -1.732: u = -1.732
    if v > 1.732: v = 1.732
    if v < -1.732: v = -1.732

    observations[i, 290] = r
    observations[i, 291] = p
    observations[i, 292] = y_ang
    observations[i, 293] = pz
    observations[i, 294] = thrust_cmd
    observations[i, 295] = roll_rate_cmd
    observations[i, 296] = pitch_rate_cmd
    observations[i, 297] = yaw_rate_cmd
    observations[i, 298] = u
    observations[i, 299] = v

    cdef float size, conf
    size = 10.0 / (zc*zc + 1.0)
    size += (rand_float() - 0.5) * 0.02 # Noise on size

    cdef float w2 = roll_rate_cmd*roll_rate_cmd + pitch_rate_cmd*pitch_rate_cmd + yaw_rate_cmd*yaw_rate_cmd
    conf = exp(-0.1 * w2)
    if zc <= 0.1:
        if (c30 * xb + s30 * zb) < 0:
            conf = 0.0

    cdef float rvx, rvy, rvz
    rvx = vtvx_val - vx
    rvy = vtvy_val - vy
    rvz = vtvz_val - vz

    cdef float rvx_b, rvy_b, rvz_b
    rvx_b = r11 * rvx + r12 * rvy + r13 * rvz
    rvy_b = r21 * rvx + r22 * rvy + r23 * rvz
    rvz_b = r31 * rvx + r32 * rvy + r33 * rvz

    cdef float dist_sq = dx_w*dx_w + dy_w*dy_w + dz_w*dz_w
    cdef float dist = sqrt(dist_sq)
    cdef float dist_safe = dist
    if dist_safe < 0.1: dist_safe = 0.1

    observations[i, 300] = rvx_b
    observations[i, 301] = rvy_b
    observations[i, 302] = rvz_b
    observations[i, 303] = dist

    observations[i, 304] = u
    observations[i, 305] = v
    observations[i, 306] = size
    observations[i, 307] = conf

    # Rewards (Same logic)
    cdef float rvel_sq = rvx*rvx + rvy*rvy + rvz*rvz
    cdef float r_dot_v = dx_w*rvx + dy_w*rvy + dz_w*rvz

    cdef float dist_sq_safe = dist_sq
    if dist_sq_safe < 0.01: dist_sq_safe = 0.01

    cdef float omega_sq = (rvel_sq / dist_sq_safe) - ((r_dot_v*r_dot_v) / (dist_sq_safe*dist_sq_safe))
    if omega_sq < 0: omega_sq = 0.0

    cdef float rew_pn = -2.0 * omega_sq

    cdef float vd_dot_r = vx*dx_w + vy*dy_w + vz*dz_w
    cdef float closing = vd_dot_r / dist_safe
    cdef float rew_closing = 0.5 * closing

    vx_b = r11 * vx + r12 * vy + r13 * vz
    cdef float v_ideal = 0.1 * vx_b
    cdef float v_err = v - v_ideal
    cdef float gaze_err = u*u + v_err*v_err
    cdef float rew_gaze = -0.01 * gaze_err

    cdef float funnel = 1.0 / (dist + 1.0)
    cdef float rew_guidance = (rew_pn + rew_gaze + rew_closing) * funnel

    cdef float rew_rate = -1.0 * w2
    cdef float upright_err = 1.0 - r33
    cdef float rew_upright = -5.0 * upright_err * upright_err

    cdef float diff_thrust = 0.4 - thrust_cmd
    if diff_thrust < 0.0: diff_thrust = 0.0
    cdef float rew_eff = -10.0 * diff_thrust

    rew = rew_guidance + rew_rate + rew_upright + rew_eff

    cdef float bonus = 0.0
    if dist < 0.2:
        bonus = 10.0
    rew += bonus

    cdef float penalty = 0.0
    if r33 < 0.5:
        penalty += 10.0
    if collision == 1:
        penalty += 10.0
    rew -= penalty

    rewards[i] = rew

    reward_components[i, 0] = rew_pn
    reward_components[i, 1] = rew_closing
    reward_components[i, 2] = rew_gaze
    reward_components[i, 3] = rew_rate
    reward_components[i, 4] = rew_upright
    reward_components[i, 5] = rew_eff
    reward_components[i, 6] = -penalty
    reward_components[i, 7] = bonus

    cdef float d_flag = 0.0
    if t >= episode_length:
        d_flag = 1.0
    if dist < 0.2:
        d_flag = 1.0
    if r33 < 0.5:
        d_flag = 1.0
    if collision == 1:
        d_flag = 1.0

    done_flags[i] = d_flag

def step_cython(
    float[:] pos_x, float[:] pos_y, float[:] pos_z,
    float[:] vel_x, float[:] vel_y, float[:] vel_z,
    float[:] roll, float[:] pitch, float[:] yaw,
    float[:] masses, float[:] drag_coeffs, float[:] thrust_coeffs,
    float[:] wind_x, float[:] wind_y, float[:] wind_z, # New
    float[:] target_vx, float[:] target_vy, float[:] target_vz, float[:] target_yaw_rate,
    float[:] vt_x, float[:] vt_y, float[:] vt_z,
    float[:, :] traj_params,
    float[:, :, :] target_trajectory,
    float[:, :, :] pos_history,
    float[:, :] observations,
    float[:] rewards,
    float[:, :] reward_components,
    float[:] done_flags,
    int[:] step_counts,
    float[:] actions,
    float[:, :, :] action_buffer, # New
    int[:] delays, # New
    int[:] rng_states, # New
    int num_agents,
    int episode_length,
    int[:] env_ids,
):
    cdef int i
    step_counts[0] += 1
    cdef int t = step_counts[0]

    # Determine split for AVX (multiples of 8)
    cdef int limit_avx = (num_agents // 8) * 8

    with nogil:
        # AVX Loop (stride 8)
        for i in prange(0, limit_avx, 8):
            step_agents_avx2(
                i,
                &pos_x[0], &pos_y[0], &pos_z[0],
                &vel_x[0], &vel_y[0], &vel_z[0],
                &roll[0], &pitch[0], &yaw[0],
                &masses[0], &drag_coeffs[0], &thrust_coeffs[0],
                &wind_x[0], &wind_y[0], &wind_z[0],
                &target_vx[0], &target_vy[0], &target_vz[0], &target_yaw_rate[0],
                &vt_x[0], &vt_y[0], &vt_z[0],
                &target_trajectory[0,0,0],
                &pos_history[0,0,0],
                &observations[0,0],
                &rewards[0],
                &reward_components[0,0],
                &done_flags[0],
                &actions[0],
                &action_buffer[0,0,0],
                &delays[0],
                &rng_states[0],
                episode_length,
                t,
                num_agents
            )

        # Scalar Loop for remainder
        for i in range(limit_avx, num_agents):
            _step_agent_scalar(
                i,
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                roll, pitch, yaw,
                masses, drag_coeffs, thrust_coeffs,
                wind_x, wind_y, wind_z,
                target_vx, target_vy, target_vz, target_yaw_rate,
                vt_x, vt_y, vt_z,
                target_trajectory,
                pos_history,
                observations,
                rewards,
                reward_components,
                done_flags,
                actions,
                action_buffer,
                delays,
                rng_states,
                episode_length,
                t
            )

# Helper function for single agent reset logic
cdef void _reset_agent_scalar_wrapper(
    int i,
    float[:] pos_x, float[:] pos_y, float[:] pos_z,
    float[:] vel_x, float[:] vel_y, float[:] vel_z,
    float[:] roll, float[:] pitch, float[:] yaw,
    float[:] masses, float[:] drag_coeffs, float[:] thrust_coeffs,
    float[:] wind_x, float[:] wind_y, float[:] wind_z, # New
    float[:] target_vx, float[:] target_vy, float[:] target_vz, float[:] target_yaw_rate,
    float[:] vt_x, float[:] vt_y, float[:] vt_z,
    float[:, :] traj_params,
    float[:, :] observations,
    float[:, :, :] action_buffer,
    int[:] delays
) noexcept nogil:
    cdef float rnd_cmd
    cdef float tvx, tvy, tvz, tyr

    # Pre-declare loop variables (reset)
    cdef float ax_p, fx_p, px_p, ay_p, fy_p, py_p, az_p, fz_p, pz_p, oz_p
    cdef float sx, cx, sy_t, cy_t, sz, cz
    cdef float vtx_val, vtvx_val, vty_val, vtvy_val, vtz_val, vtvz_val
    cdef float init_angle, dist_xy_desired, sa, ca
    cdef float speed, dx, dy, dz, dist_xy, dir_x, dir_y
    cdef float rvx, rvy, rvz, r0, p0, y0
    cdef float sr, cr, sp, cp, sy, cy
    cdef float r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef float rvx_b, rvy_b, rrvz_b, dist
    cdef float xb, yb, zb, xc, yc, zc, u, v, size, conf

    # Randomize Dynamics (Wider)
    masses[i] = 0.5 + rand_float() * 2.0 # 0.5 - 2.5
    drag_coeffs[i] = 0.05 + rand_float() * 0.15 # 0.05 - 0.2
    thrust_coeffs[i] = 0.5 + rand_float() * 1.0 # 0.5 - 1.5

    # Wind (random 0-5m/s vector)
    wind_x[i] = (rand_float() - 0.5) * 10.0
    wind_y[i] = (rand_float() - 0.5) * 10.0
    wind_z[i] = (rand_float() - 0.5) * 2.0

    # Delays (0-10)
    delays[i] = <int>(rand_float() * 11.0)
    if delays[i] > 10: delays[i] = 10

    # Clear Buffer
    cdef int k
    for k in range(11):
        action_buffer[i, k, 0] = 0.0
        action_buffer[i, k, 1] = 0.0
        action_buffer[i, k, 2] = 0.0
        action_buffer[i, k, 3] = 0.0

    # Randomize Trajectory Params
    traj_params[0, i] = 3.0 + rand_float() * 4.0
    traj_params[1, i] = 0.01 + rand_float() * 0.03
    traj_params[2, i] = rand_float() * 6.28318

    traj_params[3, i] = 3.0 + rand_float() * 4.0
    traj_params[4, i] = 0.01 + rand_float() * 0.03
    traj_params[5, i] = rand_float() * 6.28318

    traj_params[6, i] = 0.0 + rand_float() * 0.1
    traj_params[7, i] = 0.01 + rand_float() * 0.05
    traj_params[8, i] = rand_float() * 6.28318
    traj_params[9, i] = 2.0

    rnd_cmd = rand_float()
    tvx=0.0; tvy=0.0; tvz=0.0; tyr=0.0

    if rnd_cmd < 0.2: pass
    elif rnd_cmd < 0.3: tvx = 1.0
    elif rnd_cmd < 0.4: tvx = -1.0
    elif rnd_cmd < 0.5: tvy = 1.0
    elif rnd_cmd < 0.6: tvy = -1.0
    elif rnd_cmd < 0.7: tvz = 1.0
    elif rnd_cmd < 0.8: tvz = -1.0
    elif rnd_cmd < 0.9: tyr = 1.0
    else: tyr = -1.0

    target_vx[i] = tvx
    target_vy[i] = tvy
    target_vz[i] = tvz
    target_yaw_rate[i] = tyr

    memset(&observations[i, 0], 0, 308 * 4)

    # Initial Target State
    ax_p = traj_params[0, i]
    fx_p = traj_params[1, i]
    px_p = traj_params[2, i]
    ay_p = traj_params[3, i]
    fy_p = traj_params[4, i]
    py_p = traj_params[5, i]
    az_p = traj_params[6, i]
    fz_p = traj_params[7, i]
    pz_p = traj_params[8, i]
    oz_p = traj_params[9, i]

    sincosf(px_p, &sx, &cx)
    vtx_val = ax_p * sx
    vtvx_val = ax_p * fx_p * cx

    sincosf(py_p, &sy_t, &cy_t)
    vty_val = ay_p * sy_t
    vtvy_val = ay_p * fy_p * cy_t

    sincosf(pz_p, &sz, &cz)
    vtz_val = oz_p + az_p * sz
    vtvz_val = az_p * fz_p * cz

    vt_x[i] = vtx_val
    vt_y[i] = vty_val
    vt_z[i] = vtz_val

    # Initial Position
    init_angle = rand_float() * 6.2831853
    dist_xy_desired = 5.0 + rand_float() * 195.0
    sincosf(init_angle, &sa, &ca)

    pos_x[i] = vtx_val + dist_xy_desired * ca
    pos_y[i] = vty_val + dist_xy_desired * sa
    pos_z[i] = vtz_val

    speed = rand_float() * 2.0
    dx = vtx_val - pos_x[i]
    dy = vty_val - pos_y[i]
    dz = vtz_val - pos_z[i]
    dist_xy = sqrt(dx*dx + dy*dy)

    dir_x = dx / (dist_xy + 1e-6)
    dir_y = dy / (dist_xy + 1e-6)

    vel_x[i] = dir_x * speed
    vel_y[i] = dir_y * speed
    vel_z[i] = 0.0

    roll[i] = 0.0

    # Yaw with offset
    yaw[i] = atan2f(dy, dx) + (rand_float() - 0.5) * 1.57 # +/- 45 deg
    # Pitch with offset
    pitch[i] = 0.1 + (rand_float() - 0.5) * 0.7 # +/- 20 deg

    # Populate Obs
    rvx = vtvx_val - vel_x[i]
    rvy = vtvy_val - vel_y[i]
    rvz = vtvz_val - vel_z[i]

    r0 = roll[i]
    p0 = pitch[i]
    y0 = yaw[i]

    sincosf(r0, &sr, &cr)
    sincosf(p0, &sp, &cp)
    sincosf(y0, &sy, &cy)

    r11 = cy * cp
    r12 = sy * cp
    r13 = -sp
    r21 = cy * sp * sr - sy * cr
    r22 = sy * sp * sr + cy * cr
    r23 = cp * sr
    r31 = cy * sp * cr + sy * sr
    r32 = sy * sp * cr - cy * sr
    r33 = cp * cr

    rvx_b = r11 * rvx + r12 * rvy + r13 * rvz
    rvy_b = r21 * rvx + r22 * rvy + r23 * rvz
    rrvz_b = r31 * rvx + r32 * rvy + r33 * rvz

    observations[i, 300] = rvx_b
    observations[i, 301] = rvy_b
    observations[i, 302] = rrvz_b

    dist = sqrt(dx*dx + dy*dy + dz*dz)
    observations[i, 303] = dist

    xb = r11 * dx + r12 * dy + r13 * dz
    yb = r21 * dx + r22 * dy + r23 * dz
    zb = r31 * dx + r32 * dy + r33 * dz

    s30 = 0.5
    c30 = 0.866025
    xc = yb
    yc = -s30 * xb + c30 * zb
    zc = c30 * xb + s30 * zb

    if zc < 0.1: zc = 0.1
    u = xc / zc
    v = yc / zc

    if u > 10.0: u = 10.0
    if u < -10.0: u = -10.0
    if v > 10.0: v = 10.0
    if v < -10.0: v = -10.0

    size = 10.0 / (zc*zc + 1.0)
    conf = 1.0
    if (c30 * xb + s30 * zb) < 0: conf = 0.0

    observations[i, 304] = u
    observations[i, 305] = v
    observations[i, 306] = size
    observations[i, 307] = conf

def reset_cython(
    float[:] pos_x, float[:] pos_y, float[:] pos_z,
    float[:] vel_x, float[:] vel_y, float[:] vel_z,
    float[:] roll, float[:] pitch, float[:] yaw,
    float[:] masses, float[:] drag_coeffs, float[:] thrust_coeffs,
    float[:] wind_x, float[:] wind_y, float[:] wind_z, # New
    float[:] target_vx, float[:] target_vy, float[:] target_vz, float[:] target_yaw_rate,
    float[:] vt_x, float[:] vt_y, float[:] vt_z,
    float[:, :] traj_params,
    float[:, :, :] target_trajectory,
    float[:, :, :] pos_history,
    float[:, :] observations,
    int[:] rng_states,
    int[:] step_counts,
    int num_agents,
    int[:] reset_indices,
    float[:, :, :] action_buffer, # New
    int[:] delays # New
):
    cdef int i
    cdef int t_idx
    cdef int steps = target_trajectory.shape[0]

    with nogil:
        for i in prange(num_agents):
            _reset_agent_scalar_wrapper(
                i,
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                roll, pitch, yaw,
                masses, drag_coeffs, thrust_coeffs,
                wind_x, wind_y, wind_z,
                target_vx, target_vy, target_vz, target_yaw_rate,
                vt_x, vt_y, vt_z,
                traj_params,
                observations,
                action_buffer,
                delays
            )

            for t_idx in range(steps):
                target_trajectory[t_idx, i, 0] = traj_params[0, i] * sin(traj_params[1, i] * <float>t_idx + traj_params[2, i])
                target_trajectory[t_idx, i, 1] = traj_params[3, i] * sin(traj_params[4, i] * <float>t_idx + traj_params[5, i])
                target_trajectory[t_idx, i, 2] = traj_params[9, i] + traj_params[6, i] * sin(traj_params[7, i] * <float>t_idx + traj_params[8, i])

    if reset_indices.shape[0] > 0:
         step_counts[0] = 0

def update_target_trajectory_from_params(
    float[:, :] traj_params,
    float[:, :, :] target_trajectory,
    int num_agents,
    int steps
):
    cdef int i, t_idx

    with nogil:
        for i in prange(num_agents):
            for t_idx in range(steps):
                target_trajectory[t_idx, i, 0] = traj_params[0, i] * sin(traj_params[1, i] * <float>t_idx + traj_params[2, i])
                target_trajectory[t_idx, i, 1] = traj_params[3, i] * sin(traj_params[4, i] * <float>t_idx + traj_params[5, i])
                target_trajectory[t_idx, i, 2] = traj_params[9, i] + traj_params[6, i] * sin(traj_params[7, i] * <float>t_idx + traj_params[8, i])
