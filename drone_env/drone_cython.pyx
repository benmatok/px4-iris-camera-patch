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
        float* target_vx, float* target_vy, float* target_vz, float* target_yaw_rate,
        float* vt_x, float* vt_y, float* vt_z, # Virtual Target Position
        float* traj_params, # Trajectory Parameters
        float* pos_history,
        float* observations,
        float* rewards,
        float* done_flags,
        float* actions,
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
    float[:] target_vx, float[:] target_vy, float[:] target_vz, float[:] target_yaw_rate,
    float[:] vt_x, float[:] vt_y, float[:] vt_z,
    float[:, :] traj_params, # Shape (10, num_agents)
    float[:, :, :] pos_history, # Shape (episode_length, num_agents, 3)
    float[:, :] observations,
    float[:] rewards,
    float[:] done_flags,
    float[:] actions,
    int episode_length,
    int t
) noexcept nogil:
    cdef int s, k
    cdef int substeps = 10
    cdef float dt = 0.01
    cdef float g = 9.81

    cdef float px, py, pz, vx, vy, vz, r, p, y_ang
    cdef float thrust_cmd, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd
    cdef float mass, drag, thrust_coeff
    cdef float tvx, tvy, tvz, tyr

    cdef float max_thrust, thrust_force
    cdef float sr, cr, sp, cp, sy, cy
    cdef float ax_thrust, ay_thrust, az_thrust, az_gravity
    cdef float ax_drag, ay_drag, az_drag
    cdef float ax, ay, az
    cdef float terr_z

    cdef float r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef float acc_w_x, acc_w_y, acc_w_z
    cdef float acc_b_x, acc_b_y, acc_b_z

    cdef float vx_b, vy_b, vz_b
    cdef float v_err_sq, yaw_rate_err_sq
    cdef float rew, rew_vis, rew_range, rew_smooth
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

    thrust_cmd = actions[i * 4 + 0]
    roll_rate_cmd = actions[i * 4 + 1]
    pitch_rate_cmd = actions[i * 4 + 2]
    yaw_rate_cmd = actions[i * 4 + 3]

    # Update Virtual Target using Trajectory Params
    cdef float t_f = <float>t

    # 0:Ax, 1:Fx, 2:Px, 3:Ay, 4:Fy, 5:Py, 6:Az, 7:Fz, 8:Pz, 9:Oz
    # New Layout: (10, num_agents) -> traj_params[param_idx, agent_idx]
    cdef float ax_p = traj_params[0, i]
    cdef float fx_p = traj_params[1, i]
    cdef float px_p = traj_params[2, i]
    cdef float ay_p = traj_params[3, i]
    cdef float fy_p = traj_params[4, i]
    cdef float py_p = traj_params[5, i]
    cdef float az_p = traj_params[6, i]
    cdef float fz_p = traj_params[7, i]
    cdef float pz_p = traj_params[8, i]
    cdef float oz_p = traj_params[9, i]

    cdef float vtx_val = ax_p * sin(fx_p * t_f + px_p)
    cdef float vty_val = ay_p * sin(fy_p * t_f + py_p)
    cdef float vtz_val = oz_p + az_p * sin(fz_p * t_f + pz_p)

    vt_x[i] = vtx_val
    vt_y[i] = vty_val
    vt_z[i] = vtz_val

    # Shift Observations
    # 608 total. 6..600 -> 0..594
    memmove(&observations[i, 0], &observations[i, 6], 594 * 4)

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

        ax_drag = -drag * vx
        ay_drag = -drag * vy
        az_drag = -drag * vz

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

        # Capture Samples at s=4 and s=9
        if s == 4 or s == 9:
            k = 0 if s == 4 else 1 # 0 or 1 index
            # Noise +/- 0.02
            # 0.04 * (rand - 0.5)
            # Offset = 594 + k*3
            observations[i, 594 + k*3 + 0] = r + (rand_float() - 0.5) * 0.04
            observations[i, 594 + k*3 + 1] = p + (rand_float() - 0.5) * 0.04
            observations[i, 594 + k*3 + 2] = y_ang + (rand_float() - 0.5) * 0.04

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

    # Pos History: (episode_length, num_agents, 3)
    if t <= episode_length:
        pos_history[t-1, i, 0] = px
        pos_history[t-1, i, 1] = py
        pos_history[t-1, i, 2] = pz

    # Tracker Features
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
    yc = s30 * xb + c30 * zb
    zc = c30 * xb - s30 * zb

    cdef float u, v, size, conf
    if zc < 0.1:
        zc = 0.1
    u = xc / zc
    v = yc / zc
    size = 10.0 / (zc*zc + 1.0)

    cdef float w2 = roll_rate_cmd*roll_rate_cmd + pitch_rate_cmd*pitch_rate_cmd + yaw_rate_cmd*yaw_rate_cmd
    conf = exp(-0.1 * w2)
    if zc <= 0.1: # Actually checked < 0.1 above
        if (c30 * xb - s30 * zb) < 0:
            conf = 0.0

    # Targets
    tvx = target_vx[i]
    tvy = target_vy[i]
    tvz = target_vz[i]
    tyr = target_yaw_rate[i]

    observations[i, 600] = tvx
    observations[i, 601] = tvy
    observations[i, 602] = tvz
    observations[i, 603] = tyr

    observations[i, 604] = u
    observations[i, 605] = v
    observations[i, 606] = size
    observations[i, 607] = conf

    # Rewards
    vx_b = r11 * vx + r12 * vy + r13 * vz
    vy_b = r21 * vx + r22 * vy + r23 * vz
    vz_b = r31 * vx + r32 * vy + r33 * vz

    v_err_sq = (vx_b - tvx)*(vx_b - tvx) + (vy_b - tvy)*(vy_b - tvy) + (vz_b - tvz)*(vz_b - tvz)
    yaw_rate_err_sq = (yaw_rate_cmd - tyr)*(yaw_rate_cmd - tyr)

    # 1. Base
    rew = 1.0 * exp(-2.0 * v_err_sq)
    rew += 0.5 * exp(-2.0 * yaw_rate_err_sq)

    # 2. Visual Servoing
    rew_vis = exp(-2.0 * (u*u + v*v))
    rew += 0.5 * rew_vis

    # 3. Range
    rew_range = exp(-2.0 * (size - 1.0)*(size - 1.0))
    rew += 0.5 * rew_range

    # 4. Smoothness
    rew_smooth = exp(-0.1 * w2)
    rew += 0.2 * rew_smooth

    # Penalties
    rew -= 0.01 * (r*r + p*p)

    if fabs(r) > 1.0 or fabs(p) > 1.0:
        rew -= 0.1

    if collision == 1:
        rew -= 10.0

    rew += 0.1
    rewards[i] = rew

    if t >= episode_length:
        done_flags[i] = 1.0
    else:
        done_flags[i] = 0.0

# Helper function for single agent reset
cdef void _reset_agent_scalar(
    int i,
    float[:] pos_x, float[:] pos_y, float[:] pos_z,
    float[:] vel_x, float[:] vel_y, float[:] vel_z,
    float[:] roll, float[:] pitch, float[:] yaw,
    float[:] masses, float[:] drag_coeffs, float[:] thrust_coeffs,
    float[:] target_vx, float[:] target_vy, float[:] target_vz, float[:] target_yaw_rate,
    float[:, :] traj_params, # Shape (10, num_agents)
    float[:, :] observations
) noexcept nogil:
    cdef int k
    cdef float rnd_cmd
    cdef float tvx, tvy, tvz, tyr

    # Randomize Dynamics
    masses[i] = 0.5 + rand_float() * 1.0
    drag_coeffs[i] = 0.05 + rand_float() * 0.1
    thrust_coeffs[i] = 0.8 + rand_float() * 0.4

    # Randomize Trajectory Params
    # 0:Ax, 1:Fx, 2:Px, 3:Ay, 4:Fy, 5:Py, 6:Az, 7:Fz, 8:Pz, 9:Oz
    traj_params[0, i] = 3.0 + rand_float() * 4.0
    traj_params[1, i] = 0.03 + rand_float() * 0.07
    traj_params[2, i] = rand_float() * 6.28318

    traj_params[3, i] = 3.0 + rand_float() * 4.0
    traj_params[4, i] = 0.03 + rand_float() * 0.07
    traj_params[5, i] = rand_float() * 6.28318

    traj_params[6, i] = 1.0 + rand_float() * 2.0
    traj_params[7, i] = 0.05 + rand_float() * 0.1
    traj_params[8, i] = rand_float() * 6.28318
    traj_params[9, i] = 8.0 + rand_float() * 4.0

    rnd_cmd = rand_float()
    tvx=0.0; tvy=0.0; tvz=0.0; tyr=0.0

    if rnd_cmd < 0.2:
        pass # Hover
    elif rnd_cmd < 0.3:
        tvx = 1.0
    elif rnd_cmd < 0.4:
        tvx = -1.0
    elif rnd_cmd < 0.5:
        tvy = 1.0
    elif rnd_cmd < 0.6:
        tvy = -1.0
    elif rnd_cmd < 0.7:
        tvz = 1.0
    elif rnd_cmd < 0.8:
        tvz = -1.0
    elif rnd_cmd < 0.9:
        tyr = 1.0
    else:
        tyr = -1.0

    target_vx[i] = tvx
    target_vy[i] = tvy
    target_vz[i] = tvz
    target_yaw_rate[i] = tyr

    # Reset Observations (Size 608)
    memset(&observations[i, 0], 0, 608 * 4)

    observations[i, 600] = tvx
    observations[i, 601] = tvy
    observations[i, 602] = tvz
    observations[i, 603] = tyr

    pos_x[i] = 0.0
    pos_y[i] = 0.0
    pos_z[i] = 10.0

    vel_x[i] = 0.0
    vel_y[i] = 0.0
    vel_z[i] = 0.0
    roll[i] = 0.0
    pitch[i] = 0.0
    yaw[i] = 0.0

def step_cython(
    float[:] pos_x, float[:] pos_y, float[:] pos_z,
    float[:] vel_x, float[:] vel_y, float[:] vel_z,
    float[:] roll, float[:] pitch, float[:] yaw,
    float[:] masses, float[:] drag_coeffs, float[:] thrust_coeffs,
    float[:] target_vx, float[:] target_vy, float[:] target_vz, float[:] target_yaw_rate,
    float[:] vt_x, float[:] vt_y, float[:] vt_z,
    float[:, :] traj_params, # Shape (10, num_agents)
    float[:, :, :] pos_history, # Shape (episode_length, num_agents, 3)
    float[:, :] observations,
    float[:] rewards,
    float[:] done_flags,
    int[:] step_counts,
    float[:] actions,
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
                &target_vx[0], &target_vy[0], &target_vz[0], &target_yaw_rate[0],
                &vt_x[0], &vt_y[0], &vt_z[0],
                &traj_params[0,0], # This is now start of contiguous block for (10, num_agents) -> Correct
                &pos_history[0,0,0], # Correct start
                &observations[0,0],
                &rewards[0],
                &done_flags[0],
                &actions[0],
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
                target_vx, target_vy, target_vz, target_yaw_rate,
                vt_x, vt_y, vt_z,
                traj_params,
                pos_history,
                observations,
                rewards,
                done_flags,
                actions,
                episode_length,
                t
            )

def reset_cython(
    float[:] pos_x, float[:] pos_y, float[:] pos_z,
    float[:] vel_x, float[:] vel_y, float[:] vel_z,
    float[:] roll, float[:] pitch, float[:] yaw,
    float[:] masses, float[:] drag_coeffs, float[:] thrust_coeffs,
    float[:] target_vx, float[:] target_vy, float[:] target_vz, float[:] target_yaw_rate,
    float[:, :] traj_params, # Shape (10, num_agents)
    float[:, :, :] pos_history,
    float[:, :] observations,
    int[:] rng_states,
    int[:] step_counts,
    int num_agents,
    int[:] reset_indices
):
    cdef int i

    # We use scalar loop for reset since it's just randoms, and AVX rand is complex.
    # But we parallelize.
    with nogil:
        for i in prange(num_agents):
            _reset_agent_scalar(
                i,
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                roll, pitch, yaw,
                masses, drag_coeffs, thrust_coeffs,
                target_vx, target_vy, target_vz, target_yaw_rate,
                traj_params,
                observations
            )

    # Reset step counts
    if reset_indices.shape[0] > 0:
         step_counts[0] = 0
