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
    float[:] pos_history,
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
    cdef float rew
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

    # Shift Observations
    memmove(&observations[i, 0], &observations[i, 60], 1740 * 4)

    # Substeps
    for s in range(substeps):
        # 1. Dynamics
        r += roll_rate_cmd * dt
        p += pitch_rate_cmd * dt
        y_ang += yaw_rate_cmd * dt

        max_thrust = 20.0 * thrust_coeff
        thrust_force = thrust_cmd * max_thrust

        sr = sin(r); cr = cos(r)
        sp = sin(p); cp = cos(p)
        sy = sin(y_ang); cy = cos(y_ang)

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

        # 2. Capture IMU
        r11 = cy * cp
        r12 = sy * cp
        r13 = -sp
        r21 = cy * sp * sr - sy * cr
        r22 = sy * sp * sr + cy * cr
        r23 = cp * sr
        r31 = cy * sp * cr + sy * sr
        r32 = sy * sp * cr - cy * sr
        r33 = cp * cr

        acc_w_x = ax_thrust + ax_drag
        acc_w_y = ay_thrust + ay_drag
        acc_w_z = az_thrust + az_drag

        acc_b_x = r11 * acc_w_x + r12 * acc_w_y + r13 * acc_w_z
        acc_b_y = r21 * acc_w_x + r22 * acc_w_y + r23 * acc_w_z
        acc_b_z = r31 * acc_w_x + r32 * acc_w_y + r33 * acc_w_z

        # Write directly to observations end buffer
        observations[i, 1740 + s*6 + 0] = acc_b_x
        observations[i, 1740 + s*6 + 1] = acc_b_y
        observations[i, 1740 + s*6 + 2] = acc_b_z
        observations[i, 1740 + s*6 + 3] = roll_rate_cmd
        observations[i, 1740 + s*6 + 4] = pitch_rate_cmd
        observations[i, 1740 + s*6 + 5] = yaw_rate_cmd

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
        # pos_history is flat: idx * episode_length * 3 + (t-1) * 3
        # But here pos_history is memoryview.
        pos_history[i * episode_length * 3 + (t-1) * 3 + 0] = px
        pos_history[i * episode_length * 3 + (t-1) * 3 + 1] = py
        pos_history[i * episode_length * 3 + (t-1) * 3 + 2] = pz

    # Targets
    tvx = target_vx[i]
    tvy = target_vy[i]
    tvz = target_vz[i]
    tyr = target_yaw_rate[i]

    observations[i, 1800] = tvx
    observations[i, 1801] = tvy
    observations[i, 1802] = tvz
    observations[i, 1803] = tyr

    # Rewards
    sr = sin(r); cr = cos(r)
    sp = sin(p); cp = cos(p)
    sy = sin(y_ang); cy = cos(y_ang)

    r11 = cy * cp
    r12 = sy * cp
    r13 = -sp
    r21 = cy * sp * sr - sy * cr
    r22 = sy * sp * sr + cy * cr
    r23 = cp * sr
    r31 = cy * sp * cr + sy * sr
    r32 = sy * sp * cr - cy * sr
    r33 = cp * cr

    vx_b = r11 * vx + r12 * vy + r13 * vz
    vy_b = r21 * vx + r22 * vy + r23 * vz
    vz_b = r31 * vx + r32 * vy + r33 * vz

    v_err_sq = (vx_b - tvx)*(vx_b - tvx) + (vy_b - tvy)*(vy_b - tvy) + (vz_b - tvz)*(vz_b - tvz)
    yaw_rate_err_sq = (yaw_rate_cmd - tyr)*(yaw_rate_cmd - tyr)

    rew = 1.0 * exp(-2.0 * v_err_sq)
    rew += 0.5 * exp(-2.0 * yaw_rate_err_sq)
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
    float[:, :] observations
) noexcept nogil:
    cdef int k
    cdef float rnd_cmd
    cdef float tvx, tvy, tvz, tyr

    # Randomize Dynamics
    masses[i] = 0.5 + rand_float() * 1.0
    drag_coeffs[i] = 0.05 + rand_float() * 0.1
    thrust_coeffs[i] = 0.8 + rand_float() * 0.4

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

    # Reset Observations
    memset(&observations[i, 0], 0, 1800 * 4)

    observations[i, 1800] = tvx
    observations[i, 1801] = tvy
    observations[i, 1802] = tvz
    observations[i, 1803] = tyr

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
    float[:] pos_history,
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
                &pos_history[0],
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
    float[:] pos_history,
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
                observations
            )

    # Reset step counts
    if reset_indices.shape[0] > 0:
         step_counts[0] = 0
