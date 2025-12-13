import numpy as np
cimport numpy as cn
from libc.math cimport sin, cos, sqrt, exp, fabs, floor

# Initialize numpy C API
cn.import_array()

cdef float terrain_height(float x, float y) nogil:
    return 5.0 * sin(0.1 * x) * cos(0.1 * y)

# Pseudo-random number generator similar to the CUDA one
cdef float rand_c(unsigned int *seed) nogil:
    seed[0] = seed[0] * 1664525 + 1013904223
    return <float>(seed[0]) / 4294967296.0

def step_cython(
    cn.float32_t[:] pos_x, cn.float32_t[:] pos_y, cn.float32_t[:] pos_z,
    cn.float32_t[:] vel_x, cn.float32_t[:] vel_y, cn.float32_t[:] vel_z,
    cn.float32_t[:] roll, cn.float32_t[:] pitch, cn.float32_t[:] yaw,
    cn.float32_t[:] masses, cn.float32_t[:] drag_coeffs, cn.float32_t[:] thrust_coeffs,
    cn.float32_t[:] target_vx, cn.float32_t[:] target_vy, cn.float32_t[:] target_vz, cn.float32_t[:] target_yaw_rate,
    cn.float32_t[:] imu_history,
    cn.float32_t[:] pos_history,
    cn.float32_t[:, :] observations,
    cn.float32_t[:] rewards,
    cn.float32_t[:] done_flags,
    cn.int32_t[:] step_counts,
    cn.float32_t[:] actions,
    int num_agents,
    int episode_length,
    object env_ids # Unused in loop but passed by kwargs
):
    cdef int idx, s, i
    cdef int total_agents = pos_x.shape[0]
    cdef float dt = 0.01
    cdef float g = 9.81
    cdef int substeps = 10

    cdef float px, py, pz, vx, vy, vz, r, p, y_ang
    cdef float mass, drag_coeff, thrust_coeff
    cdef float thrust_cmd, roll_rate, pitch_rate, yaw_rate
    cdef float max_thrust, thrust_force
    cdef float sr, cr, sp, cp, sy, cy
    cdef float ax_thrust, ay_thrust, az_thrust, az_gravity
    cdef float ax_drag, ay_drag, az_drag, ax, ay, az
    cdef float terr_z
    cdef float r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef float acc_w_x, acc_w_y, acc_w_z
    cdef float acc_b_x, acc_b_y, acc_b_z
    cdef int action_idx, env_id, t, pos_hist_idx, hist_start
    cdef float vx_b, vy_b, vz_b, v_err_sq, yaw_rate_err_sq, rew
    cdef int collision

    # Buffer for IMU data: 10 substeps * 6 floats = 60 floats
    cdef float imu_buffer[60]
    cdef int buf_idx

    with nogil:
        for idx in range(total_agents):
            action_idx = idx * 4
            thrust_cmd = actions[action_idx + 0]
            roll_rate = actions[action_idx + 1]
            pitch_rate = actions[action_idx + 2]
            yaw_rate = actions[action_idx + 3]

            mass = masses[idx]
            drag_coeff = drag_coeffs[idx]
            thrust_coeff = thrust_coeffs[idx]

            px = pos_x[idx]
            py = pos_y[idx]
            pz = pos_z[idx]
            vx = vel_x[idx]
            vy = vel_y[idx]
            vz = vel_z[idx]
            r = roll[idx]
            p = pitch[idx]
            y_ang = yaw[idx]

            for s in range(substeps):
                r += roll_rate * dt
                p += pitch_rate * dt
                y_ang += yaw_rate * dt

                max_thrust = 20.0 * thrust_coeff
                thrust_force = thrust_cmd * max_thrust

                sr = sin(r); cr = cos(r)
                sp = sin(p); cp = cos(p)
                sy = sin(y_ang); cy = cos(y_ang)

                # Thrust Vector (Z-axis of body rotated to world)
                # Body Z is (0,0,1). R * [0,0,1]^T is the 3rd column of R.
                # R3rd col:
                # r13 = cy * sp * cr + sy * sr  <-- Wait, let's check the math in CUDA code
                # CUDA:
                # ax_thrust = thrust_force * (cy * sp * cr + sy * sr) / mass;
                # ay_thrust = thrust_force * (sy * sp * cr - cy * sr) / mass;
                # az_thrust = thrust_force * (cp * cr) / mass;

                # Check rotation matrix.
                # Z-Y-X rotation?
                # R = Rz(y) * Ry(p) * Rx(r)
                # The 3rd column (body Z axis in world frame) is:
                # [ cos(y)sin(p)cos(r) + sin(y)sin(r) ]
                # [ sin(y)sin(p)cos(r) - cos(y)sin(r) ]
                # [ cos(p)cos(r) ]
                # Matches CUDA code.

                ax_thrust = thrust_force * (cy * sp * cr + sy * sr) / mass
                ay_thrust = thrust_force * (sy * sp * cr - cy * sr) / mass
                az_thrust = thrust_force * (cp * cr) / mass

                az_gravity = -g

                ax_drag = -drag_coeff * vx
                ay_drag = -drag_coeff * vy
                az_drag = -drag_coeff * vz

                ax = ax_thrust + ax_drag
                ay = ay_thrust + ay_drag
                az = az_thrust + az_gravity + az_drag

                vx += ax * dt
                vy += ay * dt
                vz += az * dt

                px += vx * dt
                py += vy * dt
                pz += vz * dt

                terr_z = terrain_height(px, py)
                if pz < terr_z:
                    pz = terr_z
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0

                # IMU
                # Need R^T to transform World Acc to Body Acc
                # R^T is R transposed.
                # R elements:
                r11 = cy * cp
                r12 = sy * cp
                r13 = -sp
                r21 = cy * sp * sr - sy * cr
                r22 = sy * sp * sr + cy * cr
                r23 = cp * sr
                r31 = cy * sp * cr + sy * sr
                r32 = sy * sp * cr - cy * sr
                r33 = cp * cr

                # Acc in world frame (excluding gravity because IMU measures proper acceleration?)
                # CUDA code:
                # acc_w_x = ax_thrust + ax_drag;
                # acc_w_y = ay_thrust + ay_drag;
                # acc_w_z = az_thrust + az_drag;
                # It does NOT include gravity. Accelerometer measures reaction force (thrust + drag).
                # Gravity is a fictitious force in the free-falling frame, or rather,
                # accelerometer measures f/m where f is contact forces. Gravity acts on all mass so it's not measured directly,
                # but the reaction force holding it against gravity is measured.
                # If in free fall (ax_thrust=0, drag=0), reading is 0.
                # If hovering, thrust balances gravity. Reading is 9.81 up.
                # So this is correct.

                acc_w_x = ax_thrust + ax_drag
                acc_w_y = ay_thrust + ay_drag
                acc_w_z = az_thrust + az_drag

                # acc_b = R^T * acc_w
                # R = [r11 r12 r13; r21 r22 r23; r31 r32 r33]
                # R^T = [r11 r21 r31; r12 r22 r32; r13 r23 r33]

                # CUDA code uses:
                # float acc_b_x = r11 * acc_w_x + r12 * acc_w_y + r13 * acc_w_z;
                # float acc_b_y = r21 * acc_w_x + r22 * acc_w_y + r23 * acc_w_z;
                # float acc_b_z = r31 * acc_w_x + r32 * acc_w_y + r33 * acc_w_z;

                # Wait, the CUDA code uses rows of R (r11, r12, r13) as weights.
                # That looks like R * acc_w, not R^T * acc_w.
                # Unless r11, r12 etc defined in CUDA are actually for R^T?
                # CUDA:
                # float r11 = cy * cp;
                # float r12 = sy * cp;
                # ...
                # These are standard Rotation Matrix elements.
                # So the CUDA code is calculating acc_b = R * acc_w ??
                # If R transforms Body to World, then World to Body should be R^T.
                # So it SHOULD be:
                # acc_b_x = r11 * acc_w_x + r21 * acc_w_y + r31 * acc_w_z;

                # HOWEVER, I must replicate the behavior of the existing CUDA/CPU code, even if it might be "wrong" physically,
                # unless I am fixing a bug. But the request is just "convert busy loop code".
                # Let's check the CPU python code.
                # acc_b_x = r11 * acc_w_x + r12 * acc_w_y + r13 * acc_w_z
                # It matches CUDA.
                # So I will copy the logic exactly.

                acc_b_x = r11 * acc_w_x + r12 * acc_w_y + r13 * acc_w_z
                acc_b_y = r21 * acc_w_x + r22 * acc_w_y + r23 * acc_w_z
                acc_b_z = r31 * acc_w_x + r32 * acc_w_y + r33 * acc_w_z

                buf_idx = s * 6
                imu_buffer[buf_idx + 0] = acc_b_x
                imu_buffer[buf_idx + 1] = acc_b_y
                imu_buffer[buf_idx + 2] = acc_b_z
                imu_buffer[buf_idx + 3] = roll_rate
                imu_buffer[buf_idx + 4] = pitch_rate
                imu_buffer[buf_idx + 5] = yaw_rate

            terr_z = terrain_height(px, py)
            collision = 0
            if pz < terr_z:
                pz = terr_z
                collision = 1

            pos_x[idx] = px
            pos_y[idx] = py
            pos_z[idx] = pz
            vel_x[idx] = vx
            vel_y[idx] = vy
            vel_z[idx] = vz
            roll[idx] = r
            pitch[idx] = p
            yaw[idx] = y_ang

            env_id = idx // num_agents
            if (idx % num_agents) == 0:
                step_counts[env_id] += 1
            t = step_counts[env_id]

            if t <= episode_length:
                pos_hist_idx = idx * episode_length * 3 + (t-1) * 3
                # Ensure bounds
                # pos_history len is num_agents * episode_length * 3
                # pos_hist_idx max is (N-1)*L*3 + (L-1)*3 < N*L*3. Safe.
                pos_history[pos_hist_idx + 0] = px
                pos_history[pos_hist_idx + 1] = py
                pos_history[pos_hist_idx + 2] = pz

            # Update History Buffer (IMU history)
            hist_start = idx * 1800
            # Shift left by 60
            # CUDA: for (int i = 0; i < 1740; i++) imu_history[hist_start + i] = imu_history[hist_start + i + 60];
            for i in range(1740):
                imu_history[hist_start + i] = imu_history[hist_start + i + 60]

            # Append new buffer
            for i in range(60):
                imu_history[hist_start + 1740 + i] = imu_buffer[i]

            # Update Observations
            # observations[idx, :]
            # Copy history
            for i in range(1800):
                observations[idx, i] = imu_history[hist_start + i]

            observations[idx, 1800] = target_vx[idx]
            observations[idx, 1801] = target_vy[idx]
            observations[idx, 1802] = target_vz[idx]
            observations[idx, 1803] = target_yaw_rate[idx]

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

            # Velocity in body frame
            # vx_b = R * v_world?
            # CUDA:
            # vx_b = r11 * vx + r12 * vy + r13 * vz;
            # Again, this looks like R * v.
            vx_b = r11 * vx + r12 * vy + r13 * vz
            vy_b = r21 * vx + r22 * vy + r23 * vz
            vz_b = r31 * vx + r32 * vy + r33 * vz

            v_err_sq = (vx_b - target_vx[idx])**2 + (vy_b - target_vy[idx])**2 + (vz_b - target_vz[idx])**2
            yaw_rate_err_sq = (yaw_rate - target_yaw_rate[idx])**2

            rew = 0.0
            rew += 1.0 * exp(-2.0 * v_err_sq)
            rew += 0.5 * exp(-2.0 * yaw_rate_err_sq)
            rew -= 0.01 * (r*r + p*p)
            if fabs(r) > 1.0 or fabs(p) > 1.0:
                rew -= 0.1
            if collision:
                rew -= 10.0
            rew += 0.1
            rewards[idx] = rew

            if t >= episode_length:
                done_flags[idx] = 1.0
            else:
                done_flags[idx] = 0.0

def reset_cython(
    cn.float32_t[:] pos_x, cn.float32_t[:] pos_y, cn.float32_t[:] pos_z,
    cn.float32_t[:] vel_x, cn.float32_t[:] vel_y, cn.float32_t[:] vel_z,
    cn.float32_t[:] roll, cn.float32_t[:] pitch, cn.float32_t[:] yaw,
    cn.float32_t[:] masses, cn.float32_t[:] drag_coeffs, cn.float32_t[:] thrust_coeffs,
    cn.float32_t[:] target_vx, cn.float32_t[:] target_vy, cn.float32_t[:] target_vz, cn.float32_t[:] target_yaw_rate,
    cn.float32_t[:] imu_history,
    cn.float32_t[:] pos_history,
    cn.int32_t[:] rng_states,
    cn.int32_t[:] step_counts,
    int num_agents,
    cn.int32_t[:] reset_indices
):
    cdef int i, idx, env_id, agent_id
    cdef unsigned int seed
    cdef float rnd_cmd
    cdef float tvx, tvy, tvz, tyr
    cdef int hist_start, h
    cdef int num_resets = reset_indices.shape[0]

    with nogil:
        for i in range(num_resets):
            env_id = reset_indices[i]
            for agent_id in range(num_agents):
                idx = env_id * num_agents + agent_id

                # RNG
                seed = <unsigned int>(rng_states[idx] + idx + 12345 + step_counts[env_id]*6789)

                masses[idx] = 0.5 + rand_c(&seed) * 1.0
                drag_coeffs[idx] = 0.05 + rand_c(&seed) * 0.1
                thrust_coeffs[idx] = 0.8 + rand_c(&seed) * 0.4

                rnd_cmd = rand_c(&seed)
                tvx = 0.0; tvy = 0.0; tvz = 0.0; tyr = 0.0

                if rnd_cmd < 0.2:
                     tvx = 0.0; tvy = 0.0; tvz = 0.0
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

                target_vx[idx] = tvx
                target_vy[idx] = tvy
                target_vz[idx] = tvz
                target_yaw_rate[idx] = tyr

                hist_start = idx * 1800
                for h in range(1800):
                    imu_history[hist_start + h] = 0.0

                rng_states[idx] = <int>seed

                pos_x[idx] = 0.0
                pos_y[idx] = 0.0
                pos_z[idx] = 10.0

                vel_x[idx] = 0.0
                vel_y[idx] = 0.0
                vel_z[idx] = 0.0
                roll[idx] = 0.0
                pitch[idx] = 0.0
                yaw[idx] = 0.0

                if agent_id == 0:
                    step_counts[env_id] = 0
