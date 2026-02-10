import numpy as np
import math
import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ghost_dpc.ghost_dpc import PyGhostModel

def step_euler_classic(state_dict, action_dict, dt, model):
    """
    Simulates one step using the old Explicit Euler integration (Classic).
    """
    # Unpack State
    px, py, pz = state_dict['px'], state_dict['py'], state_dict['pz']
    vx, vy, vz = state_dict['vx'], state_dict['vy'], state_dict['vz']
    roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
    wx = state_dict.get('wx', 0.0)
    wy = state_dict.get('wy', 0.0)
    wz = state_dict.get('wz', 0.0)

    # Unpack Action
    thrust = action_dict['thrust']
    roll_rate_cmd = action_dict['roll_rate']
    pitch_rate_cmd = action_dict['pitch_rate']
    yaw_rate_cmd = action_dict['yaw_rate']

    # 1. Update Angular Velocities (Explicit Lag Dynamics)
    # next_w = w + (cmd - w) * (dt/tau)
    next_wx = wx + (roll_rate_cmd - wx) * (dt / model.tau)
    next_wy = wy + (pitch_rate_cmd - wy) * (dt / model.tau)
    next_wz = wz + (yaw_rate_cmd - wz) * (dt / model.tau)

    # 2. Update Attitude (Euler Integration)
    sr = math.sin(roll); cr = math.cos(roll)
    sp = math.sin(pitch); cp = math.cos(pitch)

    # Avoid singularity
    if abs(cp) < 1e-6: cp = 1e-6
    tt = sp / cp
    st = 1.0 / cp

    r_dot = next_wx + next_wy * sr * tt + next_wz * cr * tt
    p_dot = next_wy * cr - next_wz * sr
    y_dot = (next_wy * sr + next_wz * cr) * st

    next_roll = roll + r_dot * dt
    next_pitch = pitch + p_dot * dt
    next_yaw = yaw + y_dot * dt

    # 3. Compute Forces based on New Attitude (Euler angles)
    max_thrust = model.MAX_THRUST_BASE * model.thrust_coeff
    thrust_force = thrust * max_thrust
    if thrust_force < 0: thrust_force = 0.0

    cr_n = math.cos(next_roll); sr_n = math.sin(next_roll)
    cp_n = math.cos(next_pitch); sp_n = math.sin(next_pitch)
    cy_n = math.cos(next_yaw); sy_n = math.sin(next_yaw)

    # R31
    ax_dir = cy_n * sp_n * cr_n + sy_n * sr_n
    # R32
    ay_dir = sy_n * sp_n * cr_n - cy_n * sr_n
    # R33
    az_dir = cp_n * cr_n

    # Accelerations
    ax_thrust = thrust_force * ax_dir / model.mass
    ay_thrust = thrust_force * ay_dir / model.mass
    az_thrust = thrust_force * az_dir / model.mass

    ax_drag = -model.drag_coeff * (vx - model.wind_x)
    ay_drag = -model.drag_coeff * (vy - model.wind_y)
    az_drag = -model.drag_coeff * vz

    ax = ax_thrust + ax_drag
    ay = ay_thrust + ay_drag
    az = az_thrust + az_drag - model.G

    # 3. Update Velocity (Explicit Euler)
    next_vx = vx + ax * dt
    next_vy = vy + ay * dt
    next_vz = vz + az * dt

    # 4. Update Position (Explicit Euler)
    next_px = px + next_vx * dt
    next_py = py + next_vy * dt
    next_pz = pz + next_vz * dt

    # Normalize
    next_roll = (next_roll + math.pi) % (2 * math.pi) - math.pi
    next_pitch = (next_pitch + math.pi) % (2 * math.pi) - math.pi
    next_yaw = (next_yaw + math.pi) % (2 * math.pi) - math.pi

    return {
        'px': next_px, 'py': next_py, 'pz': next_pz,
        'vx': next_vx, 'vy': next_vy, 'vz': next_vz,
        'roll': next_roll, 'pitch': next_pitch, 'yaw': next_yaw,
        'wx': next_wx, 'wy': next_wy, 'wz': next_wz
    }

def run_simulation(integration_func, dt, duration, model, action_func):
    """
    Runs a simulation using the provided step function.
    """
    steps = int(duration / dt)
    state = {
        'px': 0.0, 'py': 0.0, 'pz': 10.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'wx': 0.0, 'wy': 0.0, 'wz': 0.0
    }

    history = {'px': [], 'py': [], 'pz': [], 'roll': [], 'pitch': [], 'yaw': []}

    for i in range(steps):
        t = i * dt
        action = action_func(t)

        # Determine step function signature
        # PyGhostModel uses self implicit, step_euler_classic needs model passed
        if integration_func == step_euler_classic:
            state = step_euler_classic(state, action, dt, model)
        else:
            # Assuming it's model.step
            state = integration_func(state, action, dt)

        history['px'].append(state['px'])
        history['py'].append(state['py'])
        history['pz'].append(state['pz'])
        history['roll'].append(state['roll'])
        history['pitch'].append(state['pitch'])
        history['yaw'].append(state['yaw'])

    return state, history

def main():
    # Parameters
    model_se3 = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, tau=0.1)

    # Action Scenario:
    # 0-1s: Hover + Roll Rate (Banking)
    # 1-2s: Pitch Rate + Thrust Boost (Climbing turn)
    def action_profile(t):
        thrust = 0.55 # Slight lift
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0

        if t < 1.0:
            roll_rate = 0.5 # 0.5 rad/s roll
        else:
            pitch_rate = 0.3 # Pitch up
            thrust = 0.7     # Accelerate
            yaw_rate = 0.2   # Yaw

        return {
            'thrust': thrust,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'yaw_rate': yaw_rate
        }

    duration = 2.0

    # 1. Run Classic Euler (Tiny Step - Ground Truth)
    dt_tiny = 0.001
    print(f"Running Classic Euler (dt={dt_tiny}s) as Ground Truth...")
    final_gt, hist_gt = run_simulation(step_euler_classic, dt_tiny, duration, model_se3, action_profile)

    # 2. Run Classic Euler (Coarse Step)
    dt_coarse = 0.05
    print(f"Running Classic Euler (dt={dt_coarse}s)...")
    final_classic, hist_classic = run_simulation(step_euler_classic, dt_coarse, duration, model_se3, action_profile)

    # 3. Run New SE3 (Coarse Step)
    print(f"Running New SE3 (dt={dt_coarse}s)...")
    # model_se3.step is bound method
    final_se3, hist_se3 = run_simulation(model_se3.step, dt_coarse, duration, model_se3, action_profile)

    # --- Comparison ---
    print("\n--- Final State Comparison ---")

    def calc_error(s1, s2):
        dp = np.sqrt((s1['px']-s2['px'])**2 + (s1['py']-s2['py'])**2 + (s1['pz']-s2['pz'])**2)
        dv = np.sqrt((s1['vx']-s2['vx'])**2 + (s1['vy']-s2['vy'])**2 + (s1['vz']-s2['vz'])**2)
        # Orientation error (simple Euler dist for now, approximations apply)
        dr = abs(s1['roll'] - s2['roll'])
        dpitch = abs(s1['pitch'] - s2['pitch'])
        dy = abs(s1['yaw'] - s2['yaw'])
        return dp, dv, dr, dpitch, dy

    # Classic(0.05) vs GT
    dp_c, dv_c, dr_c, dpi_c, dy_c = calc_error(final_classic, final_gt)
    print(f"Classic Euler (dt={dt_coarse}) Error vs GT:")
    print(f"  Pos: {dp_c:.4f} m")
    print(f"  Vel: {dv_c:.4f} m/s")
    print(f"  Att: R={dr_c:.4f}, P={dpi_c:.4f}, Y={dy_c:.4f} rad")

    # SE3(0.05) vs GT
    dp_s, dv_s, dr_s, dpi_s, dy_s = calc_error(final_se3, final_gt)
    print(f"SE3 Manifold (dt={dt_coarse}) Error vs GT:")
    print(f"  Pos: {dp_s:.4f} m")
    print(f"  Vel: {dv_s:.4f} m/s")
    print(f"  Att: R={dr_s:.4f}, P={dpi_s:.4f}, Y={dy_s:.4f} rad")

    # Success Criteria?
    # SE3 should ideally be better or comparable. If it's vastly worse, something is wrong.
    # Note: Implicit Euler for rates in SE3 introduces damping compared to Explicit Euler in Classic.
    # This might cause divergence from "Classic GT".
    # However, Implicit is more stable.
    # Let's check if the difference is reasonable.

    if dp_s < 1.0:
        print("\nSUCCESS: SE3 Integration produced a valid trajectory comparable to Ground Truth.")
    else:
        print("\nWARNING: Large divergence in SE3 integration.")

if __name__ == "__main__":
    main()
