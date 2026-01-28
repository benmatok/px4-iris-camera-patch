#ifndef GHOST_MODEL_HPP
#define GHOST_MODEL_HPP

#include <cmath>
#include <vector>
#include <iostream>
#include <cstring>

struct GhostModel {
    float mass;
    float drag_coeff;
    float thrust_coeff;
    float wind_x;
    float wind_y;
};

struct GhostState {
    float px, py, pz;
    float vx, vy, vz;
    float roll, pitch, yaw;
};

struct GhostAction {
    float thrust;
    float roll_rate;
    float pitch_rate;
    float yaw_rate;
};

// Jacobian Matrix (9 rows x 4 cols)
// Rows: px, py, pz, vx, vy, vz, r, p, y
// Cols: thrust, roll_rate, pitch_rate, yaw_rate
struct Jacobian {
    float data[36]; // Row-major (9x4) w.r.t Action
    float grad_mass[9]; // (9x1) w.r.t Mass

    void set_zero() {
        for(int i=0; i<36; i++) data[i] = 0.0f;
        for(int i=0; i<9; i++) grad_mass[i] = 0.0f;
    }

    float& operator()(int row, int col) {
        return data[row * 4 + col];
    }
};

class GhostPhysics {
public:
    static constexpr float G = 9.81f;
    static constexpr float MAX_THRUST_BASE = 20.0f;

    // Forward Step
    static GhostState step(const GhostModel& model, const GhostState& state, const GhostAction& action, float dt) {
        GhostState next_state;

        // 1. Update Attitude (Euler Integration)
        next_state.roll = state.roll + action.roll_rate * dt;
        next_state.pitch = state.pitch + action.pitch_rate * dt;
        next_state.yaw = state.yaw + action.yaw_rate * dt;

        // 2. Compute Forces based on New Attitude
        float max_thrust = MAX_THRUST_BASE * model.thrust_coeff;
        float thrust_force = action.thrust * max_thrust;

        // Clamp Thrust [0, 1] for safety, though gradients assume linear in active region
        if (thrust_force < 0) thrust_force = 0;
        // if (thrust_force > max_thrust) thrust_force = max_thrust; // Optional clamping

        float cr = cosf(next_state.roll);
        float sr = sinf(next_state.roll);
        float cp = cosf(next_state.pitch);
        float sp = sinf(next_state.pitch);
        float cy = cosf(next_state.yaw);
        float sy = sinf(next_state.yaw);

        // Rotation Matrix Elements (World Acceleration components)
        // R31
        float ax_dir = cy * sp * cr + sy * sr;
        // R32
        float ay_dir = sy * sp * cr - cy * sr;
        // R33
        float az_dir = cp * cr;

        // Accelerations
        float ax_thrust = thrust_force * ax_dir / model.mass;
        float ay_thrust = thrust_force * ay_dir / model.mass;
        float az_thrust = thrust_force * az_dir / model.mass;

        float ax_drag = -model.drag_coeff * (state.vx - model.wind_x);
        float ay_drag = -model.drag_coeff * (state.vy - model.wind_y);
        float az_drag = -model.drag_coeff * state.vz; // No vertical wind modeled yet

        float ax = ax_thrust + ax_drag;
        float ay = ay_thrust + ay_drag;
        float az = az_thrust + az_drag - G;

        // 3. Update Velocity
        next_state.vx = state.vx + ax * dt;
        next_state.vy = state.vy + ay * dt;
        next_state.vz = state.vz + az * dt;

        // 4. Update Position
        next_state.px = state.px + next_state.vx * dt;
        next_state.py = state.py + next_state.vy * dt;
        next_state.pz = state.pz + next_state.vz * dt;

        return next_state;
    }

    // Analytical Gradients
    static Jacobian get_gradients(const GhostModel& model, const GhostState& state, const GhostAction& action, float dt) {
        Jacobian J;
        J.set_zero();

        // Intermediate values for Next Attitude
        float r = state.roll + action.roll_rate * dt;
        float p = state.pitch + action.pitch_rate * dt;
        float y = state.yaw + action.yaw_rate * dt;

        float cr = cosf(r); float sr = sinf(r);
        float cp = cosf(p); float sp = sinf(p);
        float cy = cosf(y); float sy = sinf(y);

        float max_thrust = MAX_THRUST_BASE * model.thrust_coeff;
        float F = action.thrust * max_thrust; // Assume raw action is valid (or handle derivative of clamp)

        // Force Directions (R31, R32, R33)
        float D_x = cy * sp * cr + sy * sr;
        float D_y = sy * sp * cr - cy * sr;
        float D_z = cp * cr;

        // ----------------------------------------------------------------
        // 1. Derivatives w.r.t THRUST (Column 0)
        // ----------------------------------------------------------------
        // d(Att)/d(Thrust) = 0
        J(6, 0) = 0; J(7, 0) = 0; J(8, 0) = 0;

        // d(a)/d(Thrust) = (F_max/m) * Direction
        float da_dT_x = (max_thrust / model.mass) * D_x;
        float da_dT_y = (max_thrust / model.mass) * D_y;
        float da_dT_z = (max_thrust / model.mass) * D_z;

        // d(v)/d(Thrust) = da/dT * dt
        J(3, 0) = da_dT_x * dt;
        J(4, 0) = da_dT_y * dt;
        J(5, 0) = da_dT_z * dt;

        // d(p)/d(Thrust) = dv/dT * dt
        J(0, 0) = J(3, 0) * dt;
        J(1, 0) = J(4, 0) * dt;
        J(2, 0) = J(5, 0) * dt;

        // ----------------------------------------------------------------
        // 2. Derivatives w.r.t RATES (Columns 1, 2, 3)
        // ----------------------------------------------------------------
        // d(Att_next)/d(Rate) = dt
        J(6, 1) = dt; // d(roll)/d(roll_rate)
        J(7, 2) = dt; // d(pitch)/d(pitch_rate)
        J(8, 3) = dt; // d(yaw)/d(yaw_rate)

        // Accelerations depend on Attitude (r, p, y)
        // a = (F/m) * D(r, p, y)
        // da/dRate = (F/m) * (dD/dr * dr/dRate + dD/dp * dp/dRate + dD/dy * dy/dRate)
        // dr/dRate = dt (diagonal), others 0.
        // So da/d(roll_rate) = (F/m) * dD/dr * dt

        float F_m = F / model.mass;

        // Partial derivatives of Directions D_x, D_y, D_z w.r.t r, p, y

        // --- D_x = cy*sp*cr + sy*sr ---
        float dDx_dr = cy*sp*(-sr) + sy*cr;
        float dDx_dp = cy*cp*cr;
        float dDx_dy = -sy*sp*cr + cy*sr;

        // --- D_y = sy*sp*cr - cy*sr ---
        float dDy_dr = sy*sp*(-sr) - cy*cr;
        float dDy_dp = sy*cp*cr;
        float dDy_dy = cy*sp*cr - (-sy)*sr; // cy*sp*cr + sy*sr

        // --- D_z = cp*cr ---
        float dDz_dr = cp*(-sr);
        float dDz_dp = -sp*cr;
        float dDz_dy = 0.0f;

        // Column 1: Roll Rate
        float da_dRr_x = F_m * dDx_dr;
        float da_dRr_y = F_m * dDy_dr;
        float da_dRr_z = F_m * dDz_dr;

        J(3, 1) = da_dRr_x * dt * dt; // dv/dRr = da/dAtt * dAtt/dRate * dt = (da/dAtt * dt) * dt
        J(4, 1) = da_dRr_y * dt * dt;
        J(5, 1) = da_dRr_z * dt * dt;

        J(0, 1) = J(3, 1) * dt;
        J(1, 1) = J(4, 1) * dt;
        J(2, 1) = J(5, 1) * dt;

        // Column 2: Pitch Rate
        float da_dPr_x = F_m * dDx_dp;
        float da_dPr_y = F_m * dDy_dp;
        float da_dPr_z = F_m * dDz_dp;

        J(3, 2) = da_dPr_x * dt * dt;
        J(4, 2) = da_dPr_y * dt * dt;
        J(5, 2) = da_dPr_z * dt * dt;

        J(0, 2) = J(3, 2) * dt;
        J(1, 2) = J(4, 2) * dt;
        J(2, 2) = J(5, 2) * dt;

        // Column 3: Yaw Rate
        float da_dYr_x = F_m * dDx_dy;
        float da_dYr_y = F_m * dDy_dy;
        float da_dYr_z = F_m * dDz_dy;

        J(3, 3) = da_dYr_x * dt * dt;
        J(4, 3) = da_dYr_y * dt * dt;
        J(5, 3) = da_dYr_z * dt * dt;

        J(0, 3) = J(3, 3) * dt;
        J(1, 3) = J(4, 3) * dt;
        J(2, 3) = J(5, 3) * dt;

        // ----------------------------------------------------------------
        // 3. Derivatives w.r.t MASS
        // ----------------------------------------------------------------
        // da_thrust / dm = -a_thrust / m
        // a_thrust components:
        // ax = F/m * Dx
        // da_x/dm = F * Dx * (-1/m^2) = - (F/m * Dx) / m = -ax_thrust / m

        // Actually: F_m = F / mass.
        // ax_thrust = F_m * D_x.
        float ax_th = F_m * D_x;
        float ay_th = F_m * D_y;
        float az_th = F_m * D_z;

        float da_dm_x = -ax_th / model.mass;
        float da_dm_y = -ay_th / model.mass;
        float da_dm_z = -az_th / model.mass;

        J.grad_mass[3] = da_dm_x * dt;
        J.grad_mass[4] = da_dm_y * dt;
        J.grad_mass[5] = da_dm_z * dt;

        J.grad_mass[0] = J.grad_mass[3] * dt;
        J.grad_mass[1] = J.grad_mass[4] * dt;
        J.grad_mass[2] = J.grad_mass[5] * dt;

        return J;
    }

    // State Jacobian (9x9)
    static void get_state_jacobian(const GhostModel& model, const GhostState& state, const GhostAction& action, float dt, float* J_state) {
        // Initialize to Identity for P, V, Att diagonals?
        // Structure:
        // P' = P + V' dt
        // V' = V + a dt
        // Att' = Att + Rates dt

        // Rows: P(0-2), V(3-5), Att(6-8)
        // Cols: P(0-2), V(3-5), Att(6-8)

        // Reset
        for(int i=0; i<81; i++) J_state[i] = 0.0f;
        auto set = [&](int r, int c, float v) { J_state[r*9 + c] = v; };

        // dAtt'/dAtt = I
        set(6, 6, 1.0f); set(7, 7, 1.0f); set(8, 8, 1.0f);

        // dV'/dV = I * (1 - Cd * dt)
        float dv_dv = 1.0f - model.drag_coeff * dt;
        set(3, 3, dv_dv); set(4, 4, dv_dv); set(5, 5, dv_dv);

        // dP'/dP = I
        set(0, 0, 1.0f); set(1, 1, 1.0f); set(2, 2, 1.0f);

        // dP'/dV = dV'/dV * dt
        set(0, 3, dv_dv * dt); set(1, 4, dv_dv * dt); set(2, 5, dv_dv * dt);

        // dV'/dAtt
        // We need da/dAtt.
        // Recalculate forces/directions.
        // Note: step uses next_att. But for Jacobian at step t, we linearized around current state?
        // step uses next_state.roll = state.roll + rate*dt.
        // So a depends on (state.roll + rate*dt).
        // da / d(state.roll) is same as da / d(next_state.roll).

        float r = state.roll + action.roll_rate * dt;
        float p = state.pitch + action.pitch_rate * dt;
        float y = state.yaw + action.yaw_rate * dt;
        float cr = cosf(r); float sr = sinf(r);
        float cp = cosf(p); float sp = sinf(p);
        float cy = cosf(y); float sy = sinf(y);

        float max_thrust = MAX_THRUST_BASE * model.thrust_coeff;
        float F = action.thrust * max_thrust;
        float F_m = F / model.mass;

        float dDx_dr = cy*sp*(-sr) + sy*cr;
        float dDx_dp = cy*cp*cr;
        float dDx_dy = -sy*sp*cr + cy*sr;

        float dDy_dr = sy*sp*(-sr) - cy*cr;
        float dDy_dp = sy*cp*cr;
        float dDy_dy = cy*sp*cr + sy*sr; // Fixed sign error in previous derivation?
        // Dy = sy*sp*cr - cy*sr. dDy/dy = cy*sp*cr - (-sy)*sr = cy*sp*cr + sy*sr. Correct.

        float dDz_dr = cp*(-sr);
        float dDz_dp = -sp*cr;
        float dDz_dy = 0.0f;

        // dV/dRoll = da/dRoll * dt = F_m * dD/dr * dt
        float dv_dr_x = F_m * dDx_dr * dt;
        float dv_dr_y = F_m * dDy_dr * dt;
        float dv_dr_z = F_m * dDz_dr * dt;

        set(3, 6, dv_dr_x); set(4, 6, dv_dr_y); set(5, 6, dv_dr_z);

        float dv_dp_x = F_m * dDx_dp * dt;
        float dv_dp_y = F_m * dDy_dp * dt;
        float dv_dp_z = F_m * dDz_dp * dt;

        set(3, 7, dv_dp_x); set(4, 7, dv_dp_y); set(5, 7, dv_dp_z);

        float dv_dy_x = F_m * dDx_dy * dt;
        float dv_dy_y = F_m * dDy_dy * dt;
        float dv_dy_z = F_m * dDz_dy * dt;

        set(3, 8, dv_dy_x); set(4, 8, dv_dy_y); set(5, 8, dv_dy_z);

        // dP'/dAtt = dV'/dAtt * dt
        set(0, 6, dv_dr_x * dt); set(1, 6, dv_dr_y * dt); set(2, 6, dv_dr_z * dt);
        set(0, 7, dv_dp_x * dt); set(1, 7, dv_dp_y * dt); set(2, 7, dv_dp_z * dt);
        set(0, 8, dv_dy_x * dt); set(1, 8, dv_dy_y * dt); set(2, 8, dv_dy_z * dt);
    }
};

#endif
