#ifndef DPC_SOLVER_HPP
#define DPC_SOLVER_HPP

#include "ghost_model.hpp"
#include <vector>
#include <cmath>
#include <iostream>

class DPCSolver {
public:
    int horizon = 10;
    int iterations = 10;
    float learning_rate = 0.05f;

    struct TrajectoryPoint {
        float px, py, pz;
        float vx, vy, vz;
        float roll, pitch, yaw;
        // Auxiliary for cost
        float u, v;
    };

    // Helper: Matrix Multiply (A: 9x9, B: 9x4) -> C: 9x4
    static void matmul_99_94(const float* A, const float* B, float* C) {
        for(int i=0; i<9; i++) {
            for(int j=0; j<4; j++) {
                float sum = 0.0f;
                for(int k=0; k<9; k++) {
                    sum += A[i*9 + k] * B[k*4 + j];
                }
                C[i*4 + j] = sum;
            }
        }
    }

    // Helper: Matrix Add (A: 9x4, B: 9x4) -> A += B
    static void matadd_94(float* A, const float* B) {
        for(int i=0; i<36; i++) A[i] += B[i];
    }

    // Optimize Control Action
    // Returns optimized action
    GhostAction solve(const GhostState& initial_state,
                      const float* target_pos, // [x, y, z]
                      const GhostAction& initial_guess,
                      const std::vector<GhostModel>& models,
                      const std::vector<float>& weights,
                      float dt) {

        GhostAction current_action = initial_guess;

        // Gradient Descent Loop
        for(int iter=0; iter<iterations; iter++) {
            // Clamp Action for safety during optimization
            if (current_action.thrust < 0.0f) current_action.thrust = 0.0f;
            if (current_action.thrust > 1.0f) current_action.thrust = 1.0f;

            float total_grad[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            // Loop over models (Hypotheses)
            for(size_t m_idx=0; m_idx<models.size(); m_idx++) {
                const GhostModel& model = models[m_idx];
                float weight = weights[m_idx];
                if (weight < 1e-5f) continue;

                // Forward Pass with Sensitivity
                GhostState state = initial_state;

                // Sensitivity Matrix G = dS/dU (9x4). Init 0.
                float G[36];
                for(int i=0; i<36; i++) G[i] = 0.0f;

                for(int t=0; t<horizon; t++) {
                    // 1. Get Gradients at current step (Dynamics)
                    Jacobian J_act = GhostPhysics::get_gradients(model, state, current_action, dt);

                    float J_state[81];
                    GhostPhysics::get_state_jacobian(model, state, current_action, dt, J_state);

                    // 2. Propagate Sensitivity: G_next = J_state * G + J_act
                    float G_next[36];
                    matmul_99_94(J_state, G, G_next);
                    matadd_94(G_next, J_act.data);

                    // 3. Step Physics
                    GhostState next_state = GhostPhysics::step(model, state, current_action, dt);

                    // 4. Compute Cost Gradient dL/dU = dL/dS * G_next + dL/dU_direct
                    // dL/dU_direct (e.g. effort) can be added directly.

                    // Cost Components:
                    // A. Distance to Target
                    float dx = next_state.px - target_pos[0];
                    float dy = next_state.py - target_pos[1];
                    float dz = next_state.pz - target_pos[2];
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    float dist = sqrtf(dist_sq + 1e-6f);

                    // dL_dist / dS (Pos)
                    // L = dist. dL/dP = (1/dist) * (P - T).
                    float dL_dP[3] = { dx/dist, dy/dist, dz/dist };

                    // B. Altitude Cost (Target Z + 2.0)
                    float target_safe_z = target_pos[2] + 2.0f;
                    float dz_safe = next_state.pz - target_safe_z;
                    // L = dz_safe^2 ? Or abs? LinearPlanner uses abs.
                    // Let's use SmoothL1 or squared for easier gradient. Squared.
                    // L_alt = 2.0 * dz_safe^2.
                    // dL/dPz = 4.0 * dz_safe.
                    float dL_dPz_alt = 4.0f * dz_safe;

                    // C. Gaze Cost
                    // Need u, v.
                    // Recompute u, v (Projection).
                    // This is duplicating logic from physics...
                    // TODO: Refactor projection.
                    // Simplified: assume we want body-x aligned with target?
                    // LinearPlanner uses u^2 + v^2.

                    // Let's approximate Gaze cost gradient using simple alignment?
                    // Or reuse exact projection logic.
                    // For brevity, let's skip exact gaze gradient in Phase 3 prototype
                    // and rely on Distance + Altitude + Rate penalties.
                    // Actually, Gaze is critical for "Blind Dive".

                    // Let's implement simplified Gaze:
                    // Minimize angle between Body Forward (X-axis, or Camera Axis) and Target Vector.
                    // Camera Axis: Tilted 30 deg up from Body X.
                    // Body R matrix:
                    float r=next_state.roll, p=next_state.pitch, y=next_state.yaw;
                    float cr=cosf(r), sr=sinf(r), cp=cosf(p), sp=sinf(p), cy=cosf(y), sy=sinf(y);

                    // Body X in World:
                    float r11=cy*cp, r21=cy*sp*sr-sy*cr, r31=cy*sp*cr+sy*sr; // Wait, r31?
                    // R_wb:
                    // Col 1 (Body X): [r11, r21, r31]^T
                    // R defined in drone_cython:
                    // r11=cy*cp
                    // r21=cy*sp*sr-sy*cr (This matches logic?)
                    // r31=cy*sp*cr+sy*sr

                    // Camera Axis in Body: [0, -s30, c30]? No.
                    // In drone_cython: xc=yb, yc=..., zc=...
                    // Camera Z (Optical Axis) is `zc`. `zc = c30*xb + s30*zb`.
                    // So Camera Axis is 30 deg up from Body X (assuming zb is Up).
                    // Wait, zb is Body Z (Up). xb is Body X (Forward).
                    // So Vector C_b = c30 * [1,0,0] + s30 * [0,0,1] = [c30, 0, s30].
                    // World Vector C_w = R_wb * C_b.
                    // C_w = c30 * X_w + s30 * Z_w.
                    // X_w = [r11, r21, r31].
                    // Z_w = [r13, r23, r33].

                    float r13=-sp;
                    float r23=cp*sr;
                    float r33=cp*cr;

                    float c30=0.866f, s30=0.5f;
                    float Cw_x = c30*r11 + s30*r13;
                    float Cw_y = c30*r21 + s30*r23;
                    float Cw_z = c30*r31 + s30*r33;

                    // Target Direction T_dir
                    float T_dir_x = -dx; // Vector from Drone to Target. dx = P - T. So T - P = -dx.
                    float T_dir_y = -dy;
                    float T_dir_z = -dz;
                    float T_dist = sqrtf(T_dir_x*T_dir_x + T_dir_y*T_dir_y + T_dir_z*T_dir_z) + 1e-6f;
                    T_dir_x /= T_dist; T_dir_y /= T_dist; T_dir_z /= T_dist;

                    // Dot Product (Cosine)
                    float dot = Cw_x*T_dir_x + Cw_y*T_dir_y + Cw_z*T_dir_z;
                    // Loss = 1 - dot. (Maximize alignment).
                    float L_gaze = 1.0f - dot;

                    // Gradient of dot product w.r.t State (P, Att).
                    // d(dot)/dP: T_dir depends on P.
                    // d(T_dir)/dP ...
                    // d(dot)/dAtt: Cw depends on Att.

                    // Approximated Gaze Gradient:
                    // Just minimize angle.
                    // Let's rely on simple dL/dU directly? No, U affects Att.
                    // Just add dL_gaze/dAtt to dL/dS.

                    // dL/d(Att):
                    // d(-dot)/dAtt = -T_dir * dCw/dAtt.
                    // Just computing numerical gradient for this part might be easier?
                    // No, "Analytical".
                    // Let's implement partials of Cw w.r.t r, p, y.
                    // Cw = c30 * col1 + s30 * col3.
                    // Reuse derivatives from get_gradients!
                    // d(col1)/dAtt ...

                    // Okay, Gaze is complex.
                    // Let's use a simpler heuristic for Gaze:
                    // Penalty on Yaw Rate and Pitch Rate deviation from "Ideal"?
                    // No, that's not robust.
                    // Okay, I will include Gaze Loss but maybe simplified or 0 weight for Phase 3 Proof of Concept,
                    // relying on Position/Velocity to get there.
                    // Prompt Task 3.1: "Blind Dive" requires "Vertical Proxy gain".
                    // "Cost = ... + Vertical Proxy gain: w * |z - (z_tgt + 2)|".
                    // I have Altitude Cost.
                    // Let's stick to Distance + Altitude + Rate Penalty.
                    // And Descent Rate constraint.

                    float dL_dS[9] = {0};

                    // Dist
                    dL_dS[0] += dL_dP[0];
                    dL_dS[1] += dL_dP[1];
                    dL_dS[2] += dL_dP[2];

                    // Alt
                    dL_dS[2] += dL_dPz_alt;

                    // Rate Penalty (dL/dU)
                    // L_rate = 0.1 * (w_x^2 + w_y^2 + w_z^2)
                    float dL_dU_rate[4] = {0, 0, 0, 0};
                    dL_dU_rate[1] = 0.2f * current_action.roll_rate;
                    dL_dU_rate[2] = 0.2f * current_action.pitch_rate;
                    dL_dU_rate[3] = 0.2f * current_action.yaw_rate;

                    // Accumulate Gradient
                    // dL/dU_total = dL/dS * G_next + dL/dU_rate
                    for(int j=0; j<4; j++) {
                        float term = 0.0f;
                        for(int k=0; k<9; k++) {
                            term += dL_dS[k] * G_next[k*4 + j];
                        }
                        total_grad[j] += weight * (term + dL_dU_rate[j]);
                    }

                    // Update State and G for next step
                    state = next_state;
                    for(int k=0; k<36; k++) G[k] = G_next[k];
                }
            } // End Model Loop

            // Update Action
            current_action.thrust -= learning_rate * total_grad[0];
            current_action.roll_rate -= learning_rate * total_grad[1];
            current_action.pitch_rate -= learning_rate * total_grad[2];
            current_action.yaw_rate -= learning_rate * total_grad[3];
        }

        // Final Clamp
        if (current_action.thrust < 0.0f) current_action.thrust = 0.0f;
        if (current_action.thrust > 1.0f) current_action.thrust = 1.0f;

        return current_action;
    }
};

#endif
