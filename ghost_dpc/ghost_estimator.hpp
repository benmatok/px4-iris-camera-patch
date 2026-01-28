#ifndef GHOST_ESTIMATOR_HPP
#define GHOST_ESTIMATOR_HPP

#include "ghost_model.hpp"
#include <vector>
#include <cmath>
#include <numeric>

class GhostEstimator {
public:
    std::vector<GhostModel> models;
    std::vector<float> probabilities;
    float lambda_param = 5.0f; // Sensitivity parameter

    GhostEstimator() {}

    void initialize(const std::vector<GhostModel>& initial_models) {
        models = initial_models;
        int n = models.size();
        if (n > 0) {
            probabilities.assign(n, 1.0f / n);
        }
    }

    // Returns the index of the most likely model
    int get_best_model_index() const {
        int best = 0;
        float max_p = -1.0f;
        for (size_t i = 0; i < probabilities.size(); i++) {
            if (probabilities[i] > max_p) {
                max_p = probabilities[i];
                best = i;
            }
        }
        return best;
    }

    // Returns weighted average model
    GhostModel get_weighted_model() const {
        GhostModel avg;
        avg.mass = 0;
        avg.drag_coeff = 0;
        avg.thrust_coeff = 0;
        avg.wind_x = 0;
        avg.wind_y = 0;

        for (size_t i = 0; i < models.size(); i++) {
            float p = probabilities[i];
            avg.mass += models[i].mass * p;
            avg.drag_coeff += models[i].drag_coeff * p;
            avg.thrust_coeff += models[i].thrust_coeff * p;
            avg.wind_x += models[i].wind_x * p;
            avg.wind_y += models[i].wind_y * p;
        }
        return avg;
    }

    void update(const GhostState& state, const GhostAction& action, const float* measured_accel, float dt) {
        // measured_accel is float[3] (ax, ay, az) in WORLD frame (or Body? IMU usually Body)
        // Prompt says "Real IMU Acceleration". IMU measures proper acceleration (excludes gravity usually, or includes? Accelerometer measures f = a - g. So includes gravity component).
        // If drone is hovering, a=0, g=-9.81. Accel measures +9.81 Up (Reaction force).
        // Or simulation output "accel" might be World Acceleration (kinematic).
        // "Input: Real IMU Acceleration (a_real)."
        // "Compute Error: || a_real - a_pred ||^2".

        // In this simulation environment, we likely have access to World Acceleration or Body Acceleration.
        // Let's assume we pass World Kinematic Acceleration (dv/dt) for simplicity, or match what the Sim provides.
        // If Sim provides "IMU", we need to know what it is.
        // But usually, we compare Kinematic Acceleration.
        // Let's assume `measured_accel` is Kinematic Acceleration [ax, ay, az] in World Frame.

        // Step 1: Compute Predicted Acceleration for each model
        std::vector<float> likelihoods(models.size());
        float sum_p = 0.0f;

        for (size_t i = 0; i < models.size(); i++) {
            // Re-use logic from GhostPhysics::step, but we just need Acceleration.
            // We can refactor step to extract acceleration, or just copy-paste for efficiency.

            const GhostModel& m = models[i];

            // Thrust Force
            float max_thrust = GhostPhysics::MAX_THRUST_BASE * m.thrust_coeff;
            float thrust_force = action.thrust * max_thrust;
            if (thrust_force < 0) thrust_force = 0;

            // Note: Use NEXT attitude for thrust direction?
            // "Real IMU" is measured over the step dt? Or instantaneous?
            // If instantaneous at state t, we use state attitude.
            // If average over step, we use average attitude.
            // Let's use STATE attitude (instantaneous prediction).

            float cr = cosf(state.roll);
            float sr = sinf(state.roll);
            float cp = cosf(state.pitch);
            float sp = sinf(state.pitch);
            float cy = cosf(state.yaw);
            float sy = sinf(state.yaw);

            float ax_dir = cy * sp * cr + sy * sr;
            float ay_dir = sy * sp * cr - cy * sr;
            float az_dir = cp * cr;

            float ax_thrust = thrust_force * ax_dir / m.mass;
            float ay_thrust = thrust_force * ay_dir / m.mass;
            float az_thrust = thrust_force * az_dir / m.mass;

            float ax_drag = -m.drag_coeff * (state.vx - m.wind_x);
            float ay_drag = -m.drag_coeff * (state.vy - m.wind_y);
            float az_drag = -m.drag_coeff * state.vz;

            float pred_ax = ax_thrust + ax_drag;
            float pred_ay = ay_thrust + ay_drag;
            float pred_az = az_thrust + az_drag - GhostPhysics::G;

            float dx = measured_accel[0] - pred_ax;
            float dy = measured_accel[1] - pred_ay;
            float dz = measured_accel[2] - pred_az;

            float error_sq = dx*dx + dy*dy + dz*dz;

            float likelihood = expf(-lambda_param * error_sq);
            likelihoods[i] = likelihood;
        }

        // Update Probabilities
        for (size_t i = 0; i < models.size(); i++) {
            probabilities[i] *= likelihoods[i];
            // Lower bound to prevent death?
            if (probabilities[i] < 1e-6f) probabilities[i] = 1e-6f;
            sum_p += probabilities[i];
        }

        // Normalize
        for (size_t i = 0; i < models.size(); i++) {
            probabilities[i] /= sum_p;
        }
    }
};

#endif
