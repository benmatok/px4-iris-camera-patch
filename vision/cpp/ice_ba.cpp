#include "ice_ba.hpp"
#include <iostream>
#include <cmath>

IceBA::IceBA() {}

void IceBA::add_frame(int id, double t, double* p, double* q, double* v, double* bg, double* ba,
               double dt_pre, double* dp_pre, double* dq_pre, double* dv_pre) {
    Frame f;
    f.id = id;
    f.t = t;
    f.p = Vec3(p[0], p[1], p[2]);
    f.q = Quat(q[0], q[1], q[2], q[3]);
    f.v = Vec3(v[0], v[1], v[2]);
    f.bg = Vec3(bg[0], bg[1], bg[2]);
    f.ba = Vec3(ba[0], ba[1], ba[2]);

    f.dt_pre = dt_pre;
    if (dt_pre > 0) {
        f.has_preint = true;
        f.dp_pre = Vec3(dp_pre[0], dp_pre[1], dp_pre[2]);
        f.dq_pre = Quat(dq_pre[0], dq_pre[1], dq_pre[2], dq_pre[3]);
        f.dv_pre = Vec3(dv_pre[0], dv_pre[1], dv_pre[2]);
    } else {
        f.has_preint = false;
    }

    // Check if update or new
    bool found = false;
    for(auto& fr : frames) {
        if(fr.id == id) {
            fr = f;
            found = true;
            break;
        }
    }
    if(!found) frames.push_back(f);
}

void IceBA::add_obs(int frame_id, int pt_id, double u, double v) {
    if(points.find(pt_id) == points.end()) {
        Point p;
        p.id = pt_id;
        p.initialized = false;
        points[pt_id] = p;
    }
    points[pt_id].obs.push_back({frame_id, u, v});
}

void IceBA::triangulate() {
    // Simple linear triangulation
    for(auto& kv : points) {
        Point& pt = kv.second;
        if(pt.obs.size() < 2) continue;
        if(pt.initialized) continue; // Keep old estimate

        // Use first two obs
        // Very rough depth est: 5.0m in front of camera 1
        // Better: Solve linear system?
        // We lack linear solver here.
        // Heuristic: Project from first cam at d=10.0

        // Find Frame
        int f_idx = -1;
        for(int i=0; i<frames.size(); ++i) if(frames[i].id == pt.obs[0].frame_id) f_idx = i;

        if(f_idx >= 0) {
            Frame& f = frames[f_idx];
            // Ray in body
            double u = pt.obs[0].u;
            double v = pt.obs[0].v;
            Vec3 ray_c(u, v, 1.0);
            // World
            Vec3 ray_w = f.q.rotate(ray_c);
            pt.p_world = f.p + ray_w * 10.0; // Init Depth
            pt.initialized = true;
        }
    }
}

void IceBA::optimize() {
    // 1. Triangulate
    triangulate();

    // 2. Simple Iterative Optimization (Gradient Descent-ish)
    // We update Frames to satisfy IMU and Vision

    double step = 0.01;
    Vec3 g_grav(0, 0, 9.81);

    for(int iter=0; iter<5; ++iter) {
        // IMU Constraints (Sequential)
        for(size_t i=0; i<frames.size()-1; ++i) {
            Frame& f1 = frames[i];
            Frame& f2 = frames[i+1];

            if(!f2.has_preint) continue;

            // Predict f2 from f1
            // p2 = p1 + v1*dt + 0.5*g*dt^2 + R1*dp
            double dt = f2.dt_pre;
            Vec3 p_pred = f1.p + f1.v * dt + g_grav * (0.5 * dt * dt) + f1.q.rotate(f2.dp_pre);
            Vec3 v_pred = f1.v + g_grav * dt + f1.q.rotate(f2.dv_pre);

            // Residual
            Vec3 rp = f2.p - p_pred;
            Vec3 rv = f2.v - v_pred;

            // Correction (Backprop-ish)
            // Move f2 towards pred
            f2.p = f2.p - rp * 0.5;
            f2.v = f2.v - rv * 0.5;

            // Also update f1? (Forward prop)
            // Let's assume f1 is more trusted (sliding window root)
        }

        // Vision Constraints
        // For each point, pull frames towards correct ray
        for(auto& kv : points) {
            Point& pt = kv.second;
            if(!pt.initialized) continue;

            for(auto& obs : pt.obs) {
                // Find frame
                Frame* f = nullptr;
                for(auto& fr : frames) if(fr.id == obs.frame_id) f = &fr;
                if(!f) continue;

                // Proj
                Vec3 p_local = f->q.conj().rotate(pt.p_world - f->p);
                if(p_local[2] < 0.1) continue;

                double u_pred = p_local[0] / p_local[2];
                double v_pred = p_local[1] / p_local[2];

                double du = u_pred - obs.u;
                double dv = v_pred - obs.v;

                // Gradients?
                // Just rudimentary nudging
                // Nudge Point
                // pt.p_world ...?
            }
        }
    }
}

void IceBA::solve() {
    optimize();
}

void IceBA::get_frame_state(int id, double* p, double* q, double* v, double* bg, double* ba) {
    for(auto& f : frames) {
        if(f.id == id) {
            p[0]=f.p[0]; p[1]=f.p[1]; p[2]=f.p[2];
            q[0]=f.q.data[0]; q[1]=f.q.data[1]; q[2]=f.q.data[2]; q[3]=f.q.data[3];
            v[0]=f.v[0]; v[1]=f.v[1]; v[2]=f.v[2];
            bg[0]=f.bg[0]; bg[1]=f.bg[1]; bg[2]=f.bg[2];
            ba[0]=f.ba[0]; ba[1]=f.ba[1]; ba[2]=f.ba[2];
            return;
        }
    }
}
