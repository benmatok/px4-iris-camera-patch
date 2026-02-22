#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <vector>
#include <cmath>
#include <iostream>

// Minimal Matrix/Vector Utils for BA
// We use std::vector for dynamic, raw arrays for fixed 3x3

struct Vec3 {
    double data[3];
    Vec3() { data[0]=0; data[1]=0; data[2]=0; }
    Vec3(double x, double y, double z) { data[0]=x; data[1]=y; data[2]=z; }
    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }

    Vec3 operator+(const Vec3& o) const { return Vec3(data[0]+o[0], data[1]+o[1], data[2]+o[2]); }
    Vec3 operator-(const Vec3& o) const { return Vec3(data[0]-o[0], data[1]-o[1], data[2]-o[2]); }
    Vec3 operator*(double s) const { return Vec3(data[0]*s, data[1]*s, data[2]*s); }
    double norm() const { return std::sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2]); }
};

struct Quat {
    double data[4]; // x, y, z, w
    Quat() { data[0]=0; data[1]=0; data[2]=0; data[3]=1; }
    Quat(double x, double y, double z, double w) { data[0]=x; data[1]=y; data[2]=z; data[3]=w; }

    // Rotation Matrix
    void to_matrix(double R[3][3]) const {
        double x=data[0], y=data[1], z=data[2], w=data[3];
        R[0][0] = 1 - 2*y*y - 2*z*z; R[0][1] = 2*x*y - 2*z*w; R[0][2] = 2*x*z + 2*y*w;
        R[1][0] = 2*x*y + 2*z*w; R[1][1] = 1 - 2*x*x - 2*z*z; R[1][2] = 2*y*z - 2*x*w;
        R[2][0] = 2*x*z - 2*y*w; R[2][1] = 2*y*z + 2*x*w; R[2][2] = 1 - 2*x*x - 2*y*y;
    }

    Vec3 rotate(const Vec3& v) const {
        double R[3][3];
        to_matrix(R);
        return Vec3(
            R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
            R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
            R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]
        );
    }

    Quat operator*(const Quat& q) const {
        return Quat(
            data[3]*q.data[0] + data[0]*q.data[3] + data[1]*q.data[2] - data[2]*q.data[1],
            data[3]*q.data[1] - data[0]*q.data[2] + data[1]*q.data[3] + data[2]*q.data[0],
            data[3]*q.data[2] + data[0]*q.data[1] - data[1]*q.data[0] + data[2]*q.data[3],
            data[3]*q.data[3] - data[0]*q.data[0] - data[1]*q.data[1] - data[2]*q.data[2]
        );
    }

    Quat conj() const { return Quat(-data[0], -data[1], -data[2], data[3]); }
};

#endif
