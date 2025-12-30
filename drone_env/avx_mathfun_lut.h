#ifndef AVX_MATHFUN_LUT_H
#define AVX_MATHFUN_LUT_H

#include <immintrin.h>
#include <cmath>
#include <vector>

// Constants
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// LUT Size must be power of 2
#define LUT_SIZE 2048
#define LUT_MASK 2047

struct SinLUT {
    float table[LUT_SIZE + 4]; // +4 for padding/overflow ease
    float scale;

    SinLUT() {
        scale = (float)LUT_SIZE / (2.0f * M_PI);
        for (int i = 0; i < LUT_SIZE; ++i) {
            table[i] = sinf((float)i * 2.0f * M_PI / (float)LUT_SIZE);
        }
        // Cyclic padding for interpolation
        table[LUT_SIZE] = table[0];
        table[LUT_SIZE+1] = table[1];
        table[LUT_SIZE+2] = table[2];
        table[LUT_SIZE+3] = table[3];
    }
};

static SinLUT global_sin_lut;

// Catmull-Rom Spline
// f(t) = 0.5 * ( (2 p1) + (-p0 + p2) t + (2 p0 - 5 p1 + 4 p2 - p3) t^2 + (-p0 + 3 p1 - 3 p2 + p3) t^3 )
inline __m256 cubic_interp(__m256 p0, __m256 p1, __m256 p2, __m256 p3, __m256 t) {
    __m256 c05 = _mm256_set1_ps(0.5f);
    __m256 c2 = _mm256_set1_ps(2.0f);
    __m256 c3 = _mm256_set1_ps(3.0f);
    __m256 c4 = _mm256_set1_ps(4.0f);
    __m256 c5 = _mm256_set1_ps(5.0f);

    __m256 t2 = _mm256_mul_ps(t, t);
    __m256 t3 = _mm256_mul_ps(t2, t);

    // term0 = 2 * p1
    __m256 term0 = _mm256_mul_ps(c2, p1);

    // term1 = (-p0 + p2) * t
    __m256 term1_coef = _mm256_sub_ps(p2, p0);
    __m256 term1 = _mm256_mul_ps(term1_coef, t);

    // term2 = (2 p0 - 5 p1 + 4 p2 - p3) * t^2
    __m256 t2_c1 = _mm256_mul_ps(c2, p0);
    __m256 t2_c2 = _mm256_mul_ps(c5, p1);
    __m256 t2_c3 = _mm256_mul_ps(c4, p2);
    __m256 term2_coef = _mm256_sub_ps(_mm256_add_ps(t2_c1, t2_c3), _mm256_add_ps(t2_c2, p3));
    __m256 term2 = _mm256_mul_ps(term2_coef, t2);

    // term3 = (-p0 + 3 p1 - 3 p2 + p3) * t^3
    __m256 t3_c2 = _mm256_mul_ps(c3, p1);
    __m256 t3_c3 = _mm256_mul_ps(c3, p2);
    __m256 term3_coef = _mm256_add_ps(_mm256_sub_ps(t3_c2, t3_c3), _mm256_sub_ps(p3, p0));
    __m256 term3 = _mm256_mul_ps(term3_coef, t3);

    __m256 res = _mm256_add_ps(term0, term1);
    res = _mm256_add_ps(res, term2);
    res = _mm256_add_ps(res, term3);
    res = _mm256_mul_ps(res, c05);

    return res;
}

inline __m256 lut_sin256_ps(__m256 x) {
    __m256 scale = _mm256_set1_ps(global_sin_lut.scale);
    __m256 vals = _mm256_mul_ps(x, scale);

    // Handle negatives by adding a large multiple of LUT_SIZE
    // Or just using floor.
    // _mm256_floor_ps returns float.
    __m256 fidx = _mm256_floor_ps(vals);
    __m256 t = _mm256_sub_ps(vals, fidx);

    __m256i idx = _mm256_cvtps_epi32(fidx);
    __m256i mask = _mm256_set1_epi32(LUT_MASK);

    // p0 = idx - 1
    // p1 = idx
    // p2 = idx + 1
    // p3 = idx + 2

    __m256i i1 = _mm256_and_si256(idx, mask);
    __m256i i0 = _mm256_and_si256(_mm256_sub_epi32(idx, _mm256_set1_epi32(1)), mask);
    __m256i i2 = _mm256_and_si256(_mm256_add_epi32(idx, _mm256_set1_epi32(1)), mask);
    __m256i i3 = _mm256_and_si256(_mm256_add_epi32(idx, _mm256_set1_epi32(2)), mask);

    float* base = global_sin_lut.table;
    __m256 p0 = _mm256_i32gather_ps(base, i0, 4);
    __m256 p1 = _mm256_i32gather_ps(base, i1, 4);
    __m256 p2 = _mm256_i32gather_ps(base, i2, 4);
    __m256 p3 = _mm256_i32gather_ps(base, i3, 4);

    return cubic_interp(p0, p1, p2, p3, t);
}

inline __m256 lut_cos256_ps(__m256 x) {
    __m256 pio2 = _mm256_set1_ps(1.57079632679f);
    return lut_sin256_ps(_mm256_add_ps(x, pio2));
}

#endif
