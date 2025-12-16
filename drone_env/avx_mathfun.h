/*
   AVX implementation of sin, cos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2007  Julien Pommier

   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.

   (Adapted for AVX2 by various contributors, simplified for this project)
*/

#ifndef AVX_MATHFUN_H
#define AVX_MATHFUN_H

#include <immintrin.h>

/* Defines for AVX */
#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))

typedef __m256 v8sf; // vector of 8 float (avx)
typedef __m256i v8si; // vector of 8 int (avx)

/* Constants */
#define _PS256_CONST(Name, Val)                                            \
  static const float _ps256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const int _pi32_256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PS256_CONST_TYPE(Name, Type, Val)                                 \
  static const Type _ps256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

_PS256_CONST(1  , 1.0f);
_PS256_CONST(0p5, 0.5f);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);
_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);

/* Sine / Cosine constants */
_PS256_CONST(fo_pi, 1.27323954473516f); // 4 / M_PI
_PS256_CONST(dp1, -0.78515625f);
_PS256_CONST(dp2, -2.4187564849853515625e-4f);
_PS256_CONST(dp3, -3.77489497744594108e-8f);
_PS256_CONST(sincof_p0, -1.9515295891E-4f);
_PS256_CONST(sincof_p1,  8.3321608736E-3f);
_PS256_CONST(sincof_p2, -1.6666654611E-1f);
_PS256_CONST(coscof_p0,  2.443315711809948E-5f);
_PS256_CONST(coscof_p1, -1.388731625493765E-3f);
_PS256_CONST(coscof_p2,  4.166664568298827E-2f);
_PS256_CONST(cephes_FOPI, 1.27323954473516f); // 4 / pi

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

// Removed duplicate definitions of minus_cephes_DP1 etc. assuming we use dp1 etc.
// But implementation uses `minus_cephes_DP1`. I will alias or rename.
// The previous code used `_ps256_minus_cephes_DP1`. I defined `_ps256_dp1`.
// I will just use `_ps256_dp1`.

/* Exp constants */
_PS256_CONST(exp_hi,	88.3762626647949f);
_PS256_CONST(exp_lo,	-88.3762626647949f);
_PS256_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS256_CONST(cephes_exp_C1, 0.693359375f);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4f);
_PS256_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1f);

inline v8sf exp256_ps(v8sf x) {
  v8sf tmp = _mm256_setzero_ps(), fx;
  v8si imm0;
  v8sf one = *(v8sf*)_ps256_1;

  x = _mm256_min_ps(x, *(v8sf*)_ps256_exp_hi);
  x = _mm256_max_ps(x, *(v8sf*)_ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_LOG2EF);
  fx = _mm256_add_ps(fx, *(v8sf*)_ps256_0p5);

  /* how to perform a floorf with SSE: just cast to int. */
  imm0 = _mm256_cvttps_epi32(fx);
  tmp  = _mm256_cvtepi32_ps(imm0);

  /* if greater, subtract 1 */
  v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OQ);
  v8sf mask_one = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask_one);

  tmp = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C1);
  v8sf z = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);

  v8sf y = *(v8sf*)_ps256_cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, *(v8si*)_pi32_256_0x7f);
  imm0 = _mm256_slli_epi32(imm0, 23);
  v8sf pow2n = _mm256_castsi256_ps(imm0);

  y = _mm256_mul_ps(y, pow2n);
  return y;
}

inline v8sf sin256_ps(v8sf x) {
  v8sf xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit, y;
  v8si imm0, imm2;

  v8sf sign_mask = *(v8sf*)_ps256_sign_mask;
  v8sf inv_sign_mask = *(v8sf*)_ps256_inv_sign_mask;
  v8sf fo_pi = *(v8sf*)_ps256_fo_pi;
  v8sf dp1 = *(v8sf*)_ps256_dp1;
  v8sf dp2 = *(v8sf*)_ps256_dp2;
  v8sf dp3 = *(v8sf*)_ps256_dp3;
  // Use initialized pointers, not reloading constants if optimizing, but statics are fine.

  sign_bit = x;
  x = _mm256_and_ps(x, inv_sign_mask);
  sign_bit = _mm256_and_ps(sign_bit, sign_mask);

  y = _mm256_mul_ps(x, fo_pi);
  imm2 = _mm256_cvttps_epi32(y);
  imm2 = _mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);

  imm0 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  v8sf swap_sign_bit = _mm256_castsi256_ps(imm0);

  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, _mm256_setzero_si256());
  v8sf poly_mask = _mm256_castsi256_ps(imm2);

  xmm1 = _mm256_mul_ps(y, dp1);
  xmm2 = _mm256_mul_ps(y, dp2);
  xmm3 = _mm256_mul_ps(y, dp3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

  sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

  v8sf z = _mm256_mul_ps(x,x);

  // Cosine Poly
  y = *(v8sf*)_ps256_coscof_p0;
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  y = _mm256_sub_ps(y, _mm256_mul_ps(z, *(v8sf*)_ps256_0p5));
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1);

  // Sine Poly
  v8sf y2 = *(v8sf*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  v8sf ysin2 = _mm256_and_ps(poly_mask, y2);
  v8sf ysin1 = _mm256_andnot_ps(poly_mask, y);
  y2 = _mm256_add_ps(ysin1, ysin2);

  return _mm256_xor_ps(y2, sign_bit);
}

inline v8sf cos256_ps(v8sf x) {
  // cos(x) = sin(x + pi/2)
  v8sf pio2 = _mm256_set1_ps(1.57079632679489661923f);
  x = _mm256_add_ps(x, pio2);
  return sin256_ps(x);
}

inline void sincos256_ps(v8sf x, v8sf *s, v8sf *c) {
  v8sf xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
  v8si imm0, imm2, imm4;

  v8sf sign_mask = *(v8sf*)_ps256_sign_mask;
  v8sf inv_sign_mask = *(v8sf*)_ps256_inv_sign_mask;
  v8sf fo_pi = *(v8sf*)_ps256_fo_pi;
  v8sf dp1 = *(v8sf*)_ps256_dp1;
  v8sf dp2 = *(v8sf*)_ps256_dp2;
  v8sf dp3 = *(v8sf*)_ps256_dp3;

  sign_bit_sin = x;
  x = _mm256_and_ps(x, inv_sign_mask);
  sign_bit_sin = _mm256_and_ps(sign_bit_sin, sign_mask);

  y = _mm256_mul_ps(x, fo_pi);
  imm2 = _mm256_cvttps_epi32(y);
  imm2 = _mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);

  v8si quadrant = imm2; // Save quadrant for cos sign calc

  // Sin Sign
  imm0 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

  // Poly Mask
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, _mm256_setzero_si256());
  v8sf poly_mask = _mm256_castsi256_ps(imm2);

  // Range Reduction
  xmm1 = _mm256_mul_ps(y, dp1);
  xmm2 = _mm256_mul_ps(y, dp2);
  xmm3 = _mm256_mul_ps(y, dp3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

  // Apply Sin Sign Bit
  sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  v8sf z = _mm256_mul_ps(x,x);

  // Cosine Poly
  y = *(v8sf*)_ps256_coscof_p0;
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  y = _mm256_sub_ps(y, _mm256_mul_ps(z, *(v8sf*)_ps256_0p5));
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1);

  // Sine Poly
  v8sf y2 = *(v8sf*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  // Sin Selection
  v8sf ysin2 = _mm256_and_ps(poly_mask, y2);
  v8sf ysin1 = _mm256_andnot_ps(poly_mask, y);
  *s = _mm256_add_ps(ysin1, ysin2);
  *s = _mm256_xor_ps(*s, sign_bit_sin);

  // Cos Selection
  // If poly_mask is set (Sin Poly used for Sin), then for Cos we use Cos Poly (y).
  // No!
  // Quadrant 0 (imm2=0): Sin uses y2 (Sin Poly). Cos uses y (Cos Poly).
  // Quadrant 1 (imm2=2): Sin uses y (Cos Poly). Cos uses y2 (Sin Poly).
  // poly_mask = (imm2 & 2) == 0 ? 0xFF : 0.
  // So if imm2=0, poly_mask=1. Sin uses ysin2=y2 (sin poly). Correct.
  // Then Cos should use y (cos poly).
  // So for Cos: if poly_mask=1, use y.
  // v8sf ycos2 = _mm256_and_ps(poly_mask, y);
  // v8sf ycos1 = _mm256_andnot_ps(poly_mask, y2);
  // *c = _mm256_add_ps(ycos1, ycos2);
  // Wait, I reused 'y' variable for Cosine Poly result.
  // And 'y2' for Sine Poly.
  // So yes, I can reuse them.

  v8sf ycos2 = _mm256_and_ps(poly_mask, y);
  v8sf ycos1 = _mm256_andnot_ps(poly_mask, y2);
  *c = _mm256_add_ps(ycos1, ycos2);

  // Cos Sign
  imm4 = _mm256_sub_epi32(quadrant, *(v8si*)_pi32_256_2);
  imm4 = _mm256_andnot_si256(imm4, *(v8si*)_pi32_256_4);
  imm4 = _mm256_slli_epi32(imm4, 29);
  v8sf swap_sign_bit_cos = _mm256_castsi256_ps(imm4);

  // Note: original sign_bit_sin variable has been modified!
  // I need the ORIGINAL x sign bit.
  // But I overwrote sign_bit_sin.
  // `sign_bit_sin = _mm256_and_ps(sign_bit_sin, sign_mask);`
  // And then `sign_bit_sin = _mm256_xor_ps(...)`.
  // So I can't recover x sign from it easily without reversing xor.
  // But I can recalculate it or save it?
  // Actually, I should have saved `x_sign_bit`.
  // Wait, `sign_bit_sin` is initialized to `x` then masked.
  // I will introduce `x_sign_bit` variable.

  // Re-deriving x sign bit (simpler than changing above code layout significantly):
  // Or simpler:
  // x_sign_bit = sign_bit_sin ^ swap_sign_bit_sin
  v8sf x_sign_bit = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  v8sf sign_bit_cos = _mm256_xor_ps(x_sign_bit, swap_sign_bit_cos);
  *c = _mm256_xor_ps(*c, sign_bit_cos);
}

#endif
