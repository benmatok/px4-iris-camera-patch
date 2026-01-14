# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt, atan2, exp, fabs, pow, M_PI, cbrt
from libc.stdlib cimport malloc, free

# AVX Intrinsics definitions
cdef extern from "<immintrin.h>" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(float *p)
    void _mm256_storeu_ps(float *p, __m256 a)
    __m256 _mm256_set1_ps(float a)
    __m256 _mm256_setzero_ps()
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_sub_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_div_ps(__m256 a, __m256 b)
    __m256 _mm256_sqrt_ps(__m256 a)
    __m256 _mm256_max_ps(__m256 a, __m256 b)
    __m256 _mm256_and_ps(__m256 a, __m256 b)
    __m256 _mm256_cmp_ps(__m256 a, __m256 b, int imm8)
    # Constants for cmp
    int _CMP_GT_OQ

cdef float[5] GAUSS_KERNEL = [0.06136, 0.24477, 0.38774, 0.24477, 0.06136]

# --- Convolution Helpers ---

cdef inline float process_conv_h(int r, int c, int w, float[:, ::1] src) nogil:
    cdef float val = 0.0
    cdef int k
    for k in range(-2, 3):
        if c + k >= 0 and c + k < w:
            val += src[r, c + k] * GAUSS_KERNEL[k + 2]
    return val

cdef void convolve_horizontal(float[:, ::1] src, float[:, ::1] dst, int h, int w) noexcept nogil:
    cdef int r, c
    for r in prange(h, schedule='static'):
        for c in range(w):
            dst[r, c] = process_conv_h(r, c, w, src)

cdef inline float process_conv_v(int r, int c, int h, float[:, ::1] src) nogil:
    cdef float val = 0.0
    cdef int k
    for k in range(-2, 3):
        if r + k >= 0 and r + k < h:
            val += src[r + k, c] * GAUSS_KERNEL[k + 2]
    return val

cdef void convolve_vertical(float[:, ::1] src, float[:, ::1] dst, int h, int w) noexcept nogil:
    cdef int r, c
    for r in prange(h, schedule='static'):
        for c in range(w):
            dst[r, c] = process_conv_v(r, c, h, src)

cdef void gaussian_blur(float[:, ::1] src, float[:, ::1] dst, float[:, ::1] temp, int h, int w) noexcept nogil:
    convolve_horizontal(src, temp, h, w)
    convolve_vertical(temp, dst, h, w)

cdef void downsample(float[:, ::1] src, float[:, ::1] dst, int h_src, int w_src) noexcept nogil:
    cdef int r, c
    cdef int h_dst = h_src // 2
    cdef int w_dst = w_src // 2
    for r in prange(h_dst, schedule='static'):
        for c in range(w_dst):
            dst[r, c] = src[r * 2, c * 2]

# --- AVX Helper: Absolute Value ---
cdef inline __m256 _mm256_abs_ps(__m256 a) noexcept nogil:
    return _mm256_max_ps(a, _mm256_sub_ps(_mm256_setzero_ps(), a))

# --- Tile Processing with SIMD ---

cdef inline void process_tile_features_simd(
    int r_start, int r_end, int c_start, int c_end,
    int h, int w,
    float[:, ::1] img,
    float[:, ::1] orient_out,
    float[:, ::1] coher_out,
    float[:, ::1] hess_out,
    float[:, ::1] ged_out,
    float hess_scale_norm,
    float *Ix_buf, float *Iy_buf,
    int buf_stride
) noexcept nogil:

    # 1. Compute Gradients into Local Buffers
    cdef int h_tile_plus_halo = r_end - r_start + 2
    cdef int w_tile_plus_halo = c_end - c_start + 2

    cdef int br, bc
    cdef int global_r, global_c
    cdef float val_x, val_y

    for br in range(h_tile_plus_halo):
        global_r = r_start - 1 + br
        for bc in range(w_tile_plus_halo):
            global_c = c_start - 1 + bc

            if global_r >= 0 and global_r < h and global_c >= 0 and global_c < w:
                if global_c > 0 and global_c < w - 1:
                    val_x = (img[global_r, global_c + 1] - img[global_r, global_c - 1]) * 0.5
                else:
                    val_x = 0.0
                if global_r > 0 and global_r < h - 1:
                    val_y = (img[global_r + 1, global_c] - img[global_r - 1, global_c]) * 0.5
                else:
                    val_y = 0.0
            else:
                val_x = 0.0
                val_y = 0.0

            Ix_buf[br * buf_stride + bc] = val_x
            Iy_buf[br * buf_stride + bc] = val_y

    # 2. Compute Features (SIMD)
    cdef int r, c, i, j, k

    # SIMD constants
    cdef __m256 v_half = _mm256_set1_ps(0.5)
    cdef __m256 v_two = _mm256_set1_ps(2.0)
    cdef __m256 v_hess_scale = _mm256_set1_ps(hess_scale_norm)
    cdef __m256 v_epsilon = _mm256_set1_ps(1e-6)
    cdef __m256 v_zero = _mm256_setzero_ps()
    cdef __m256 v_four = _mm256_set1_ps(4.0)

    # Temp vars for scalar atan2 fallback
    cdef float[8] tmp_s_xy
    cdef float[8] tmp_denom

    # Vars declaration
    cdef __m256 v_s_xx, v_s_yy, v_s_xy
    cdef __m256 v_ix, v_iy
    cdef __m256 v_diff, v_sum_eigen, v_lambda1, v_lambda2, v_coher
    cdef __m256 v_ixx, v_iyy, v_ixy, v_det, v_hess
    cdef __m256 v_diff_term, v_xy_sq, v_sum_safe, v_denom

    # Vars for scalar gradient fallback (hessian)
    cdef __m256 v_ix_p1, v_ix_m1, v_iy_p1, v_iy_m1, v_ix_y_p1, v_ix_y_m1

    # Scalar fallbacks
    cdef float s_xx, s_yy, s_xy, val_ix, val_iy, diff, sum_eigen
    cdef float i_xx, i_yy, i_xy, det
    cdef float sum_in, sum_out, cnt_in, cnt_out, val_img
    cdef int curr_c

    cdef int valid_w = c_end - c_start
    cdef int vec_limit = c_start + (valid_w // 8) * 8

    for r in range(r_start, r_end):
        if r >= h: break
        br = r - r_start + 1

        # Vectorized Loop
        for c in range(c_start, vec_limit, 8):
            bc = c - c_start + 1

            # --- Structure Tensor ---
            v_s_xx = _mm256_setzero_ps()
            v_s_yy = _mm256_setzero_ps()
            v_s_xy = _mm256_setzero_ps()

            # 3x3 Window Accumulation
            for i in range(-1, 2):
                for j in range(-1, 2):
                    v_ix = _mm256_loadu_ps(&Ix_buf[(br+i) * buf_stride + (bc+j)])
                    v_iy = _mm256_loadu_ps(&Iy_buf[(br+i) * buf_stride + (bc+j)])

                    v_s_xx = _mm256_add_ps(v_s_xx, _mm256_mul_ps(v_ix, v_ix))
                    v_s_yy = _mm256_add_ps(v_s_yy, _mm256_mul_ps(v_iy, v_iy))
                    v_s_xy = _mm256_add_ps(v_s_xy, _mm256_mul_ps(v_ix, v_iy))

            v_diff_term = _mm256_sub_ps(v_s_xx, v_s_yy)
            v_diff_term = _mm256_mul_ps(v_diff_term, v_diff_term)

            v_xy_sq = _mm256_mul_ps(v_s_xy, v_s_xy)
            v_xy_sq = _mm256_mul_ps(v_xy_sq, v_four)

            v_diff = _mm256_sqrt_ps(_mm256_add_ps(v_diff_term, v_xy_sq))
            v_sum_eigen = _mm256_add_ps(v_s_xx, v_s_yy)

            v_sum_safe = _mm256_max_ps(v_sum_eigen, v_epsilon)
            v_coher = _mm256_div_ps(v_diff, v_sum_safe)

            _mm256_storeu_ps(&coher_out[r, c], v_coher)

            _mm256_storeu_ps(tmp_s_xy, v_s_xy)
            v_denom = _mm256_sub_ps(v_s_xx, v_s_yy)
            _mm256_storeu_ps(tmp_denom, v_denom)

            for k in range(8):
                orient_out[r, c+k] = 0.5 * atan2(2.0 * tmp_s_xy[k], tmp_denom[k])

            # --- Hessian ---
            v_ix_p1 = _mm256_loadu_ps(&Ix_buf[br * buf_stride + (bc+1)])
            v_ix_m1 = _mm256_loadu_ps(&Ix_buf[br * buf_stride + (bc-1)])
            v_ixx = _mm256_mul_ps(_mm256_sub_ps(v_ix_p1, v_ix_m1), v_half)

            v_iy_p1 = _mm256_loadu_ps(&Iy_buf[(br+1) * buf_stride + bc])
            v_iy_m1 = _mm256_loadu_ps(&Iy_buf[(br-1) * buf_stride + bc])
            v_iyy = _mm256_mul_ps(_mm256_sub_ps(v_iy_p1, v_iy_m1), v_half)

            v_ix_y_p1 = _mm256_loadu_ps(&Ix_buf[(br+1) * buf_stride + bc])
            v_ix_y_m1 = _mm256_loadu_ps(&Ix_buf[(br-1) * buf_stride + bc])
            v_ixy = _mm256_mul_ps(_mm256_sub_ps(v_ix_y_p1, v_ix_y_m1), v_half)

            v_det = _mm256_sub_ps(
                _mm256_mul_ps(v_ixx, v_iyy),
                _mm256_mul_ps(v_ixy, v_ixy)
            )
            v_hess = _mm256_mul_ps(_mm256_abs_ps(v_det), v_hess_scale)
            _mm256_storeu_ps(&hess_out[r, c], v_hess)

            # --- GED ---
            for k in range(8):
                sum_in = 0.0; cnt_in = 0.0
                sum_out = 0.0; cnt_out = 0.0
                curr_c = c + k

                for i in range(-3, 4):
                    for j in range(-3, 4):
                        if r+i >= 0 and r+i < h and curr_c+j >= 0 and curr_c+j < w:
                            val_img = img[r+i, curr_c+j]
                            if i >= -1 and i <= 1 and j >= -1 and j <= 1:
                                sum_in += val_img
                                cnt_in += 1.0
                            else:
                                sum_out += val_img
                                cnt_out += 1.0
                if cnt_in > 0 and cnt_out > 0:
                    ged_out[r, curr_c] = fabs(sum_in/cnt_in - sum_out/cnt_out)
                else:
                    ged_out[r, curr_c] = 0.0

        # Handle Remaining Columns Scalar
        for c in range(vec_limit, c_end):
            bc = c - c_start + 1

            # ST Scalar
            s_xx = 0.0; s_yy = 0.0; s_xy = 0.0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    val_ix = Ix_buf[(br+i) * buf_stride + (bc+j)]
                    val_iy = Iy_buf[(br+i) * buf_stride + (bc+j)]
                    s_xx += val_ix * val_ix
                    s_yy += val_iy * val_iy
                    s_xy += val_ix * val_iy

            diff = sqrt((s_xx - s_yy)**2 + 4 * s_xy * s_xy)
            sum_eigen = s_xx + s_yy

            if sum_eigen > 1e-6:
                coher_out[r, c] = ((sum_eigen + diff)/2.0 - (sum_eigen - diff)/2.0) / sum_eigen
            else:
                coher_out[r, c] = 0.0
            orient_out[r, c] = 0.5 * atan2(2 * s_xy, s_xx - s_yy)

            # Hessian Scalar
            i_xx = (Ix_buf[br * buf_stride + (bc+1)] - Ix_buf[br * buf_stride + (bc-1)]) * 0.5
            i_yy = (Iy_buf[(br+1) * buf_stride + bc] - Iy_buf[(br-1) * buf_stride + bc]) * 0.5
            i_xy = (Ix_buf[(br+1) * buf_stride + bc] - Ix_buf[(br-1) * buf_stride + bc]) * 0.5
            det = i_xx * i_yy - i_xy * i_xy
            hess_out[r, c] = fabs(det) * hess_scale_norm

            # GED Scalar
            sum_in = 0.0; cnt_in = 0.0
            sum_out = 0.0; cnt_out = 0.0
            for i in range(-3, 4):
                for j in range(-3, 4):
                    if r+i >= 0 and r+i < h and c+j >= 0 and c+j < w:
                        val_img = img[r+i, c+j]
                        if i >= -1 and i <= 1 and j >= -1 and j <= 1:
                            sum_in += val_img
                            cnt_in += 1.0
                        else:
                            sum_out += val_img
                            cnt_out += 1.0
            if cnt_in > 0 and cnt_out > 0:
                ged_out[r, c] = fabs(sum_in/cnt_in - sum_out/cnt_out)
            else:
                ged_out[r, c] = 0.0

cdef inline void collapse_features(int r, int w,
                                   float[:, ::1] o_l0, float[:, ::1] o_l1,
                                   float[:, ::1] c_l0, float[:, ::1] c_l1,
                                   float[:, ::1] h_l0, float[:, ::1] h_l1, float[:, ::1] h_l2,
                                   float[:, ::1] g_l0, float[:, ::1] g_l1, float[:, ::1] g_l2,
                                   float[:, :, ::1] output) noexcept nogil:
    cdef int c
    cdef float o0, o1, c0, c1, h0, h1, h2, g0, g1, g2
    cdef float drift

    for c in range(w):
        o0 = o_l0[r, c]
        c0 = c_l0[r, c]
        h0 = h_l0[r, c]
        g0 = g_l0[r, c]

        if r//2 < o_l1.shape[0] and c//2 < o_l1.shape[1]:
            o1 = o_l1[r//2, c//2]
            c1 = c_l1[r//2, c//2]
            h1 = h_l1[r//2, c//2]
            g1 = g_l1[r//2, c//2]
        else:
            o1 = 0.0; c1 = 0.0; h1 = 0.0; g1 = 0.0

        if r//4 < h_l2.shape[0] and c//4 < h_l2.shape[1]:
            h2 = h_l2[r//4, c//4]
            g2 = g_l2[r//4, c//4]
        else:
            h2 = 0.0; g2 = 0.0

        output[r, c, 0] = o0
        output[r, c, 1] = c0

        drift = o1 - o0
        if drift > M_PI * 0.5:
            drift -= M_PI
        elif drift < -M_PI * 0.5:
            drift += M_PI
        output[r, c, 2] = drift

        output[r, c, 3] = c0 - c1

        if h0 >= h1 and h0 >= h2:
            output[r, c, 4] = 0.0
        elif h1 >= h0 and h1 >= h2:
            output[r, c, 4] = 1.0
        else:
            output[r, c, 4] = 2.0

        output[r, c, 5] = cbrt(g0 * g1 * g2)

def compute_texture_hypercube(float[:, ::1] image, int levels=3):
    if levels != 3:
        raise ValueError("Texture Engine currently supports exactly 3 levels.")

    cdef int h = image.shape[0]
    cdef int w = image.shape[1]

    st_orientations = []
    st_coherences = []
    hessian_responses = []
    ged_responses = []

    cdef float[:, ::1] current_img = image
    cdef float[:, ::1] temp_blur_v
    cdef float[:, ::1] temp_blur_h
    cdef float[:, ::1] next_img

    cdef float[:, ::1] orient
    cdef float[:, ::1] coher
    cdef float[:, ::1] hess
    cdef float[:, ::1] ged

    cdef int l, ch, cw
    cdef int TILE_SIZE = 32
    cdef float scale_factor

    cdef int num_strips_r, num_strips_c, total_tiles, tile_idx
    cdef int strip_r, strip_c, r_start, r_end, c_start, c_end

    cdef float *Ix_buf
    cdef float *Iy_buf
    cdef int buf_w, buf_h, buf_size

    for l in range(levels):
        ch = current_img.shape[0]
        cw = current_img.shape[1]

        orient = np.empty((ch, cw), dtype=np.float32)
        coher = np.empty((ch, cw), dtype=np.float32)
        hess = np.empty((ch, cw), dtype=np.float32)
        ged = np.empty((ch, cw), dtype=np.float32)

        scale_factor = pow(2.0, l)

        num_strips_r = (ch + TILE_SIZE - 1) // TILE_SIZE
        num_strips_c = (cw + TILE_SIZE - 1) // TILE_SIZE
        total_tiles = num_strips_r * num_strips_c

        buf_h = TILE_SIZE + 2
        buf_w = TILE_SIZE + 2
        buf_size = buf_h * buf_w

        # We allocate buffers inside the loop to avoid race conditions and ensure thread safety.
        # Although malloc inside the loop adds overhead, it is negligible compared to the image processing.
        for tile_idx in prange(total_tiles, schedule='dynamic', nogil=True):
            Ix_buf = <float *> malloc(buf_size * sizeof(float))
            Iy_buf = <float *> malloc(buf_size * sizeof(float))

            if Ix_buf != NULL and Iy_buf != NULL:
                strip_r = tile_idx // num_strips_c
                strip_c = tile_idx % num_strips_c

                r_start = strip_r * TILE_SIZE
                r_end = r_start + TILE_SIZE
                if r_end > ch: r_end = ch

                c_start = strip_c * TILE_SIZE
                c_end = c_start + TILE_SIZE
                if c_end > cw: c_end = cw

                process_tile_features_simd(
                    r_start, r_end, c_start, c_end,
                    ch, cw,
                    current_img,
                    orient, coher, hess, ged,
                    pow(scale_factor, 4),
                    Ix_buf, Iy_buf, buf_w
                )

                free(Ix_buf)
                free(Iy_buf)

        st_orientations.append(orient)
        st_coherences.append(coher)
        hessian_responses.append(hess)
        ged_responses.append(ged)

        if l < levels - 1:
            temp_blur_h = np.empty((ch, cw), dtype=np.float32)
            temp_blur_v = np.empty((ch, cw), dtype=np.float32)
            gaussian_blur(current_img, temp_blur_v, temp_blur_h, ch, cw)
            next_img = np.empty((ch // 2, cw // 2), dtype=np.float32)
            downsample(temp_blur_v, next_img, ch, cw)
            current_img = next_img

    cdef int out_channels = 6
    cdef float[:, :, ::1] output = np.empty((h, w, out_channels), dtype=np.float32)

    cdef float[:, ::1] o_l0 = st_orientations[0]
    cdef float[:, ::1] o_l1 = st_orientations[1]

    cdef float[:, ::1] c_l0 = st_coherences[0]
    cdef float[:, ::1] c_l1 = st_coherences[1]

    cdef float[:, ::1] h_l0 = hessian_responses[0]
    cdef float[:, ::1] h_l1 = hessian_responses[1]
    cdef float[:, ::1] h_l2 = hessian_responses[2]

    cdef float[:, ::1] g_l0 = ged_responses[0]
    cdef float[:, ::1] g_l1 = ged_responses[1]
    cdef float[:, ::1] g_l2 = ged_responses[2]

    cdef int r

    for r in prange(h, schedule='static', nogil=True):
        collapse_features(r, w, o_l0, o_l1, c_l0, c_l1, h_l0, h_l1, h_l2, g_l0, g_l1, g_l2, output)

    return np.asarray(output)
