# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, atan2, exp, fabs, pow, M_PI

cdef float[5] GAUSS_KERNEL = [0.06136, 0.24477, 0.38774, 0.24477, 0.06136]

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

cdef void compute_gradients(float[:, ::1] img, float[:, ::1] Ix, float[:, ::1] Iy, int h, int w) noexcept nogil:
    cdef int r, c
    for r in prange(h, schedule='static'):
        for c in range(w):
            if c > 0 and c < w - 1:
                Ix[r, c] = (img[r, c + 1] - img[r, c - 1]) * 0.5
            else:
                Ix[r, c] = 0.0

            if r > 0 and r < h - 1:
                Iy[r, c] = (img[r + 1, c] - img[r - 1, c]) * 0.5
            else:
                Iy[r, c] = 0.0

cdef inline void process_st_pixel(int r, int c, int h, int w,
                                  float[:, ::1] Ix, float[:, ::1] Iy,
                                  float[:, ::1] orientation, float[:, ::1] coherence) noexcept nogil:
    cdef float s_xx = 0.0
    cdef float s_yy = 0.0
    cdef float s_xy = 0.0
    cdef float val_ix, val_iy
    cdef int i, j

    for i in range(-1, 2):
        for j in range(-1, 2):
            if r+i >= 0 and r+i < h and c+j >= 0 and c+j < w:
                val_ix = Ix[r+i, c+j]
                val_iy = Iy[r+i, c+j]
                s_xx += val_ix * val_ix
                s_yy += val_iy * val_iy
                s_xy += val_ix * val_iy

    cdef float diff = sqrt((s_xx - s_yy)**2 + 4 * s_xy * s_xy)
    cdef float sum_eigen = s_xx + s_yy
    cdef float lambda1 = (sum_eigen + diff) * 0.5
    cdef float lambda2 = (sum_eigen - diff) * 0.5

    if sum_eigen > 1e-6:
        coherence[r, c] = (lambda1 - lambda2) / (sum_eigen)
    else:
        coherence[r, c] = 0.0

    orientation[r, c] = 0.5 * atan2(2 * s_xy, s_xx - s_yy)

cdef void compute_structure_tensor_features(float[:, ::1] Ix, float[:, ::1] Iy,
                                            float[:, ::1] orientation, float[:, ::1] coherence,
                                            int h, int w) noexcept nogil:
    cdef int r, c
    for r in prange(h, schedule='static'):
        for c in range(w):
            process_st_pixel(r, c, h, w, Ix, Iy, orientation, coherence)

cdef inline void process_hessian_pixel(int r, int c, int h, int w,
                                       float[:, ::1] Ix, float[:, ::1] Iy,
                                       float[:, ::1] response, float scale_norm) noexcept nogil:
    cdef float i_xx, i_yy, i_xy

    if c > 0 and c < w - 1:
        i_xx = (Ix[r, c + 1] - Ix[r, c - 1]) * 0.5
    else:
        i_xx = 0.0

    if r > 0 and r < h - 1:
        i_yy = (Iy[r + 1, c] - Iy[r - 1, c]) * 0.5
    else:
        i_yy = 0.0

    if r > 0 and r < h - 1:
        i_xy = (Ix[r + 1, c] - Ix[r - 1, c]) * 0.5
    else:
        i_xy = 0.0

    cdef float det = i_xx * i_yy - i_xy * i_xy
    response[r, c] = fabs(det) * scale_norm

cdef void compute_hessian_features(float[:, ::1] Ix, float[:, ::1] Iy,
                                   float[:, ::1] response, float scale_norm,
                                   int h, int w) noexcept nogil:
    cdef int r, c
    for r in prange(h, schedule='static'):
        for c in range(w):
            process_hessian_pixel(r, c, h, w, Ix, Iy, response, scale_norm)

cdef inline void process_ged_pixel(int r, int c, int h, int w,
                                   float[:, ::1] img, float[:, ::1] output) noexcept nogil:
    cdef float sum_in = 0.0
    cdef float sum_out = 0.0
    cdef float cnt_in = 0.0
    cdef float cnt_out = 0.0
    cdef float val
    cdef int i, j

    for i in range(-1, 2):
        for j in range(-1, 2):
            if r+i >= 0 and r+i < h and c+j >= 0 and c+j < w:
                val = img[r+i, c+j]
                sum_in += val
                cnt_in += 1.0

    for i in range(-3, 4):
        for j in range(-3, 4):
            if r+i >= 0 and r+i < h and c+j >= 0 and c+j < w:
                if i >= -1 and i <= 1 and j >= -1 and j <= 1:
                    continue
                val = img[r+i, c+j]
                sum_out += val
                cnt_out += 1.0

    if cnt_in > 0 and cnt_out > 0:
        output[r, c] = fabs(sum_in/cnt_in - sum_out/cnt_out)
    else:
        output[r, c] = 0.0

cdef void compute_ged_features(float[:, ::1] img, float[:, ::1] output, int h, int w) noexcept nogil:
    cdef int r, c
    for r in prange(h, schedule='static'):
        for c in range(w):
            process_ged_pixel(r, c, h, w, img, output)

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

        # Scale Drift with cyclic wrap-around (period pi)
        drift = o1 - o0
        if drift > M_PI * 0.5:
            drift -= M_PI
        elif drift < -M_PI * 0.5:
            drift += M_PI
        output[r, c, 2] = drift

        output[r, c, 3] = c0 - c1

        # Scale selection (Max Response)
        if h0 >= h1 and h0 >= h2:
            output[r, c, 4] = 0.0
        elif h1 >= h0 and h1 >= h2:
            output[r, c, 4] = 1.0
        else:
            output[r, c, 4] = 2.0

        # Boundary: Geometric Mean (pow(g0*g1*g2, 1/3))
        # Add small epsilon to avoid zero issues if desired?
        # But geometric mean should be 0 if any is 0 (suppression).
        output[r, c, 5] = pow(g0 * g1 * g2, 0.3333333)

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
    cdef float[:, ::1] Ix
    cdef float[:, ::1] Iy
    cdef float[:, ::1] orient
    cdef float[:, ::1] coher
    cdef float[:, ::1] hess
    cdef float[:, ::1] ged

    cdef int l, ch, cw
    cdef float scale_factor

    for l in range(levels):
        ch = current_img.shape[0]
        cw = current_img.shape[1]

        Ix = np.zeros((ch, cw), dtype=np.float32)
        Iy = np.zeros((ch, cw), dtype=np.float32)
        orient = np.zeros((ch, cw), dtype=np.float32)
        coher = np.zeros((ch, cw), dtype=np.float32)
        hess = np.zeros((ch, cw), dtype=np.float32)
        ged = np.zeros((ch, cw), dtype=np.float32)

        compute_gradients(current_img, Ix, Iy, ch, cw)

        compute_structure_tensor_features(Ix, Iy, orient, coher, ch, cw)
        st_orientations.append(orient)
        st_coherences.append(coher)

        scale_factor = pow(2.0, l)
        # Normalization: pow(scale_factor, 4) matches Lindeberg's t^2 normalization (t = sigma^2)
        # t^2 det(H) = sigma^4 det(H).
        # We use scale_factor = 2^l approx sigma.
        compute_hessian_features(Ix, Iy, hess, pow(scale_factor, 4), ch, cw)
        hessian_responses.append(hess)

        compute_ged_features(current_img, ged, ch, cw)
        ged_responses.append(ged)

        if l < levels - 1:
            temp_blur_h = np.zeros((ch, cw), dtype=np.float32)
            temp_blur_v = np.zeros((ch, cw), dtype=np.float32)
            gaussian_blur(current_img, temp_blur_v, temp_blur_h, ch, cw)

            next_img = np.zeros((ch // 2, cw // 2), dtype=np.float32)
            downsample(temp_blur_v, next_img, ch, cw)
            current_img = next_img

    cdef int out_channels = 6
    cdef float[:, :, ::1] output = np.zeros((h, w, out_channels), dtype=np.float32)

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
