# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt, atan2, exp, fabs, pow, M_PI
from libc.stdlib cimport malloc, free

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

# --- Block Processing Logic ---

cdef void compute_strip_features(
    int r_start, int r_end, int h, int w,
    float[:, ::1] img,
    float[:, ::1] orient_out,
    float[:, ::1] coher_out,
    float[:, ::1] hess_out,
    float[:, ::1] ged_out,
    float hess_scale_norm
) noexcept nogil:

    # Define strip boundaries with halo
    # We need neighbors for ST/Hess.
    # ST/Hess needs Ix, Iy at (r, c).
    # ST at (r,c) needs 3x3 window of Ix, Iy. So Ix, Iy needed at r-1 to r+1.
    # Ix at r' needs Img at r'-1 to r'+1.
    # So to compute ST at r, we need Ix at r-1..r+1.
    # To compute Ix at r-1, we need Img at r-2..r.
    # To compute Ix at r+1, we need Img at r..r+2.
    # Total Img dependency: r-2 to r+2.

    cdef int h_strip = r_end - r_start
    # Halo: we need gradients for r_start-1 to r_end+1 (size h_strip + 2)
    # So we allocate local buffers for Ix, Iy of size (h_strip + 2) x w
    # We map row r_local in buffer to global row (r_start - 1 + r_local)

    cdef int buf_h = h_strip + 2
    cdef int buf_size = buf_h * w

    # Allocate scratchpads
    cdef float *Ix_buf = <float *> malloc(buf_size * sizeof(float))
    cdef float *Iy_buf = <float *> malloc(buf_size * sizeof(float))

    if Ix_buf == NULL or Iy_buf == NULL:
        # Allocation failed, abort strip (should handle error better but nogil)
        if Ix_buf != NULL: free(Ix_buf)
        if Iy_buf != NULL: free(Iy_buf)
        return

    cdef int br, c
    cdef int global_r
    cdef float val_x, val_y

    # 1. Compute Gradients into Buffer
    # Buffer row 0 corresponds to global row r_start - 1
    # Buffer row k corresponds to global row r_start - 1 + k
    # We need to compute gradients for global rows r_start-1 to r_end (inclusive? wait)
    # We need gradients for r_start-1 ... r_end (inclusive).
    # Range of rows where we compute features: r_start to r_end-1.
    # Neighbors needed: (r_start-1) to (r_end-1+1) = r_end.
    # So we need gradients for global rows: [r_start-1, r_end].
    # Total rows = (r_end) - (r_start-1) + 1 = r_end - r_start + 2 = h_strip + 2. Correct.

    for br in range(buf_h):
        global_r = r_start - 1 + br
        for c in range(w):
            # Gradient X
            if global_r >= 0 and global_r < h:
                if c > 0 and c < w - 1:
                    val_x = (img[global_r, c + 1] - img[global_r, c - 1]) * 0.5
                else:
                    val_x = 0.0

                # Gradient Y
                if global_r > 0 and global_r < h - 1:
                    val_y = (img[global_r + 1, c] - img[global_r - 1, c]) * 0.5
                else:
                    val_y = 0.0
            else:
                val_x = 0.0
                val_y = 0.0

            Ix_buf[br * w + c] = val_x
            Iy_buf[br * w + c] = val_y

    # 2. Compute Features (ST, Hess) using Buffer
    # We iterate global rows r from r_start to r_end-1
    # Local buffer index for global r is: r - (r_start - 1) = r - r_start + 1.
    # Let local_r = r - r_start + 1.
    # Neighbors: local_r - 1 to local_r + 1.

    cdef int i, j
    cdef float s_xx, s_yy, s_xy, val_ix, val_iy
    cdef float diff, sum_eigen, lambda1, lambda2
    cdef float i_xx, i_yy, i_xy, det
    cdef int r

    # 3. Compute GED (Needs Img) - independent of Gradients
    cdef float sum_in, sum_out, cnt_in, cnt_out, val_img

    for r in range(r_start, r_end):
        if r >= h: break # Safety

        br = r - r_start + 1 # Center in buffer

        for c in range(w):
            # --- Structure Tensor ---
            s_xx = 0.0; s_yy = 0.0; s_xy = 0.0

            for i in range(-1, 2):
                for j in range(-1, 2):
                    # Check boundary for columns, rows are safe in buffer (br-1 to br+1 are valid)
                    if c+j >= 0 and c+j < w:
                        val_ix = Ix_buf[(br+i) * w + (c+j)]
                        val_iy = Iy_buf[(br+i) * w + (c+j)]
                        s_xx += val_ix * val_ix
                        s_yy += val_iy * val_iy
                        s_xy += val_ix * val_iy

            diff = sqrt((s_xx - s_yy)**2 + 4 * s_xy * s_xy)
            sum_eigen = s_xx + s_yy
            lambda1 = (sum_eigen + diff) * 0.5
            lambda2 = (sum_eigen - diff) * 0.5

            if sum_eigen > 1e-6:
                coher_out[r, c] = (lambda1 - lambda2) / sum_eigen
            else:
                coher_out[r, c] = 0.0
            orient_out[r, c] = 0.5 * atan2(2 * s_xy, s_xx - s_yy)

            # --- Hessian ---
            # Ix_xx at (r,c) approx (Ix(r,c+1) - Ix(r,c-1))/2
            # Needs Ix at same row.
            # Ix_yy at (r,c) approx (Iy(r+1,c) - Iy(r-1,c))/2
            # Needs Iy at +/- 1 row.
            # Ix_xy at (r,c) approx (Ix(r+1,c) - Ix(r-1,c))/2

            if c > 0 and c < w - 1:
                i_xx = (Ix_buf[br * w + (c+1)] - Ix_buf[br * w + (c-1)]) * 0.5
            else:
                i_xx = 0.0

            i_yy = (Iy_buf[(br+1) * w + c] - Iy_buf[(br-1) * w + c]) * 0.5
            i_xy = (Ix_buf[(br+1) * w + c] - Ix_buf[(br-1) * w + c]) * 0.5

            det = i_xx * i_yy - i_xy * i_xy
            hess_out[r, c] = fabs(det) * hess_scale_norm

            # --- GED ---
            # Uses original img, large window 7x7
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

    free(Ix_buf)
    free(Iy_buf)

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

    # Allocations for features (Output)
    cdef float[:, ::1] orient
    cdef float[:, ::1] coher
    cdef float[:, ::1] hess
    cdef float[:, ::1] ged

    cdef int l, ch, cw, strip_idx, num_strips, r_start, r_end
    cdef int STRIP_HEIGHT = 64
    cdef float scale_factor

    for l in range(levels):
        ch = current_img.shape[0]
        cw = current_img.shape[1]

        orient = np.zeros((ch, cw), dtype=np.float32)
        coher = np.zeros((ch, cw), dtype=np.float32)
        hess = np.zeros((ch, cw), dtype=np.float32)
        ged = np.zeros((ch, cw), dtype=np.float32)

        scale_factor = pow(2.0, l)

        # Parallel Strip Processing
        num_strips = (ch + STRIP_HEIGHT - 1) // STRIP_HEIGHT

        for strip_idx in prange(num_strips, nogil=True, schedule='dynamic'):
            r_start = strip_idx * STRIP_HEIGHT
            r_end = r_start + STRIP_HEIGHT
            if r_end > ch: r_end = ch

            compute_strip_features(
                r_start, r_end, ch, cw,
                current_img,
                orient, coher, hess, ged,
                pow(scale_factor, 4)
            )

        st_orientations.append(orient)
        st_coherences.append(coher)
        hessian_responses.append(hess)
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
