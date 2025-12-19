# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt, atan2, exp, fabs, pow, M_PI, cbrt
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

# --- 2D Tile Processing Logic ---

cdef inline void process_tile_features(
    int r_start, int r_end, int c_start, int c_end,
    int h, int w,
    float[:, ::1] img,
    float[:, ::1] orient_out,
    float[:, ::1] coher_out,
    float[:, ::1] hess_out,
    float[:, ::1] ged_out,
    float hess_scale_norm,
    float *Ix_buf, float *Iy_buf,
    int buf_stride # Width of buffer (TILE_W + 2)
) noexcept nogil:

    # Ix_buf and Iy_buf are scratchpads of size (TILE_H + 2) * (TILE_W + 2)
    # They hold gradients for the tile PLUS 1-pixel halo.
    # Buffer coordinate (0, 0) corresponds to global (r_start-1, c_start-1)

    cdef int global_r, global_c
    cdef int br, bc # Buffer row/col indices
    cdef float val_x, val_y

    # 1. Compute Gradients into Local Buffers
    # We need to cover range [r_start-1, r_end] x [c_start-1, c_end]
    # This matches buffer size (h_tile+2) x (w_tile+2)

    cdef int h_tile_plus_halo = r_end - r_start + 2
    cdef int w_tile_plus_halo = c_end - c_start + 2

    for br in range(h_tile_plus_halo):
        global_r = r_start - 1 + br
        for bc in range(w_tile_plus_halo):
            global_c = c_start - 1 + bc

            if global_r >= 0 and global_r < h and global_c >= 0 and global_c < w:
                # Calc Gradients at global_r, global_c
                # Gradient X
                if global_c > 0 and global_c < w - 1:
                    val_x = (img[global_r, global_c + 1] - img[global_r, global_c - 1]) * 0.5
                else:
                    val_x = 0.0

                # Gradient Y
                if global_r > 0 and global_r < h - 1:
                    val_y = (img[global_r + 1, global_c] - img[global_r - 1, global_c]) * 0.5
                else:
                    val_y = 0.0
            else:
                val_x = 0.0
                val_y = 0.0

            Ix_buf[br * buf_stride + bc] = val_x
            Iy_buf[br * buf_stride + bc] = val_y

    # 2. Compute Features using Local Buffers
    # Iterate over valid output pixels [r_start, r_end) x [c_start, c_end)

    cdef int r, c
    cdef int i, j
    cdef float s_xx, s_yy, s_xy, val_ix, val_iy
    cdef float diff, sum_eigen, lambda1, lambda2
    cdef float i_xx, i_yy, i_xy, det
    cdef float sum_in, sum_out, cnt_in, cnt_out, val_img

    for r in range(r_start, r_end):
        if r >= h: break
        br = r - r_start + 1 # Buffer row center

        for c in range(c_start, c_end):
            if c >= w: break
            bc = c - c_start + 1 # Buffer col center

            # --- Structure Tensor ---
            s_xx = 0.0; s_yy = 0.0; s_xy = 0.0

            # 3x3 loop around (br, bc) in buffer
            for i in range(-1, 2):
                for j in range(-1, 2):
                    val_ix = Ix_buf[(br+i) * buf_stride + (bc+j)]
                    val_iy = Iy_buf[(br+i) * buf_stride + (bc+j)]
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
            # Ix_buf stores Ix. We need d/dx(Ix) etc.
            # Ixx ~ (Ix(x+1) - Ix(x-1)) / 2
            i_xx = (Ix_buf[br * buf_stride + (bc+1)] - Ix_buf[br * buf_stride + (bc-1)]) * 0.5
            i_yy = (Iy_buf[(br+1) * buf_stride + bc] - Iy_buf[(br-1) * buf_stride + bc]) * 0.5
            i_xy = (Ix_buf[(br+1) * buf_stride + bc] - Ix_buf[(br-1) * buf_stride + bc]) * 0.5

            det = i_xx * i_yy - i_xy * i_xy
            hess_out[r, c] = fabs(det) * hess_scale_norm

            # --- GED --- (Uses Global Image, access pattern is scattered but localized)
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

    # Output buffers
    cdef float[:, ::1] orient
    cdef float[:, ::1] coher
    cdef float[:, ::1] hess
    cdef float[:, ::1] ged

    cdef int l, ch, cw
    cdef int TILE_SIZE = 32 # Small tile to fit L1 cache (32x32x4x2 ~ 8KB)
    cdef float scale_factor

    cdef int num_strips_r, strip_r, r_start, r_end
    cdef int num_strips_c, strip_c, c_start, c_end

    # Scratchpads pointers
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

        # We perform 2D tiling.
        # Parallelize over ROW strips. Each thread handles a full row of tiles.
        # Inside the thread, we iterate over COL tiles.
        # We allocate ONE buffer per thread and reuse it.

        with nogil, parallel():
            # Allocate thread-local buffers
            # Max buffer size is (TILE_SIZE + 2) x (TILE_SIZE + 2)
            buf_h = TILE_SIZE + 2
            buf_w = TILE_SIZE + 2
            buf_size = buf_h * buf_w

            Ix_buf = <float *> malloc(buf_size * sizeof(float))
            Iy_buf = <float *> malloc(buf_size * sizeof(float))

            if Ix_buf != NULL and Iy_buf != NULL:

                for strip_r in prange(num_strips_r, schedule='static'):
                    r_start = strip_r * TILE_SIZE
                    r_end = r_start + TILE_SIZE
                    if r_end > ch: r_end = ch

                    for strip_c in range(num_strips_c):
                        c_start = strip_c * TILE_SIZE
                        c_end = c_start + TILE_SIZE
                        if c_end > cw: c_end = cw

                        process_tile_features(
                            r_start, r_end, c_start, c_end,
                            ch, cw,
                            current_img,
                            orient, coher, hess, ged,
                            pow(scale_factor, 4),
                            Ix_buf, Iy_buf, buf_w
                        )

            # Cleanup
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
