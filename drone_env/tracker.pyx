# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from drone_env.texture_features import compute_texture_hypercube

cdef class TextureTracker:
    cdef int x, y, w, h
    cdef int padding
    cdef float sigma
    cdef float lambda_val
    cdef float interp_factor
    cdef object H
    cdef object Y
    cdef object hann_window
    cdef int feature_h, feature_w
    cdef int target_h, target_w

    def __init__(self, padding=1.5, sigma=0.2, lambda_val=1e-4, interp_factor=0.02):
        self.padding = <int>(padding * 100)
        self.sigma = sigma
        self.lambda_val = lambda_val
        self.interp_factor = interp_factor

    def init(self, image, bbox):
        """
        Initialize the tracker with the first frame and bounding box.
        image: numpy array (H, W, 3) BGR or (H, W) Gray
        bbox: (x, y, w, h)
        """
        cdef int x, y, w, h
        x, y, w, h = [int(v) for v in bbox]
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        # Use attributes initialized in __init__
        padding_factor = float(self.padding) / 100.0

        # Crop size (target size + padding)
        cdef int crop_w = int(w * padding_factor)
        cdef int crop_h = int(h * padding_factor)

        # Ensure even size for easier FFT usually
        if crop_w % 2 != 0: crop_w += 1
        if crop_h % 2 != 0: crop_h += 1

        self.target_w = crop_w
        self.target_h = crop_h

        # Extract features to determine feature size
        features, _ = self._get_features(image, x, y, w, h, crop_w, crop_h)
        self.feature_h = features.shape[0]
        self.feature_w = features.shape[1]

        # Create Gaussian Target Y
        # Y is a 2D Gaussian peak centered at the middle of the feature map
        # Sigma relative to feature size
        cdef float output_sigma = np.sqrt(self.feature_w * self.feature_h) * self.sigma * 0.5 # Heuristic

        y_h, x_h = np.ogrid[:self.feature_h, :self.feature_w]
        center_y = self.feature_h // 2
        center_x = self.feature_w // 2
        dist_sq = (y_h - center_y)**2 + (x_h - center_x)**2
        y_response = np.exp(-0.5 * dist_sq / (output_sigma**2))

        # FFT of Y
        self.Y = np.fft.fft2(y_response)

        # Cosine Window (Hanning)
        hann_h = np.hanning(self.feature_h).reshape(-1, 1)
        hann_w = np.hanning(self.feature_w).reshape(1, -1)
        self.hann_window = (hann_h * hann_w).astype(np.float32)
        # Add channel dim for broadcasting
        self.hann_window = self.hann_window[:, :, np.newaxis]

        # Train initial filter
        X = np.fft.fft2(features * self.hann_window, axes=(0, 1))

        # DCF/MOSSE update
        # Numerator: Y * conj(X)
        # Denominator: X * conj(X) + lambda
        # For multi-channel: We sum denominator over channels for KCF linear kernel simplification?
        # Standard DCF: Min Sum_c || H_c * X_c - Y ||^2
        # Solution: H_c = (Y * conj(X_c)) / (Sum_k (X_k * conj(X_k)) + lambda)

        self.H = self._train(X, self.Y)
        return True

    def update(self, image):
        """
        Update the tracker with the new frame.
        Returns: success, (x, y, w, h)
        """
        # Extract features at previous position
        features, (crop_x, crop_y) = self._get_features(image, self.x, self.y, self.w, self.h, self.target_w, self.target_h)

        # FFT
        X = np.fft.fft2(features * self.hann_window, axes=(0, 1))

        # Compute Response
        # R = IFFT( Sum_c (H_c * X_c) )
        response_freq = np.sum(self.H * X, axis=2)
        response = np.real(np.fft.ifft2(response_freq))

        # Find peak
        # Note: The peak in correlation output is relative to the center (due to circular correlation/FFT shift)
        # Actually standard FFT correlation:
        # If we trained with Y centered at (h/2, w/2), then if the object didn't move, peak is at (h/2, w/2).

        row_max, col_max = np.unravel_index(np.argmax(response), response.shape)

        # Calculate displacement
        # Features are usually downsampled. Texture features returns original size features?
        # compute_texture_hypercube returns same size as input (or slightly smaller due to padding handling?)
        # Let's check compute_texture_hypercube again. It returns (h, w, 6) where h, w are input dims.
        # Yes.

        dy = (row_max - self.feature_h // 2)
        dx = (col_max - self.feature_w // 2)

        # Wrap around handling for FFT shifts (though if using centered gaussian, usually not needed if displacement small)
        if dy > self.feature_h // 2: dy -= self.feature_h
        if dy < -self.feature_h // 2: dy += self.feature_h
        if dx > self.feature_w // 2: dx -= self.feature_w
        if dx < -self.feature_w // 2: dx += self.feature_w

        self.x += int(dx)
        self.y += int(dy)

        # Limit to image bounds
        img_h = image.shape[0]
        img_w = image.shape[1]
        self.x = max(0, min(img_w - self.w, self.x))
        self.y = max(0, min(img_h - self.h, self.y))

        # Update Model
        new_features, _ = self._get_features(image, self.x, self.y, self.w, self.h, self.target_w, self.target_h)
        X_new = np.fft.fft2(new_features * self.hann_window, axes=(0, 1))
        H_new = self._train(X_new, self.Y)

        # Moving average
        self.H = (1.0 - self.interp_factor) * self.H + self.interp_factor * H_new

        return True, (self.x, self.y, self.w, self.h)

    def _train(self, X, Y):
        # H_c = (Y * conj(X_c)) / (Sum_k (X_k * conj(X_k)) + lambda)
        # Y has shape (H, W). X has shape (H, W, C).
        # We need to broadcast Y.

        Y_expanded = Y[:, :, np.newaxis]
        num = Y_expanded * np.conj(X)
        den = np.sum(X * np.conj(X), axis=2) + self.lambda_val

        # den has shape (H, W). Expand to (H, W, 1) or broadcast
        den_expanded = den[:, :, np.newaxis]

        return num / den_expanded

    def _get_features(self, image, x, y, w, h, target_w, target_h):
        # Extract patch
        center_x = x + w / 2
        center_y = y + h / 2

        x1 = int(center_x - target_w / 2)
        y1 = int(center_y - target_h / 2)
        x2 = x1 + target_w
        y2 = y1 + target_h

        # Pad if out of bounds
        im_h, im_w = image.shape[:2]

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - im_w)
        pad_bottom = max(0, y2 - im_h)

        cx1 = x1 + pad_left
        cy1 = y1 + pad_top
        cx2 = x2 - pad_right
        cy2 = y2 - pad_bottom

        if cx2 <= cx1 or cy2 <= cy1:
             # Fallback if collapsed
             patch = np.zeros((target_h, target_w), dtype=np.float32)
        else:
             if image.ndim == 3:
                 crop = image[cy1:cy2, cx1:cx2, :]
                 # Convert to gray
                 crop = 0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
             else:
                 crop = image[cy1:cy2, cx1:cx2]

             patch = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

        # Resize if necessary (though we calculated coordinates to match target size, int rounding might be off)
        if patch.shape[0] != target_h or patch.shape[1] != target_w:
             import cv2
             patch = cv2.resize(patch, (target_w, target_h))

        # Ensure float32 C-contiguous
        patch_f32 = np.ascontiguousarray(patch, dtype=np.float32)

        # Compute Texture Features
        # compute_texture_hypercube returns (H, W, 6)
        features = compute_texture_hypercube(patch_f32)

        return features, (x1, y1)
