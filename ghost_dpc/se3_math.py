import numpy as np
import math

def so3_exp(omega):
    """
    Computes the exponential map from so(3) to SO(3) using Rodrigues' formula.
    Args:
        omega: A 3-element array representing the rotation vector (angle * axis).
    Returns:
        A 3x3 rotation matrix.
    """
    theta_sq = np.dot(omega, omega)
    theta = math.sqrt(theta_sq)

    K = np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0]
    ], dtype=np.float32)

    if theta < 1e-6:
        # Taylor expansion for small angles
        # R = I + K + K^2 / 2
        return np.eye(3, dtype=np.float32) + K + 0.5 * np.matmul(K, K)
    else:
        # Rodrigues' formula
        # R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2
        inv_theta = 1.0 / theta
        a = math.sin(theta) * inv_theta
        b = (1.0 - math.cos(theta)) * (inv_theta * inv_theta)
        return np.eye(3, dtype=np.float32) + a * K + b * np.matmul(K, K)

def rpy_to_matrix(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
    Convention: Z-Y-X intrinsic rotations (Yaw -> Pitch -> Roll).
    Args:
        roll: Rotation around X-axis in radians.
        pitch: Rotation around Y-axis in radians.
        yaw: Rotation around Z-axis in radians.
    Returns:
        A 3x3 rotation matrix.
    """
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # Rz * Ry * Rx
    # Rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    # Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    # Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]

    # Combined (row-major):
    # R11 = cy*cp
    # R12 = cy*sp*sr - sy*cr
    # R13 = cy*sp*cr + sy*sr
    # R21 = sy*cp
    # R22 = sy*sp*sr + cy*cr
    # R23 = sy*sp*cr - cy*sr
    # R31 = -sp
    # R32 = cp*sr
    # R33 = cp*cr

    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr

    return R

def matrix_to_rpy(R):
    """
    Converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    Convention: Z-Y-X intrinsic rotations.
    Args:
        R: A 3x3 rotation matrix.
    Returns:
        A tuple (roll, pitch, yaw).
    """
    # R31 = -sin(pitch)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch is +/- 90 degrees
        # R31 = -1 => pitch = 90
        # R31 = 1 => pitch = -90
        roll = 0.0
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(-R[0, 1], R[1, 1]) # Derived from R12, R22 for singular case

    return roll, pitch, yaw
