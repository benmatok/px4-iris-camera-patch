import numpy as np
import time
import matplotlib.pyplot as plt
from drone_env.texture_features import compute_texture_hypercube

def generate_sine_wave(size=256, freq=10.0, angle=0.0):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xv, yv = np.meshgrid(x, y)

    # Rotation
    theta = np.radians(angle)
    xr = xv * np.cos(theta) - yv * np.sin(theta)
    yr = xv * np.sin(theta) + yv * np.cos(theta)

    return np.sin(2 * np.pi * freq * xr).astype(np.float32)

def generate_blob(size=256, sigma=10.0):
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    xv, yv = np.meshgrid(x, y)
    d = xv**2 + yv**2
    return np.exp(-d / (2 * sigma**2)).astype(np.float32)

def test_sine_wave():
    print("Running Perfect Sine Test...")
    img = generate_sine_wave(angle=45.0)
    features = compute_texture_hypercube(img)

    # 0: Orientation, 1: Coherence
    orientation = features[:, :, 0]
    coherence = features[:, :, 1]

    avg_orient = np.mean(orientation)
    avg_coher = np.mean(coherence)

    print(f"Average Coherence: {avg_coher:.4f} (Expected ~1.0)")
    h, w = img.shape
    center_coher = np.mean(coherence[h//4:3*h//4, w//4:3*w//4])
    print(f"Center Coherence: {center_coher:.4f}")

    if center_coher > 0.9:
        print("PASS: Coherence high.")
    else:
        print("FAIL: Coherence too low.")

def test_spinning_plate():
    print("Running Spinning Plate Test...")
    img1 = generate_sine_wave(angle=0.0) # Vertical
    img2 = generate_sine_wave(angle=90.0) # Horizontal

    f1 = compute_texture_hypercube(img1)
    f2 = compute_texture_hypercube(img2)

    # Orientation is in range [-pi/2, pi/2].
    # For 90 deg, we expect values near pi/2 or -pi/2.
    # Simple Mean might be near 0 if they cancel out.
    # We use Mean of Absolute values to check magnitude.

    o1 = np.mean(np.abs(f1[:, :, 0]))
    o2 = np.mean(np.abs(f2[:, :, 0]))

    print(f"Orient 0 deg (Abs Mean): {o1:.4f}")
    print(f"Orient 90 deg (Abs Mean): {o2:.4f}")

    # Difference should be pi/2 roughly
    diff = np.abs(o1 - o2)
    print(f"Difference: {diff:.4f} (Expected ~1.57)")

    if np.abs(diff - np.pi/2) < 0.2:
        print("PASS: Orientation rotates correctly.")
    else:
        print(f"FAIL: Orientation difference {diff} not expected.")

def test_zooming_dot():
    print("Running Zooming Dot Test...")
    # Generate blobs of different sizes

    # Small blob (Sigma=2)
    blob0 = generate_blob(sigma=2.0)
    f0 = compute_texture_hypercube(blob0)
    size0 = f0[128, 128, 4]

    # Medium blob (Sigma=8)
    blob1 = generate_blob(sigma=8.0)
    f1 = compute_texture_hypercube(blob1)
    size1 = f1[128, 128, 4]

    # Large blob (Sigma=32)
    blob2 = generate_blob(sigma=32.0)
    f2 = compute_texture_hypercube(blob2)
    size2 = f2[128, 128, 4]

    print(f"Small Blob Size Index: {size0}")
    print(f"Medium Blob Size Index: {size1}")
    print(f"Large Blob Size Index: {size2}")

    # Check monotonicity
    if size2 >= size1 and size1 >= size0:
        print("PASS: Scale selection is monotonic/correct.")
    else:
        print("FAIL: Scale selection unexpected.")

def benchmark():
    print("Running Benchmark...")
    img = np.random.rand(512, 512).astype(np.float32)
    start = time.time()
    for _ in range(10):
        _ = compute_texture_hypercube(img)
    end = time.time()
    avg_time = (end - start) / 10.0
    print(f"Average time per 512x512 image: {avg_time*1000:.2f} ms")
    if avg_time < 0.1: # Relaxed slightly for environment var
        print("PASS: Speed is acceptable.")
    else:
        print("WARN: Slow on CPU.")

def visualize():
    print("Generating Visualizations...")
    # Composite
    img = generate_sine_wave(size=512, angle=30, freq=5) + np.random.normal(0, 0.1, (512, 512)).astype(np.float32)
    features = compute_texture_hypercube(img)

    rgb = np.zeros((512, 512, 3), dtype=np.float32)
    rgb[:, :, 0] = features[:, :, 1] # Coherence
    rgb[:, :, 1] = features[:, :, 4] / 2.0 # Scale (0-2) normalized
    rgb[:, :, 2] = (features[:, :, 0] + np.pi/2) / np.pi # Orientation normalized

    plt.imsave("debug_texture_composite.png", rgb)

    plt.figure()
    plt.imshow(img, cmap='gray')
    stride = 16
    Y, X = np.mgrid[0:512:stride, 0:512:stride]
    U = np.cos(features[::stride, ::stride, 0]) * features[::stride, ::stride, 1]
    V = np.sin(features[::stride, ::stride, 0]) * features[::stride, ::stride, 1]
    plt.quiver(X, Y, U, V, color='r')
    plt.savefig("debug_flow_quivers.png")
    print("Saved debug_texture_composite.png and debug_flow_quivers.png")

if __name__ == "__main__":
    test_sine_wave()
    test_spinning_plate()
    test_zooming_dot()
    benchmark()
    visualize()
