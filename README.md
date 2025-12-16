<img width="1475" height="850" alt="Screenshot from 2025-08-10 12-48-40" src="https://github.com/user-attachments/assets/47b1f314-c5a3-4acc-ab10-66812e4b3dc6" />

This repo provides a patch to add a forward-looking RGB camera pitched up at 30 degrees relative to the Iris body in PX4 Gazebo Classic simulations (compatible with v1.14.0).
### Prequesits 
- nvidia drivers
- nvidia docker
- internet connection


## How to Use in Docker Build
- Clone this repo
- build using the dockerfile
- run using the command below
```
git clone https://github.com/benmatok/px4-iris-camera-patch.git
cd px4-iris-camera-patch
sudo docker build --build-arg CACHE_BREAKER=$RANDOM -t px4-gazebo-setup .
xhost +local:
sudo docker run -it --rm --privileged --gpus=all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/dri:/dev/dri -v /dev/shm:/dev/shm --network host -v $HOME/.ssh:/home/px4user/.ssh:rw -v /dev:/dev px4-gazebo-setup
inside-container> cd ~/PX4-Autopilot
inside-container> HEADLESS=1 make px4_sitl gazebo-classic_iris__baylands
inside-container> HEADLESS=1 make px4_sitl gazebo-classic_iris__ksql_airport
```
## in a second bash
```
sudo docker ps  # Note the CONTAINER ID of your running px4 container
sudo docker exec -it <CONTAINER_ID> bash
inside-container> python3 /src/px4-iris-camera-patch/main.py
```

## Running Drone RL Training

To run the custom Drone RL training:

```bash
python3 train_drone.py
```

You can customize the configuration by editing `configs/drone.yaml` or passing a different config file:
```bash
python3 train_drone.py --config configs/my_custom_config.yaml
```

### Architecture and Scaling

This project utilizes a dual-backend architecture to support both high-performance scaling on GPUs and efficient development on CPUs.

#### 1. WarpDrive Integration (GPU Scaling)
For massive scaling, the environment is designed to integrate with **Salesforce WarpDrive**.
- **End-to-End GPU Simulation**: The entire environment logic (step, reset, physics) is written in CUDA C (`drone_env/drone.py`), allowing thousands of agents to be simulated in parallel on a single GPU without CPU-GPU data transfer overhead.
- **Zero-Copy**: Observations and rewards stay on the GPU memory, directly accessible by the PPO learner.
- **Scalability**: Enables training with 2000+ agents simultaneously, significantly accelerating RL convergence.

*Requirements:* To use this mode, ensure `pycuda` and `rl_warp_drive` are installed and a CUDA-capable GPU is available.

#### 2. Optimized Cython Backend (CPU Optimization)
When a GPU is unavailable or for debugging/unit-testing, the system falls back to a highly optimized CPU implementation.
- **Cython + OpenMP**: The `step` and `reset` functions are implemented in `drone_env/drone_cython.pyx`, compiled to native C++ code.
- **Vectorization (AVX/SIMD)**: Compiled with `-march=native -mavx2 -mfma -ffast-math` to leverage modern CPU vector instructions.
- **Multi-Threading**: Uses `prange` (OpenMP) to parallelize agent updates across all available CPU cores.
- **Performance**: Benchmarks show an **~11.3x speedup** compared to a standard vectorized NumPy implementation (0.24s vs 2.75s for 100 steps of 5000 agents).

### Performance Validation

To validate the optimized Cython backend, you can run the benchmark script:

```bash
python benchmark_cython.py
```

**Recent Benchmark Results (5000 agents, 100 steps):**
- **NumPy CPU Time:** ~3.2s - 4.6s
- **Cython Time:** ~0.10s - 0.17s
- **Speedup:** ~26x - 32x

The optimization using `sincos` for trigonometric calculations significantly improves the instruction throughput for the physics engine.

To verify correctness, `train_ae.py` confirms that the physics simulation produces valid data for learning, as evidenced by decreasing loss during training.

### Autoencoder Training

A separate script `train_ae.py` is provided to train the IMU Autoencoder independently using data generated from large-scale simulations. This script uses the Cython-optimized CPU environment to generate diverse flight trajectories (using a proportional controller) and trains the `Autoencoder1D` model using KFAC optimization.

#### Usage

To train the autoencoder from scratch:
```bash
python train_ae.py --agents 2000 --episodes 1000
```

To resume training from a checkpoint:
```bash
python train_ae.py --agents 2000 --episodes 1000 --load ae_model.pth
```

### Installation & Compilation

To run the project (especially the CPU-optimized Cython backend) on a clean installation, you must compile the C++ extensions.

#### Prerequisites
- **Python 3.x**
- **C++ Compiler**: `g++` or `clang++` with OpenMP support.
- **Python Packages**: `numpy`, `Cython`, `setuptools`.

#### Compilation Command
Run the following command in the root directory to build the Cython extension in place:

```bash
python setup.py build_ext --inplace
```

This compiles `drone_env/drone_cython.pyx` into a shared object (`.so`) file that the Python scripts import automatically.

#### Arguments
- `--agents`: Number of parallel drones to simulate (default: 2000).
- `--episodes`: Number of episodes to train (default: 1000).
- `--load`: Path to a checkpoint file (e.g., `ae_model.pth`) to resume training.

The script saves the trained model to `ae_model.pth` and a loss plot to `ae_training_loss.png` after each episode.

### Training Visualization

The training script automatically generates visualizations of the agent's performance.
After training, check the `visualizations/` directory for:
- `reward_plot.png`: A plot of mean rewards over iterations.
- `training_evolution.gif`: An animation showing the evolution of the drone's trajectory (Top-down and Side views) throughout the training process.

![Reward Plot](visualizations_mock/reward_plot.png)
![Training Evolution](visualizations_mock/training_evolution.gif)
