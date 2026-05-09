# Distributed Quantum Gaussian Process Regression

This project implements distributed quantum Gaussian process regression using ADMM optimization with state-of-the-art test functions and multi-dimensional support.

## 📁 Project Structure

```
├── main.py                         # Main distributed QGPR with ADMM consensus
├── agent_riemannian.py             # Quantum agent with Riemannian optimization
├── riemannian_optimizer.py         # Manifold-aware optimization (torus geometry)
├── real_world_datasets.py          # Real-world dataset loaders (SRTM, SST, kin40k, robot_push)
├── requirements.txt                # Project dependencies
├── README.md                       # Documentation and examples
└── srtm_data/                      # Sample SRTM elevation datasets
    ├── N17E073.hgt                 # Latitude 17°N, Longitude 73°E (Western Ghats - Maharashtra)
    ├── N43W080.hgt                 # Latitude 43°N, Longitude 80°W (Great Lakes region)
    ├── N45W123.hgt                 # Latitude 45°N, Longitude 123°W (Oregon Coast)
    └── N47W124.hgt                 # Latitude 47°N, Longitude 124°W (Washington Coast)
```

## 🚀 Quick Start

### Default Run Command
```bash
python main.py --real-world-dataset srtm --srtm-region maharashtra --dataset-max-samples 1000 --dataset-normalize --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1
```

### Basic Usage
```bash
# 1D function with default settings
python main.py --input-dim 1 --n-train 60 --n-agents 4

# 2D Goldstein-Price function
python main.py --input-dim 2 --n-train 100 --n-agents 4 --max-iter 50

# 3D Hartmann function with more qubits
python main.py --input-dim 3 --n-train 150 --num-qubits 6 --n-agents 4
```

### (TUNED) Real-World Dataset Usage 🌍
```bash
# SRTM Elevation Dataset (2D) 
--use-srtm-preprocessed for approach comparison

# Maharashtra
python main.py --real-world-dataset srtm --srtm-region maharashtra --dataset-max-samples 1000 --dataset-normalize --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

python experiment_runner.py --num-experiments 30 --real-world-dataset srtm --srtm-region maharashtra --dataset-max-samples 1000 --dataset-normalize --encoding chebyshev --kernel-type projected --num-layers 3 --use-riemannian --num-qubits 4 --outer-kernel matern --rho 100 --L 100

# Great Lakes
python main.py --real-world-dataset srtm --srtm-region great_lakes --dataset-max-samples 1000 --dataset-normalize --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

python experiment_runner.py --num-experiments 30 --real-world-dataset srtm --srtm-region great_lakes --dataset-max-samples 1000 --dataset-normalize --encoding chebyshev --kernel-type projected --num-layers 3 --use-riemannian --num-qubits 4 --outer-kernel matern --rho 100 --L 100

# Oregon Coast 
python main.py --real-world-dataset srtm --srtm-region oregon_coast --dataset-max-samples 1000 --dataset-normalize --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

python experiment_runner.py --num-experiments 30 --real-world-dataset srtm --srtm-region oregon_coast --dataset-max-samples 1000 --dataset-normalize --encoding chebyshev --kernel-type projected --num-layers 3 --use-riemannian --num-qubits 4 --outer-kernel matern --rho 100 --L 100

# Washington Coast 
python main.py --real-world-dataset srtm --srtm-region washington_coast --dataset-max-samples 1000 --dataset-normalize --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

python experiment_runner.py --num-experiments 30 --real-world-dataset srtm --srtm-region washington_coast --dataset-max-samples 1000 --dataset-normalize --encoding chebyshev --kernel-type projected --num-layers 4 --use-riemannian --num-qubits 5 --outer-kernel matern --rho 100 --L 100

# Sea Surface Temperature (2D)
python main.py --real-world-dataset sst --dataset-max-samples 1000 --dataset-normalize --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

python experiment_runner.py --num-experiments 30 --real-world-dataset sst --dataset-max-samples 1000 --encoding hubregtsen --kernel-type projected --num-layers 1 --use-riemannian --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --dataset-normalize

# Robot Push Dataset (3D)
python main.py --real-world-dataset robot_push --dataset-max-samples 1000 --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

python experiment_runner.py --num-experiments 30 --real-world-dataset robot_push --dataset-max-samples 1000 --encoding hubregtsen --kernel-type projected --num-layers 1 --use-riemannian --num-qubits 3 --outer-kernel matern --rho 100 --L 100

# Save dataset for later use
python main.py --real-world-dataset sst --dataset-max-samples 1000 --dataset-normalize --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1 --dataset-only --save-dataset
```

### Synthetic Dataset Usage
```bash
# 2D Goldstein-Price function (normalized)
python main.py --classical-dataset --input-dim 2 --n-dataset 1000 --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

python experiment_runner.py --num-experiments 30 --classical-dataset --n-dataset 1000 --input-dim 2 --encoding hubregtsen --kernel-type projected --num-layers 1 --use-riemannian --num-qubits 3 --outer-kernel matern --rho 100 --L 100

# 3D Hartmann function
python main.py --classical-dataset --input-dim 3 --n-dataset 1000 --encoding hubregtsen --kernel-type projected --num-layers 1 --num-qubits 3 --outer-kernel matern --rho 100 --L 100 --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

python experiment_runner.py --num-experiments 30 --classical-dataset --n-dataset 1000 --input-dim 3 --encoding chebyshev --kernel-type projected --num-layers 1 --use-riemannian --num-qubits 3 --outer-kernel matern --rho 100 --L 100
```

### Advanced Configuration
```bash
# Riemannian gradients with parameter shift
python main.py --differentiation parameter_shift

# Sequential data partitioning
python main.py --input-dim 2 --partition sequential --test-split 0.3

# Custom ADMM parameters
python main.py --max-iter 100 --tolerance 1e-8 --noise-std 0.05

# Real-world dataset with advanced settings
python main.py --real-world-dataset kin40k --dataset-max-samples 500 \
               --encoding hubregtsen --kernel-type projected --num-layers 1 \
               --num-qubits 3 --outer-kernel matern --rho 100 --L 100 \
               --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1

# High-dimensional experiment with robot kinematics
python main.py --real-world-dataset kin40k --dataset-max-samples 800 \
               --encoding hubregtsen --kernel-type projected --num-layers 1 \
               --num-qubits 3 --outer-kernel matern --rho 100 --L 100 \
               --n-agents 4 --prediction-method gPoE --cv-patience 5 --noise-lower-bound 0.1
```

### Quantum Hardware Jobs (TRUE QUANTUM ADVANTAGE)
```bash
# Option 1: Use launcher
python run_quantum_hardware.py

# Option 2: Direct execution
cd quantum_hardware
python main_hardware_job.py
```

## 🧪 Test Functions & Datasets

The implementation supports multiple dataset types:

### Synthetic Quantum Datasets (Default)
Generated using quantum Gaussian processes for authentic quantum ML research.

### Classical Test Functions  
State-of-the-art optimization benchmarks:

#### 1D Functions
- **Complex Synthetic**: `f(x) = 5x²sin(12x) + (x³-0.5)sin(3x-0.5) + 4cos(2x)`
- Domain: [0, 1]
- Features: Multiple local optima, complex oscillatory behavior

#### 2D Functions  
- **Goldstein-Price** (normalized): Classic optimization benchmark
- Domain: [-2, 2] × [-2, 2]
- Features: Global minimum, multiple local minima, logarithmic scaling

#### 3D Functions
- **Hartmann 3D**: Standard high-dimensional benchmark
- Domain: [0, 1]³
- Features: 4 local minima, exponential terms, smooth gradients

### Real-World Datasets 🌍
Realistic datasets commonly used in GP research:

#### Sea Surface Temperature (SST) - 2D
- **Input**: Latitude, Longitude coordinates
- **Output**: Temperature measurements
- **Use case**: Spatial regression, environmental modeling
- **Command**: `--real-world-dataset sst`

#### Robot Push Dataset - 3D
- **Input**: Object position (x, y) + push angle
- **Output**: Resulting displacement
- **Use case**: Robotics, manipulation planning
- **Command**: `--real-world-dataset robot_push`

#### Robot Kinematics (kin40k) - 8D
- **Input**: 8 joint angles
- **Output**: End-effector position
- **Use case**: High-dimensional regression, robotics
- **Command**: `--real-world-dataset kin40k`

### Data Splitting Options
- **Random**: Shuffled random partitioning among agents
- **Sequential**: Ordered partitioning (useful for spatial data)
- **Regional**: Spatial partitioning for geographical data
- **Train/Test Split**: sklearn-style splitting with reproducible seeds

## 💡 Key Features

| Feature | Description | Benefit |
|---------|------------|---------|
| **Riemannian Optimization** | Treats quantum circuit parameters as points on a torus manifold | Better convergence for periodic rotation parameters |
| **Distributed ADMM** | Consensus-based parameter optimization across agents | Scalable quantum GP with privacy preservation |
| **2D Focus** | Optimized for 2D spatial problems | Visualization and interpretability |
| **Real SRTM Data** | Actual elevation measurements | Realistic GP benchmarks |
| **Quantum Kernels** | Fidelity & Projected kernels with multiple encodings | Flexible quantum feature spaces |
| **Cross-Validation** | NLPD-based model selection | Robust hyperparameter tuning |

## ⚙️ Configuration Options

### Key Parameters

- `--n-agents`: Number of distributed agents (default: 4)
- `--num-qubits`: Quantum circuit qubits (default: 4)
- `--num-layers`: Encoding circuit layers (default: 2)
- `--encoding`: Circuit type - `chebyshev`, `hubregtsen`, `yz_cx`, `kyriienko`, etc.
- `--kernel-type`: `fidelity` or `projected`
- `--outer-kernel`: For projected kernels - `gaussian`, `matern`, `expsinesquared`, etc.
- `--rho`: ADMM penalty parameter (default: 100.0)
- `--L`: Lipschitz constant (default: 100.0)

### Riemannian Optimization

- `--riemannian-method`: `gradient_descent`, `momentum`, or `conjugate_gradient`
- `--riemannian-lr`: Learning rate (default: 0.015)
- `--riemannian-beta`: Momentum/CG parameter (default: 0.9)
- `--gradient-clip-norm`: Gradient clipping (default: 1.0)
- `--max-step-size`: Maximum step size (default: 0.1)

### Data Options

- `--partition`: Data split method - `regional`, `random`, or `sequential`
- `--test-split`: Test set ratio (default: 0.1)
- `--noise-std`: Observation noise (default: 0.1)
- `--dataset-normalize`: Normalize features and targets

## 📊 Output

The code generates:
- Convergence plots for ADMM iterations
- Prediction vs ground truth visualizations
- Cross-validation NLPD scores
- Agent-wise performance metrics
- Kernel matrix condition numbers

## 🔬 Research Use

This implementation is designed for research in:
- Quantum machine learning
- Distributed Gaussian processes
- Riemannian optimization for quantum circuits
- Spatial regression with quantum methods
- ADMM consensus algorithms

## 📝 Citation

If you use this code in your research, please cite:
```
@inproceedings{gandhi2026distributed,
  title={Distributed Quantum Gaussian Processes for Multi-Agent Systems},
  author={Gandhi, Meet and Kontoudis, George P},
  booktitle={International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year={2026}
}
```
