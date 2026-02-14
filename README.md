# Distributed Quantum Gaussian Process Regression

This project implements distributed quantum Gaussian process regression using Riemannian ADMM optimization for real-world datasets and quantum-generated synthetic data.

## 📁 Project Structure

```
├── main.py                    # Main distributed QGPR with Riemannian ADMM
├── agent_riemannian.py       # Quantum agent with Riemannian optimization
├── riemannian_optimizer.py   # Riemannian optimization framework
├── real_world_datasets.py    # Real-world dataset loaders (SRTM elevation)
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# 2D quantum GP synthetic data with default settings
python main.py --input-dim 2 --n-dataset 1000 --n-agents 4

# SRTM elevation dataset (real-world 2D data)
python main.py --real-world-dataset srtm --srtm-region maharashtra \
               --dataset-max-samples 1000 --n-agents 4
```

### (TUNED) Real-World Dataset: SRTM Elevation 🌍

The SRTM (Shuttle Radar Topography Mission) elevation dataset provides real-world 2D spatial data from different geographic regions.

```bash
# Maharashtra, India
python main.py --real-world-dataset srtm --srtm-region maharashtra \
               --dataset-max-samples 1000 --dataset-normalize \
               --encoding chebyshev --kernel-type projected --num-layers 3 \
               --num-qubits 4 --outer-kernel matern \
               --rho 100 --L 100 --n-agents 4

# Great Lakes, North America
python main.py --real-world-dataset srtm --srtm-region great_lakes \
               --dataset-max-samples 1000 --dataset-normalize \
               --encoding chebyshev --kernel-type projected --num-layers 3 \
               --num-qubits 4 --outer-kernel matern \
               --rho 100 --L 100 --n-agents 4

# Oregon Coast, USA
python main.py --real-world-dataset srtm --srtm-region oregon_coast \
               --dataset-max-samples 1000 --dataset-normalize \
               --encoding chebyshev --kernel-type projected --num-layers 3 \
               --num-qubits 4 --outer-kernel matern \
               --rho 100 --L 100 --n-agents 4

# Washington Coast, USA
python main.py --real-world-dataset srtm --srtm-region washington_coast \
               --dataset-max-samples 1000 --dataset-normalize \
               --encoding chebyshev --kernel-type projected --num-layers 4 \
               --num-qubits 5 --outer-kernel matern \
               --rho 100 --L 100 --n-agents 4
```

### Synthetic Quantum GP Dataset (2D)

Generate data using quantum Gaussian processes for authentic quantum ML research.

```bash
# 2D Quantum GP data (default - no classical function)
python main.py --input-dim 2 --n-dataset 1000 \
               --encoding hubregtsen --kernel-type projected --num-layers 1 \
               --num-qubits 3 --outer-kernel matern \
               --rho 100 --L 100 --n-agents 4

# 2D with specific encoding circuit
python main.py --input-dim 2 --n-dataset 1000 \
               --encoding chebyshev --kernel-type projected --num-layers 1 \
               --num-qubits 3 --outer-kernel matern \
               --rho 100 --L 100 --n-agents 4

# Save dataset for later use
python main.py --input-dim 2 --n-dataset 1000 --save-dataset --dataset-name my_qgp_2d
```

## 🧪 Datasets

The implementation supports two main dataset types:

### 1. Synthetic Quantum GP Datasets (Default)
Generated using quantum Gaussian processes with parameterized quantum circuits for authentic quantum ML research. The data is truly quantum in nature, not from classical test functions.

**Properties:**
- Input dimensions: 1D to 6D supported
- Quantum-generated target values using fidelity or projected quantum kernels
- Multiple encoding circuits: Chebyshev, YZ-CX, Hubregtsen, Kyriienko, etc.
- Tunable noise levels for realistic scenarios

### 2. Real-World SRTM Elevation Dataset 🌍
Shuttle Radar Topography Mission (SRTM) elevation data providing real-world 2D spatial regression problems.

**Available Regions:**
- **Maharashtra, India**: Complex terrain with varying elevations
- **Great Lakes, North America**: Large-scale water and land features
- **Oregon Coast, USA**: Coastal topography with elevation gradients
- **Washington Coast, USA**: Diverse coastal and mountain terrain

**Properties:**
- **Input**: 2D coordinates (latitude, longitude)
- **Output**: Elevation measurements (meters)
- **Use case**: Spatial regression, environmental modeling
- **Features**: Real-world spatial correlations, measurement noise

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
[Your paper citation here]
```

## 📜 License

MIT License - see LICENSE file for details
