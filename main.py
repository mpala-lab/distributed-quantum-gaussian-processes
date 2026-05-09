import numpy as np
import time
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from agent_riemannian import RiemannianAgent
from riemannian_optimizer import create_riemannian_framework, circular_mean
from real_world_datasets import load_real_world_dataset, get_dataset_info

def fast_riemannian_distance(x, y, period=np.pi):
    """
    Fast Riemannian distance calculation for quantum circuit parameters.
    
    This is an optimized version that avoids object creation overhead.
    Use this in performance-critical sections of your code.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    diff = x - y
    wrapped_diff = np.mod(diff + period * 0.5, period) - period * 0.5
    return np.linalg.norm(wrapped_diff)

from squlearn.encoding_circuit import (
    ChebyshevPQC, 
    YZ_CX_EncodingCircuit,
    HubregtsenEncodingCircuit,
    KyriienkoEncodingCircuit,
    MultiControlEncodingCircuit,
    LayeredEncodingCircuit,
    RandomEncodingCircuit,
    HighDimEncodingCircuit
)
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel
from squlearn.util import Executor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global variable to store quantum kernel in each process (for ProcessPoolExecutor)
_process_quantum_kernel = None

def create_quantum_kernel(num_qubits, num_features=1, num_layers=2, use_parameter_shift=True, encoding_type='yz_cx', kernel_type='fidelity', measurement='XYZ', outer_kernel='gaussian', outer_kernel_params=None, regularization=None):
    """
    Create a quantum kernel that can be reused across agents.
    This function can be called in each process to create the same kernel.
    
    Args:
        num_qubits (int): Number of qubits
        num_features (int): Number of features
        num_layers (int): Number of layers in the encoding circuit
        use_parameter_shift (bool): If True, use Qiskit executor (parameter shift rule).
                                   If False, use PennyLane executor (autodiff).
        encoding_type (str): Type of encoding circuit
        kernel_type (str): Type of quantum kernel - 'fidelity' or 'projected'
        measurement (str): Measurement operator for ProjectedQuantumKernel
        outer_kernel (str): Outer kernel type for ProjectedQuantumKernel. Options:
                          - 'gaussian': Gaussian RBF kernel (default)
                          - 'matern': Matern kernel from sklearn
                          - 'expsinesquared': ExpSineSquared kernel from sklearn
                          - 'rationalquadratic': RationalQuadratic kernel from sklearn
                          - 'dotproduct': DotProduct kernel from sklearn
                          - 'pairwisekernel': PairwiseKernel from sklearn
        outer_kernel_params (dict): Parameters for the outer kernel
        regularization (str): Regularization technique - 'thresholding', 'tikhonov', or None
    """
    # Create encoding circuit based on type
    if encoding_type == 'chebyshev':
        enc_circ = ChebyshevPQC(num_qubits, num_features=num_features, num_layers=num_layers)
        requires_clipping = True
        # print(f"Using Chebyshev encoding (contains arccos - may need input clipping)")
    elif encoding_type == 'yz_cx':
        enc_circ = YZ_CX_EncodingCircuit(num_qubits, num_features=num_features, num_layers=num_layers)
        requires_clipping = False
        # print(f"Using YZ_CX encoding (no arccos - safe for any input range)")
    elif encoding_type == 'hubregtsen':
        enc_circ = HubregtsenEncodingCircuit(num_qubits, num_features=num_features, num_layers=num_layers)
        requires_clipping = False
        print(f"Using Hubregtsen encoding (research-oriented design)")
    elif encoding_type == 'kyriienko':
        enc_circ = KyriienkoEncodingCircuit(num_qubits, num_features=num_features, num_layers=num_layers)
        requires_clipping = False
        # print(f"Using Kyriienko encoding (research-based design)")
    elif encoding_type == 'multi_control':
        enc_circ = MultiControlEncodingCircuit(num_qubits, num_features=num_features, num_layers=num_layers)
        requires_clipping = False
        # print(f"Using MultiControl encoding (complex entanglement patterns)")
    elif encoding_type == 'layered':
        enc_circ = LayeredEncodingCircuit(
            num_qubits, 
            num_features=num_features, 
            num_layers=num_layers,
            gates=['RX', 'RY', 'RZ']  # Customizable rotation gates
        )
        requires_clipping = False
        # print(f"Using Layered encoding (customizable gate structure)")
    elif encoding_type == 'random':
        enc_circ = RandomEncodingCircuit(num_qubits, num_features=num_features, num_layers=num_layers)
        requires_clipping = False
        # print(f"Using Random encoding (randomized circuit structure)")
    elif encoding_type == 'highdim':
        enc_circ = HighDimEncodingCircuit(num_qubits, num_features=num_features, num_layers=num_layers)
        requires_clipping = False
        # print(f"Using HighDim encoding (high-dimensional data encoding)")
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Supported: 'chebyshev', 'yz_cx', 'hubregtsen', 'kyriienko', 'multi_control', 'layered', 'random', 'highdim'")
    
    # Choose executor based on gradient method preference
    if use_parameter_shift:
        executor = Executor("statevector_simulator")  # Qiskit -> parameter shift rule
        print("Using Qiskit executor - Parameter Shift Rule will be used for gradients")
    else:
        executor = Executor("pennylane")  # PennyLane -> autodiff
        # print("Using PennyLane executor - Automatic differentiation will be used for gradients")
    
    # Create quantum kernel based on type
    if kernel_type == 'fidelity':
        q_kernel = FidelityKernel(
            encoding_circuit=enc_circ, 
            executor=executor,
            parameter_seed=0, 
            use_expectation=True,
            evaluate_duplicates="all" # Only "all" is supported for FidelityKernel dKdp in squlearn 0.2.0+ (as of October 2023)
        )
        # print(f"Using FidelityKernel (quantum state fidelity)")
    elif kernel_type == 'projected':
        # Set default outer kernel parameters if not provided
        if outer_kernel_params is None:
            outer_kernel_params = {}
        
        # ProjectedQuantumKernel: Do NOT unpack outer_kernel_params as kwargs
        # The outer kernel and its parameters are handled internally by the kernel
        q_kernel = ProjectedQuantumKernel(
            encoding_circuit=enc_circ,
            measurement=measurement,
            outer_kernel=outer_kernel,
            executor=executor,
            parameter_seed=0,
            regularization=regularization
        )
        print(f"Using ProjectedQuantumKernel with {measurement} measurement and {outer_kernel} outer kernel")
        if regularization is not None:
            print(f"Regularization: {regularization}")
        if outer_kernel_params:
            print(f"Outer kernel parameters: {outer_kernel_params}")
            
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Supported: 'fidelity', 'projected'")
    
    return q_kernel

def get_or_create_process_kernel(num_qubits, num_features=1, num_layers=2, use_parameter_shift=True, encoding_type='yz_cx', kernel_type='fidelity', measurement='XYZ', outer_kernel='gaussian', outer_kernel_params=None, regularization=None):
    """
    Get the quantum kernel for this process, creating it only once per process.
    This reuses the kernel across all ADMM iterations within the same process.
    """
    global _process_quantum_kernel
    if _process_quantum_kernel is None:
        # print(f"Process {os.getpid()} - Initializing quantum kernel once per process...")
        start_time = time.time()
        _process_quantum_kernel = create_quantum_kernel(num_qubits, num_features, num_layers, use_parameter_shift, encoding_type, kernel_type, measurement, outer_kernel, outer_kernel_params, regularization)
        init_time = time.time() - start_time
        # print(f"Process {os.getpid()} - Quantum kernel initialization took: {init_time:.4f}s")
    return _process_quantum_kernel

def generate_quantum_gp_data(num_samples, input_dim, num_qubits, num_layers=2, 
                           data_range=(-2.0, 2.0), noise_std=0.1, use_parameter_shift=True,
                           kernel_params=None, encoding_type='yz_cx', kernel_type='fidelity', measurement='XYZ', outer_kernel='gaussian', outer_kernel_params=None, regularization=None,
                           data_seed=None, param_seed=42):
    """
    Generate synthetic data using a quantum Gaussian Process.
    
    Args:
        num_samples (int): Number of samples to generate
        input_dim (int): Input dimensionality (1-6D supported)
        num_qubits (int): Number of qubits for quantum kernel
        num_layers (int): Number of layers in encoding circuit
        data_range (tuple): Range for input data generation
        noise_std (float): Noise standard deviation
        use_parameter_shift (bool): Use parameter shift rule
        kernel_params (np.array): Optional kernel parameters
        encoding_type (str): Encoding circuit type
        kernel_type (str): Quantum kernel type
        measurement (str): Measurement operator for ProjectedQuantumKernel
        outer_kernel (str): Outer kernel type for ProjectedQuantumKernel
        outer_kernel_params (dict): Parameters for the outer kernel
        regularization (str): Regularization technique for ProjectedQuantumKernel
        data_seed (int): Random seed for data generation (X, Y). If None, uses random seed based on time
        param_seed (int): Random seed for ground truth parameters (default: 42, keeps parameters consistent)
    
    Returns:
        tuple: (X, Y, ground_truth_params) where X is input, Y is output, and ground_truth_params are the kernel parameters used
    """
    print(f"Generating {input_dim}D quantum GP dataset with {num_samples} samples...")
    
    # Validate input dimension
    if input_dim < 1 or input_dim > 6:
        raise ValueError(f"Input dimension must be between 1 and 6, got {input_dim}")
    
    # Create quantum kernel
    q_kernel = create_quantum_kernel(num_qubits, input_dim, num_layers, use_parameter_shift, encoding_type, kernel_type, measurement, outer_kernel, outer_kernel_params, regularization)
    
    # For ProjectedQuantumKernel, num_parameters might be None until first evaluation
    # Do a dummy evaluation with a small sample to initialize the kernel
    if q_kernel.num_parameters is None:
        try:
            X_dummy = np.random.uniform(data_range[0], data_range[1], size=(2, input_dim))
            _ = q_kernel.evaluate(X_dummy, X_dummy)
        except Exception as e:
            print(f"Warning: Dummy evaluation for kernel initialization failed: {e}")
            print("Attempting to continue without pre-initialization...")
    
    # Set kernel parameters if provided, otherwise use random with fixed seed for consistency
    # Get num_parameters safely - if still None, use encoding circuit parameters
    num_kernel_params = q_kernel.num_parameters
    if num_kernel_params is None:
        num_kernel_params = q_kernel.encoding_circuit.num_parameters
    
    if kernel_params is not None:
        if len(kernel_params) != num_kernel_params:
            raise ValueError(f"Expected {num_kernel_params} parameters, got {len(kernel_params)}")
        q_kernel.assign_parameters(kernel_params)
        ground_truth_params = np.round(kernel_params.copy(), 4)
        print(f"Using provided kernel parameters: {ground_truth_params}")
    else:
        # Use fixed seed for ground truth parameters to keep them consistent across runs
        np.random.seed(param_seed)
        ground_truth_params = np.round(np.random.uniform(0, np.pi, num_kernel_params), 4)
        q_kernel.assign_parameters(ground_truth_params)
        print(f"Using random kernel parameters (seed={param_seed}): {ground_truth_params}")
    
    # Set seed for data generation (X) - different each time if data_seed is None
    if data_seed is None:
        data_seed = int(time.time() * 1000) % 2**32  # Use current time as seed
    np.random.seed(data_seed)
    print(f"Using data generation seed: {data_seed}")
    
    # Generate random input points
    X = np.random.uniform(data_range[0], data_range[1], size=(num_samples, input_dim))
    
    # Only clip for Chebyshev encoding (which uses arccos)
    if encoding_type == 'chebyshev':
        # Clip input data to avoid NaN issues with arccos in the encoding circuit
        # arccos requires input in [-1, 1], so we clip to a safe range
        X_clipped = np.clip(X, -0.99, 0.99)
        
        # Check if clipping was necessary
        if not np.array_equal(X, X_clipped):
            clipped_count = np.sum(X != X_clipped)
            print(f"Warning: Clipped {clipped_count} values to avoid arccos NaN issues")
            print(f"Original range: [{X.min():.3f}, {X.max():.3f}]")
            print(f"Clipped range: [{X_clipped.min():.3f}, {X_clipped.max():.3f}]")
            X = X_clipped
    else:
        print(f"Using {encoding_type} encoding - no input clipping needed")
    
    print(f"Computing quantum kernel matrix for {num_samples} points...")
    kernel_start = time.time()
    
    # Compute kernel matrix K(X, X)
    try:
        K = q_kernel.evaluate(X, X)
        
        # Check for NaN or infinite values in the kernel matrix
        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            raise ValueError("Kernel matrix contains NaN or infinite values")
            
    except Exception as e:
        print(f"Error computing kernel matrix: {e}")
        if encoding_type == 'chebyshev':
            print("This might be due to arccos receiving values outside [-1, 1]")
            print("Try using a smaller data range, e.g., --data-range -0.5 0.5")
            print("Or use --encoding yz_cx for a safer encoding without arccos")
        else:
            print(f"Error with {encoding_type} encoding - this should not happen with safe encodings")
        raise
    
    kernel_time = time.time() - kernel_start
    print(f"Kernel matrix computation took: {kernel_time:.4f}s")
    
    # Add small diagonal term for numerical stability
    jitter = 1e-6
    K += jitter * np.eye(num_samples)
    
    # Sample from multivariate normal with mean=0 and covariance=K
    print("Sampling from quantum GP prior...")
    try:
        # Cholesky decomposition for sampling
        L = np.linalg.cholesky(K)
        z = np.random.normal(0, 1, num_samples)
        Y = L @ z
        
        # Add observation noise (using same data seed for consistency with X generation)
        Y += np.random.normal(0, noise_std, num_samples)
        
    except np.linalg.LinAlgError:
        print("Cholesky decomposition failed, using eigendecomposition...")
        # Fallback: eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(K)
        eigenvals = np.maximum(eigenvals, 1e-10)  # Ensure positive
        sqrt_eigenvals = np.sqrt(eigenvals)
        z = np.random.normal(0, 1, num_samples)
        Y = eigenvecs @ (sqrt_eigenvals * z)
        Y += np.random.normal(0, noise_std, num_samples)
    
    print(f"Generated data - X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Y statistics - mean: {Y.mean():.4f}, std: {Y.std():.4f}")
    
    return X, Y, ground_truth_params

def plot_quantum_gp_data(X, Y, title="Quantum GP Generated Data", train_indices=None, test_indices=None):
    """
    Plot the generated quantum GP data for different dimensions.
    If train_indices and test_indices are provided, they will be colored differently.
    """
    input_dim = X.shape[1]
    
    if input_dim == 1:
        # 1D plot
        plt.figure(figsize=(10, 6))
        if train_indices is not None and test_indices is not None:
            plt.scatter(X[train_indices, 0], Y[train_indices], alpha=0.7, s=30, c='blue', marker='o', label='Training')
            plt.scatter(X[test_indices, 0], Y[test_indices], alpha=0.7, s=30, c='red', marker='s', label='Test')
            plt.legend()
        else:
            plt.scatter(X[:, 0], Y, alpha=0.7, s=20)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    elif input_dim == 2:
        # 2D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        if train_indices is not None and test_indices is not None:
            ax.scatter(X[train_indices, 0], X[train_indices, 1], Y[train_indices], c='blue', s=30, alpha=0.7, marker='o', label='Training')
            ax.scatter(X[test_indices, 0], X[test_indices, 1], Y[test_indices], c='red', s=30, alpha=0.7, marker='s', label='Test')
            ax.legend()
        else:
            ax.scatter(X[:, 0], X[:, 1], Y, c=Y, cmap='viridis', s=20)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_title(title)
        plt.show()
        
    elif input_dim == 3:
        # 3D input: multiple 2D projections
        fig = plt.figure(figsize=(15, 5))
        
        # Plot X1 vs X2 colored by Y
        ax1 = fig.add_subplot(131)
        if train_indices is not None and test_indices is not None:
            ax1.scatter(X[train_indices, 0], X[train_indices, 1], c='blue', s=30, alpha=0.7, marker='o', label='Training')
            ax1.scatter(X[test_indices, 0], X[test_indices, 1], c='red', s=30, alpha=0.7, marker='s', label='Test')
            ax1.legend()
        else:
            scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', s=20)
            plt.colorbar(scatter1, ax=ax1)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title('X1 vs X2' + (' (colored by Y)' if train_indices is None else ''))
        
        # Plot X1 vs X3 colored by Y
        ax2 = fig.add_subplot(132)
        if train_indices is not None and test_indices is not None:
            ax2.scatter(X[train_indices, 0], X[train_indices, 2], c='blue', s=30, alpha=0.7, marker='o', label='Training')
            ax2.scatter(X[test_indices, 0], X[test_indices, 2], c='red', s=30, alpha=0.7, marker='s', label='Test')
            ax2.legend()
        else:
            scatter2 = ax2.scatter(X[:, 0], X[:, 2], c=Y, cmap='viridis', s=20)
            plt.colorbar(scatter2, ax=ax2)
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X3')
        ax2.set_title('X1 vs X3' + (' (colored by Y)' if train_indices is None else ''))
        
        # Plot X2 vs X3 colored by Y
        ax3 = fig.add_subplot(133)
        if train_indices is not None and test_indices is not None:
            ax3.scatter(X[train_indices, 1], X[train_indices, 2], c='blue', s=30, alpha=0.7, marker='o', label='Training')
            ax3.scatter(X[test_indices, 1], X[test_indices, 2], c='red', s=30, alpha=0.7, marker='s', label='Test')
            ax3.legend()
        else:
            scatter3 = ax3.scatter(X[:, 1], X[:, 2], c=Y, cmap='viridis', s=20)
            plt.colorbar(scatter3, ax=ax3)
        ax3.set_xlabel('X2')
        ax3.set_ylabel('X3')
        ax3.set_title('X2 vs X3' + (' (colored by Y)' if train_indices is None else ''))
        
        plt.tight_layout()
        plt.show()
        
    elif input_dim >= 4:
        # Higher dimensions: show pairwise plots for first few dimensions
        n_plots = min(6, input_dim * (input_dim - 1) // 2)  # Max 6 pairwise plots
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes]
        
        plot_idx = 0
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                if plot_idx >= n_plots:
                    break
                    
                row = plot_idx // cols
                col = plot_idx % cols
                
                if rows > 1:
                    ax = axes[row][col]
                else:
                    ax = axes[col]
                
                if train_indices is not None and test_indices is not None:
                    ax.scatter(X[train_indices, i], X[train_indices, j], c='blue', s=30, alpha=0.7, marker='o', label='Training')
                    ax.scatter(X[test_indices, i], X[test_indices, j], c='red', s=30, alpha=0.7, marker='s', label='Test')
                    ax.legend()
                    ax.set_title(f'X{i+1} vs X{j+1}')
                else:
                    scatter = ax.scatter(X[:, i], X[:, j], c=Y, cmap='viridis', s=20, alpha=0.7)
                    ax.set_title(f'X{i+1} vs X{j+1} (colored by Y)')
                    plt.colorbar(scatter, ax=ax)
                
                ax.set_xlabel(f'X{i+1}')
                ax.set_ylabel(f'X{j+1}')
                
                plot_idx += 1
                
            if plot_idx >= n_plots:
                break
        
        # Hide unused subplots
        for idx in range(plot_idx, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row][col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(f'{title} ({input_dim}D Input)')
        plt.tight_layout()
        plt.show()

def save_quantum_dataset(X, Y, dataset_name, output_dir="quantum_datasets"):
    """
    Save the generated quantum dataset to CSV files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save X and Y in the same file
    combined_data = np.column_stack((X, Y))
    combined_filename = os.path.join(output_dir, f"{dataset_name}_{X.shape[1]}d_{X.shape[0]}.csv")
    
    # Create header for the CSV file
    x_headers = [f"X{i+1}" for i in range(X.shape[1])]
    y_header = ["Y"]
    header = ",".join(x_headers + y_header)
    
    np.savetxt(combined_filename, combined_data, delimiter=",", header=header, comments="")
    
    print(f"Dataset saved:")
    print(f"  Combined file: {combined_filename}")
    print(f"  Format: {X.shape[1]} input columns + 1 output column")
    
    return combined_filename

def generate_data_numpy(num_samples, input_dim=1, noise_std=0.1, data_seed=None):
    """
    Generate training data using numpy (adapted from utils.py)
    
    Args:
        num_samples (int): Number of samples to generate
        input_dim (int): Input dimensionality (1, 2, or 3)
        noise_std (float): Noise standard deviation
        data_seed (int): Random seed for data generation. If None, uses random seed based on time
    """
    # Set seed for data generation - different each time if data_seed is None
    if data_seed is None:
        data_seed = int(time.time() * 1000) % 2**32  # Use current time as seed
    np.random.seed(data_seed)
    print(f"Using data generation seed: {data_seed}")
    
    if input_dim == 1:
        # 1D function: f(x) = 5x²sin(12x) + (x³-0.5)sin(3x-0.5) + 4cos(2x), Generalized Robust Bayesian Committee Machine for Large-scale Gaussian Process Regression
        X = np.random.uniform(0, 1, size=(num_samples, 1))
        x = X[:, 0]
        Y = 5 * x**2 * np.sin(12*x) + (x**3 - 0.5) * np.sin(3*x - 0.5) + 4 * np.cos(2*x)
        Y += np.random.normal(0, noise_std, num_samples)
        
    elif input_dim == 2:
        # 2D Goldstein-Price function (normalized)
        X = np.random.uniform(-2.0, 2.0, size=(num_samples, 2))
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        fact1a = (x1 + x2 + 1)**2
        fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
        fact1 = 1 + fact1a * fact1b

        fact2a = (2*x1 - 3*x2)**2
        fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
        fact2 = 30 + fact2a * fact2b

        prod = fact1 * fact2
        Y = (np.log(prod) - 8.693) / 2.427
        Y += np.random.normal(0, noise_std, num_samples)
        
    elif input_dim == 3:
        # 3D Hartmann function
        X = np.random.uniform(0.0, 1.0, size=(num_samples, 3))
        
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0],
                      [3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0]])
        P = 1e-4 * np.array([[3689.0, 1170.0, 2673.0],
                             [4699.0, 4387.0, 7470.0],
                             [1091.0, 8732.0, 5547.0],
                             [381.0, 5743.0, 8828.0]])
        
        Y = np.zeros(num_samples)
        for i in range(4):
            inner = np.sum(A[i, :] * (X - P[i, :])**2, axis=1)
            Y += alpha[i] * np.exp(-inner)
        Y = -Y  # Minimize (Hartmann is typically maximized)
        Y += np.random.normal(0, noise_std, num_samples)
    
    else:
        raise ValueError(f"Unsupported input dimension: {input_dim}")
    
    return X, Y

def _kd_bisect_numpy(indices, pts, target_cells):
    """
    Recursively split `indices` (a 1-D array of row indices into `pts`)
    until we have `target_cells` disjoint index subsets. Each split is along
    the longest side of the current bounding box at the median.
    Numpy version of the PyTorch function from utils.py.
    """
    cells = [indices]
    while len(cells) < target_cells:
        # pick the largest cell so we don't create imbalanced tiny pieces
        big_idx = max(range(len(cells)), key=lambda i: len(cells[i]))
        big_cell = cells.pop(big_idx)
        cell_pts = pts[big_cell]

        # find longest dimension of this cell
        ranges = cell_pts.max(axis=0) - cell_pts.min(axis=0)
        split_dim = np.argmax(ranges)

        # median along that dimension
        median_val = np.median(cell_pts[:, split_dim])
        left_mask = cell_pts[:, split_dim] <= median_val

        # make sure neither side becomes empty; if so nudge the threshold
        if left_mask.all() or (~left_mask).all():
            median_val = cell_pts[:, split_dim].mean()
            left_mask = cell_pts[:, split_dim] <= median_val

        cells.insert(big_idx, big_cell[left_mask])
        cells.append(big_cell[~left_mask])
    return cells

def _regular_grid_split_numpy(X, n_agents, agent_id):
    """
    Return boolean mask of rows belonging to this agent using regular grid split.
    Numpy version of the PyTorch function from utils.py.
    """
    N, d = X.shape
    # Number of cells per dimension (must be integer)
    cells_per_dim = round(n_agents ** (1 / d))
    if cells_per_dim ** d != n_agents:
        print(f"Warning: n_agents={n_agents} is not a perfect {d}-th power. Using k-d tree split instead.")
        return None, False
    
    # Map agent_id → tuple of cell indices (i0, i1, …, id-1)
    base = cells_per_dim
    digits = []
    r = agent_id
    for _ in range(d):
        digits.append(r % base)
        r //= base
    digits = digits[::-1]  # most-significant first

    mask = np.ones(N, dtype=bool)
    for j, ij in enumerate(digits):
        low = X[:, j].min()
        high = X[:, j].max()
        edges = np.linspace(low, high, cells_per_dim + 1)
        mask &= (X[:, j] >= edges[ij]) & (X[:, j] <= edges[ij + 1])

    return mask, True

def sample_agent_data_percentage(X_agent, Y_agent, percentage, random_seed=42):
    """
    Randomly sample a percentage of the agent's data.
    
    Args:
        X_agent: Input data for agent (N, D)
        Y_agent: Output data for agent (N,)
        percentage: Percentage of data to sample (0.0 to 1.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X_sampled, Y_sampled)
    """
    if percentage <= 0.0 or percentage > 1.0:
        raise ValueError(f"Percentage must be between 0.0 and 1.0, got {percentage}")
    
    n_samples = X_agent.shape[0]
    n_to_sample = max(1, int(n_samples * percentage))  # At least 1 sample
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Randomly sample indices
    sample_indices = np.random.choice(n_samples, size=n_to_sample, replace=False)
    
    return X_agent[sample_indices], Y_agent[sample_indices]

def split_data_numpy(X, Y, n_agents, partition_method='regional', data_percentage=1.0, random_seed=42):
    """
    Split data among agents using numpy (adapted from utils.py)
    
    Args:
        X: Input data array (N, D)
        Y: Output data array (N,)
        n_agents: Number of agents to split data among
        partition_method: Method for partitioning data
            - 'regional': Spatial/local partitioning based on input space regions
            - 'random': Random shuffling and splitting
            - 'sequential': Sequential splitting
        data_percentage: Percentage of local data to give to each agent (0.0 to 1.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of (X_agent, Y_agent) tuples for each agent
    """
    n_samples = X.shape[0]
    input_dim = X.shape[1] if X.ndim > 1 else 1
    
    if partition_method == 'regional':
        # Spatial/regional partitioning based on input space
        print(f"Using regional partitioning for {input_dim}D data with {n_agents} agents")
        
        if input_dim == 1:
            # For 1D, use sequential spatial splitting
            sorted_indices = np.argsort(X[:, 0])
            splits = np.array_split(sorted_indices, n_agents)
        else:
            # For multi-dimensional data, try regular grid first, fallback to k-d tree
            splits = []
            for agent_id in range(n_agents):
                mask, success = _regular_grid_split_numpy(X, n_agents, agent_id)
                if success:
                    agent_indices = np.where(mask)[0]
                    splits.append(agent_indices)
                else:
                    # Fallback: use k-d tree bisection
                    print(f"Using k-d tree bisection for spatial partitioning")
                    all_indices = np.arange(n_samples)
                    cells = _kd_bisect_numpy(all_indices, X, n_agents)
                    splits = cells
                    break
        
    elif partition_method == 'random':
        # Random shuffling and splitting
        np.random.seed(random_seed)  # Use provided random seed for reproducibility
        indices = np.random.permutation(n_samples)
        splits = np.array_split(indices, n_agents)
        
    elif partition_method == 'sequential':
        # Sequential splitting
        splits = np.array_split(np.arange(n_samples), n_agents)
        
    else:
        raise ValueError(f"Unknown partition method: {partition_method}. Choose from: 'regional', 'random', 'sequential'")
    
    # Return list of (X_agent, Y_agent) tuples
    agent_data = []
    for split_indices in splits:
        X_agent = X[split_indices]
        Y_agent = Y[split_indices]
        
        # Apply percentage sampling if requested
        if data_percentage < 1.0:
            X_agent, Y_agent = sample_agent_data_percentage(X_agent, Y_agent, data_percentage, random_seed)
            
        agent_data.append((X_agent, Y_agent))
    
    return agent_data

def plot_agent_data_distribution(agent_data_splits, title="Agent Data Distribution", save_plot=False, output_dir="plots"):
    """
    Plot how the data is distributed among agents, similar to the training dataset plot style.
    
    Args:
        agent_data_splits: List of (X_agent, Y_agent) tuples for each agent
        title: Title for the plot
        save_plot: Whether to save the plot to file
        output_dir: Directory to save plots
    """
    n_agents = len(agent_data_splits)
    input_dim = agent_data_splits[0][0].shape[1] if agent_data_splits[0][0].ndim > 1 else 1
    
    # Create a colormap for agents
    colors = plt.cm.Set3(np.linspace(0, 1, n_agents))
    
    if input_dim == 1:
        # 1D plot - similar to plot_quantum_gp_data
        plt.figure(figsize=(10, 6))
        
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            plt.scatter(X_agent[:, 0], Y_agent, alpha=0.7, s=20, c=[colors[i]], 
                       label=f'Agent {i+1} ({len(X_agent)} samples)')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    elif input_dim == 2:
        # Enhanced 2D visualization with comprehensive analysis
        fig = plt.figure(figsize=(18, 12))
        
        # Calculate global bounds for consistent scaling
        all_X = np.vstack([X_agent for X_agent, _ in agent_data_splits])
        x1_bounds = [all_X[:, 0].min(), all_X[:, 0].max()]
        x2_bounds = [all_X[:, 1].min(), all_X[:, 1].max()]
        
        # Top row: Input space visualization and output visualization
        ax1 = fig.add_subplot(231)
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax1.scatter(X_agent[:, 0], X_agent[:, 1], c=[colors[i]], s=30, alpha=0.8,
                       label=f'Agent {i+1} ({len(X_agent)} samples)', edgecolors='black', linewidths=0.3)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title('Input Space Partitioning\n(X1 vs X2)', fontsize=12, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(x1_bounds)
        ax1.set_ylim(x2_bounds)
        
        # Show agent boundaries for regular grid partitioning
        if n_agents in [4, 9, 16, 25]:  # Perfect squares for 2D regular grid
            cells_per_dim = int(np.sqrt(n_agents))
            if cells_per_dim ** 2 == n_agents:
                # Draw grid boundaries
                x1_edges = np.linspace(x1_bounds[0], x1_bounds[1], cells_per_dim + 1)
                x2_edges = np.linspace(x2_bounds[0], x2_bounds[1], cells_per_dim + 1)
                
                for x1_edge in x1_edges:
                    ax1.axvline(x1_edge, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
                for x2_edge in x2_edges:
                    ax1.axhline(x2_edge, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Output values colored by agent (3D visualization)
        ax2 = fig.add_subplot(232, projection='3d')
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax2.scatter(X_agent[:, 0], X_agent[:, 1], Y_agent, c=[colors[i]], s=25, alpha=0.8,
                       edgecolors='black', linewidths=0.2)
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_zlabel('Y')
        ax2.set_title('Output Values by Agent\n(X1, X2, Y)', fontsize=12, fontweight='bold')
        
        # Agent boundaries visualization
        ax3 = fig.add_subplot(233)
        
        # Show agent regions as rectangles (for regular grid partitioning)
        if n_agents in [4, 9, 16, 25]:  # Perfect squares for 2D regular grid
            cells_per_dim = int(np.sqrt(n_agents))
            if cells_per_dim ** 2 == n_agents:
                # Draw grid boundaries
                x1_edges = np.linspace(x1_bounds[0], x1_bounds[1], cells_per_dim + 1)
                x2_edges = np.linspace(x2_bounds[0], x2_bounds[1], cells_per_dim + 1)
                
                for x1_edge in x1_edges:
                    ax3.axvline(x1_edge, color='black', linestyle='--', alpha=0.5)
                for x2_edge in x2_edges:
                    ax3.axhline(x2_edge, color='black', linestyle='--', alpha=0.5)
                
                # Color agent regions
                for agent_idx in range(n_agents):
                    i = agent_idx % cells_per_dim
                    j = agent_idx // cells_per_dim
                    
                    rect = plt.Rectangle((x1_edges[i], x2_edges[j]), 
                                       x1_edges[i+1] - x1_edges[i], 
                                       x2_edges[j+1] - x2_edges[j],
                                       facecolor=colors[agent_idx], alpha=0.3, 
                                       edgecolor='black', linewidth=1)
                    ax3.add_patch(rect)
                    
                    # Add agent label in center of rectangle
                    center_x1 = (x1_edges[i] + x1_edges[i+1]) / 2
                    center_x2 = (x2_edges[j] + x2_edges[j+1]) / 2
                    ax3.text(center_x1, center_x2, f'A{agent_idx+1}', 
                           ha='center', va='center', fontweight='bold', fontsize=10)
                
                ax3.set_title(f'Agent Regions\nRegular Grid: {cells_per_dim}×{cells_per_dim}', 
                            fontsize=12, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'K-d Tree Partitioning\n(Irregular boundaries)', 
                       ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'K-d Tree Partitioning\n(Irregular boundaries)', 
                   ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        ax3.set_xlabel('X1')
        ax3.set_ylabel('X2')
        ax3.set_xlim(x1_bounds)
        ax3.set_ylim(x2_bounds)
        ax3.grid(True, alpha=0.3)
        
        # Bottom row: Coverage analysis, overlap matrix, and density analysis
        # Coverage analysis - show how well agents cover the space
        ax4 = fig.add_subplot(234)
        
        # Create a grid to check coverage
        n_grid = 25
        x1_grid = np.linspace(x1_bounds[0], x1_bounds[1], n_grid)
        x2_grid = np.linspace(x2_bounds[0], x2_bounds[1], n_grid)
        
        coverage_map = np.zeros((n_grid, n_grid))
        
        for i in range(n_grid):
            for j in range(n_grid):
                # For each grid point, find which agents have data nearby
                grid_point = np.array([x1_grid[i], x2_grid[j]])
                
                agents_nearby = []
                for agent_idx, (X_agent, _) in enumerate(agent_data_splits):
                    # Check if any agent data is within a reasonable distance
                    distances = np.linalg.norm(X_agent - grid_point, axis=1)
                    if np.min(distances) < 0.15:  # Threshold distance
                        agents_nearby.append(agent_idx)
                
                coverage_map[i, j] = len(agents_nearby)
        
        im = ax4.imshow(coverage_map.T, origin='lower', extent=[x1_bounds[0], x1_bounds[1], x2_bounds[0], x2_bounds[1]], 
                       cmap='RdYlGn', alpha=0.7)
        plt.colorbar(im, ax=ax4, label='Number of agents\nwith nearby data')
        ax4.set_xlabel('X1')
        ax4.set_ylabel('X2')
        ax4.set_title('Spatial Coverage Analysis', fontsize=12, fontweight='bold')
        
        # Overlay actual data points
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax4.scatter(X_agent[:, 0], X_agent[:, 1], c=[colors[i]], s=15, alpha=0.6, 
                      edgecolors='black', linewidths=0.1)
        
        # Overlap analysis
        ax5 = fig.add_subplot(235)
        overlap_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    X_i = agent_data_splits[i][0]
                    X_j = agent_data_splits[j][0]
                    
                    # Calculate minimum distance between agent datasets
                    min_dist = float('inf')
                    for point_i in X_i:
                        distances = np.linalg.norm(X_j - point_i, axis=1)
                        min_dist = min(min_dist, np.min(distances))
                    
                    overlap_matrix[i, j] = min_dist
        
        im2 = ax5.imshow(overlap_matrix, cmap='viridis')
        ax5.set_xlabel('Agent ID')
        ax5.set_ylabel('Agent ID')
        ax5.set_title('Agent Overlap Matrix\n(Min distances)', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(n_agents))
        ax5.set_yticks(range(n_agents))
        ax5.set_xticklabels([f'A{i+1}' for i in range(n_agents)])
        ax5.set_yticklabels([f'A{i+1}' for i in range(n_agents)])
        plt.colorbar(im2, ax=ax5, label='Distance')
        
        # Add text annotations for distances
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    ax5.text(j, i, f'{overlap_matrix[i, j]:.2f}', 
                           ha='center', va='center', color='white', fontsize=8)
        
        # Density analysis
        ax6 = fig.add_subplot(236)
        densities = []
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            # Calculate area covered by each agent
            if len(X_agent) > 2:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(X_agent)
                    area = hull.volume  # In 2D, volume is area
                    density = len(X_agent) / area if area > 0 else len(X_agent)
                except:
                    # Fallback: use bounding box area
                    ranges = X_agent.max(axis=0) - X_agent.min(axis=0)
                    area = np.prod(ranges)
                    density = len(X_agent) / area if area > 0 else len(X_agent)
            else:
                density = len(X_agent)
            densities.append(density)
        
        bars = ax6.bar(range(n_agents), densities, color=colors[:n_agents], alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Agent ID')
        ax6.set_ylabel('Data Density\n(samples/area)')
        ax6.set_title('Data Density per Agent', fontsize=12, fontweight='bold')
        ax6.set_xticks(range(n_agents))
        ax6.set_xticklabels([f'A{i+1}' for i in range(n_agents)])
        
        # Add value labels on bars
        for i, (bar, density) in enumerate(zip(bars, densities)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(densities)*0.01,
                    f'{density:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Additional detailed analysis plot
        fig2, (ax_stats, ax_detailed) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Statistics table
        ax_stats.axis('off')
        ax_stats.set_title('Partitioning Statistics', fontweight='bold', fontsize=14)
        
        stats_text = f"""
Partitioning Method: Regional (Spatial)
Total Agents: {n_agents}
Total Samples: {sum(len(X_agent) for X_agent, _ in agent_data_splits)}
Input Space Bounds:
  X1: [{x1_bounds[0]:.3f}, {x1_bounds[1]:.3f}]
  X2: [{x2_bounds[0]:.3f}, {x2_bounds[1]:.3f}]

Agent Sample Counts:
"""
        
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            percentage = (len(X_agent) / sum(len(X_a) for X_a, _ in agent_data_splits)) * 100
            stats_text += f"  Agent {i+1}: {len(X_agent)} samples ({percentage:.1f}%)\n"
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Detailed density visualization
        ax_detailed.set_title('Data Point Density Visualization', fontweight='bold', fontsize=14)
        
        # Create density heatmap
        from scipy.stats import gaussian_kde
        all_points = np.vstack([X_agent for X_agent, _ in agent_data_splits])
        
        # Create grid for density calculation
        xi = np.linspace(x1_bounds[0], x1_bounds[1], 50)
        yi = np.linspace(x2_bounds[0], x2_bounds[1], 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        try:
            kde = gaussian_kde(all_points.T)
            zi = kde(np.vstack([Xi.flatten(), Yi.flatten()]))
            zi = zi.reshape(Xi.shape)
            
            contour = ax_detailed.contourf(Xi, Yi, zi, levels=20, cmap='Blues', alpha=0.6)
            plt.colorbar(contour, ax=ax_detailed, label='Data Density')
        except:
            # Fallback if KDE fails
            pass
        
        # Overlay data points
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax_detailed.scatter(X_agent[:, 0], X_agent[:, 1], c=[colors[i]], s=25, alpha=0.8,
                              edgecolors='black', linewidths=0.2, label=f'Agent {i+1}')
        
        ax_detailed.set_xlabel('X1')
        ax_detailed.set_ylabel('X2')
        ax_detailed.set_xlim(x1_bounds)
        ax_detailed.set_ylim(x2_bounds)
        ax_detailed.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax_detailed.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    elif input_dim == 3:
        # Enhanced 3D visualization with true 3D plot and 2D projections
        fig = plt.figure(figsize=(20, 12))
        
        # Calculate global bounds for consistent scaling
        all_X = np.vstack([X_agent for X_agent, _ in agent_data_splits])
        x1_bounds = [all_X[:, 0].min(), all_X[:, 0].max()]
        x2_bounds = [all_X[:, 1].min(), all_X[:, 1].max()]
        x3_bounds = [all_X[:, 2].min(), all_X[:, 2].max()]
        
        # Top row: True 3D scatter plot of input space
        ax_3d = fig.add_subplot(231, projection='3d')
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax_3d.scatter(X_agent[:, 0], X_agent[:, 1], X_agent[:, 2], 
                         c=[colors[i]], s=30, alpha=0.8,
                         label=f'Agent {i+1} ({len(X_agent)} samples)')
        ax_3d.set_xlabel('X1')
        ax_3d.set_ylabel('X2')
        ax_3d.set_zlabel('X3')
        ax_3d.set_title('3D Input Space Partitioning\n(True 3D View)', fontsize=12, fontweight='bold')
        ax_3d.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize='small')
        
        # Add bounding box to show full space coverage
        from itertools import product
        corners = list(product(x1_bounds, x2_bounds, x3_bounds))
        for i in range(len(corners)):
            for j in range(i+1, len(corners)):
                # Only draw edges between corners that differ in exactly one coordinate
                diff_count = sum(1 for k in range(3) if corners[i][k] != corners[j][k])
                if diff_count == 1:
                    ax_3d.plot([corners[i][0], corners[j][0]], 
                              [corners[i][1], corners[j][1]], 
                              [corners[i][2], corners[j][2]], 
                              'k--', alpha=0.3, linewidth=0.5)
        
        # Top right: Agent boundaries visualization
        ax_bounds = fig.add_subplot(232)
        
        # Show agent regions as rectangles (for regular grid partitioning)
        if n_agents in [8, 27, 64]:  # Perfect cubes for 3D regular grid
            cells_per_dim = round(n_agents ** (1/3))
            if cells_per_dim ** 3 == n_agents:
                # Draw grid boundaries
                x1_edges = np.linspace(x1_bounds[0], x1_bounds[1], cells_per_dim + 1)
                x2_edges = np.linspace(x2_bounds[0], x2_bounds[1], cells_per_dim + 1)
                
                # Show 2D projection of 3D grid (X1 vs X2)
                for x1_edge in x1_edges:
                    ax_bounds.axvline(x1_edge, color='black', linestyle='--', alpha=0.5)
                for x2_edge in x2_edges:
                    ax_bounds.axhline(x2_edge, color='black', linestyle='--', alpha=0.5)
                
                # Color agent regions
                for agent_idx in range(n_agents):
                    # Calculate 3D cell indices
                    i = agent_idx % cells_per_dim
                    j = (agent_idx // cells_per_dim) % cells_per_dim
                    k = agent_idx // (cells_per_dim ** 2)
                    
                    # Only show if k==0 (front slice of 3D grid)
                    if k == 0:
                        rect = plt.Rectangle((x1_edges[i], x2_edges[j]), 
                                           x1_edges[i+1] - x1_edges[i], 
                                           x2_edges[j+1] - x2_edges[j],
                                           facecolor=colors[agent_idx], alpha=0.2, 
                                           edgecolor='black', linewidth=1)
                        ax_bounds.add_patch(rect)
                        
                        # Add agent label in center of rectangle
                        center_x1 = (x1_edges[i] + x1_edges[i+1]) / 2
                        center_x2 = (x2_edges[j] + x2_edges[j+1]) / 2
                        ax_bounds.text(center_x1, center_x2, f'A{agent_idx+1}', 
                                     ha='center', va='center', fontweight='bold', fontsize=8)
                
                ax_bounds.set_title(f'Agent Regions (X1-X2 slice, k=0)\nRegular Grid: {cells_per_dim}×{cells_per_dim}×{cells_per_dim}', 
                                  fontsize=10, fontweight='bold')
            else:
                ax_bounds.text(0.5, 0.5, 'K-d Tree Partitioning\n(Irregular boundaries)', 
                             ha='center', va='center', transform=ax_bounds.transAxes, fontsize=12)
        else:
            ax_bounds.text(0.5, 0.5, 'K-d Tree Partitioning\n(Irregular boundaries)', 
                         ha='center', va='center', transform=ax_bounds.transAxes, fontsize=12)
        
        ax_bounds.set_xlabel('X1')
        ax_bounds.set_ylabel('X2')
        ax_bounds.set_xlim(x1_bounds)
        ax_bounds.set_ylim(x2_bounds)
        ax_bounds.grid(True, alpha=0.3)
        
        # Bottom row: 2D projections with enhanced visualization
        # Plot X1 vs X2 colored by agent
        ax1 = fig.add_subplot(234)
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax1.scatter(X_agent[:, 0], X_agent[:, 1], c=[colors[i]], s=25, alpha=0.8,
                       label=f'Agent {i+1} ({len(X_agent)} samples)', edgecolors='black', linewidths=0.2)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title('X1 vs X2 Projection', fontweight='bold')
        ax1.set_xlim(x1_bounds)
        ax1.set_ylim(x2_bounds)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Plot X1 vs X3 colored by agent
        ax2 = fig.add_subplot(235)
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax2.scatter(X_agent[:, 0], X_agent[:, 2], c=[colors[i]], s=25, alpha=0.8,
                       edgecolors='black', linewidths=0.2)
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X3')
        ax2.set_title('X1 vs X3 Projection', fontweight='bold')
        ax2.set_xlim(x1_bounds)
        ax2.set_ylim(x3_bounds)
        ax2.grid(True, alpha=0.3)
        
        # Plot X2 vs X3 colored by agent
        ax3 = fig.add_subplot(236)
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax3.scatter(X_agent[:, 1], X_agent[:, 2], c=[colors[i]], s=25, alpha=0.8,
                       edgecolors='black', linewidths=0.2)
        ax3.set_xlabel('X2')
        ax3.set_ylabel('X3')
        ax3.set_title('X2 vs X3 Projection', fontweight='bold')
        ax3.set_xlim(x2_bounds)
        ax3.set_ylim(x3_bounds)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional coverage analysis plot
        fig2, ((ax_coverage, ax_overlap), (ax_density, ax_stats)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Coverage analysis - show how well agents cover the space
        ax_coverage.set_title('Spatial Coverage Analysis', fontweight='bold', fontsize=14)
        
        # Create a grid to check coverage
        n_grid = 20
        x1_grid = np.linspace(x1_bounds[0], x1_bounds[1], n_grid)
        x2_grid = np.linspace(x2_bounds[0], x2_bounds[1], n_grid)
        x3_grid = np.linspace(x3_bounds[0], x3_bounds[1], n_grid)
        
        coverage_map = np.zeros((n_grid, n_grid))
        
        for i in range(n_grid):
            for j in range(n_grid):
                # For each grid point, find which agents have data nearby
                grid_point = np.array([x1_grid[i], x2_grid[j], x3_bounds[0]])  # Use middle of X3 range
                
                agents_nearby = []
                for agent_idx, (X_agent, _) in enumerate(agent_data_splits):
                    # Check if any agent data is within a reasonable distance
                    distances = np.linalg.norm(X_agent - grid_point, axis=1)
                    if np.min(distances) < 0.2:  # Threshold distance
                        agents_nearby.append(agent_idx)
                
                coverage_map[i, j] = len(agents_nearby)
        
        im = ax_coverage.imshow(coverage_map.T, origin='lower', extent=[x1_bounds[0], x1_bounds[1], x2_bounds[0], x2_bounds[1]], 
                               cmap='RdYlGn', alpha=0.7)
        plt.colorbar(im, ax=ax_coverage, label='Number of agents with nearby data')
        ax_coverage.set_xlabel('X1')
        ax_coverage.set_ylabel('X2')
        
        # Overlay actual data points
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            ax_coverage.scatter(X_agent[:, 0], X_agent[:, 1], c=[colors[i]], s=10, alpha=0.6, 
                              edgecolors='black', linewidths=0.1)
        
        # Overlap analysis
        ax_overlap.set_title('Agent Data Distribution Overlap', fontweight='bold', fontsize=14)
        overlap_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    X_i = agent_data_splits[i][0]
                    X_j = agent_data_splits[j][0]
                    
                    # Calculate minimum distance between agent datasets
                    min_dist = float('inf')
                    for point_i in X_i:
                        distances = np.linalg.norm(X_j - point_i, axis=1)
                        min_dist = min(min_dist, np.min(distances))
                    
                    overlap_matrix[i, j] = min_dist
        
        im2 = ax_overlap.imshow(overlap_matrix, cmap='viridis')
        ax_overlap.set_xlabel('Agent ID')
        ax_overlap.set_ylabel('Agent ID')
        ax_overlap.set_xticks(range(n_agents))
        ax_overlap.set_yticks(range(n_agents))
        ax_overlap.set_xticklabels([f'A{i+1}' for i in range(n_agents)])
        ax_overlap.set_yticklabels([f'A{i+1}' for i in range(n_agents)])
        plt.colorbar(im2, ax=ax_overlap, label='Minimum distance between agents')
        
        # Add text annotations for distances
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    ax_overlap.text(j, i, f'{overlap_matrix[i, j]:.2f}', 
                                  ha='center', va='center', color='white', fontsize=8)
        
        # Density analysis
        ax_density.set_title('Data Density per Agent', fontweight='bold', fontsize=14)
        densities = []
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            # Calculate bounding box volume for each agent
            if len(X_agent) > 1:
                ranges = X_agent.max(axis=0) - X_agent.min(axis=0)
                volume = np.prod(ranges)
                density = len(X_agent) / volume if volume > 0 else len(X_agent)
            else:
                density = 1
            densities.append(density)
        
        bars = ax_density.bar(range(n_agents), densities, color=colors[:n_agents], alpha=0.7, edgecolor='black')
        ax_density.set_xlabel('Agent ID')
        ax_density.set_ylabel('Data Density (samples/volume)')
        ax_density.set_xticks(range(n_agents))
        ax_density.set_xticklabels([f'A{i+1}' for i in range(n_agents)])
        
        # Add value labels on bars
        for i, (bar, density) in enumerate(zip(bars, densities)):
            ax_density.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(densities)*0.01,
                          f'{density:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Statistics table
        ax_stats.axis('off')
        ax_stats.set_title('Partitioning Statistics', fontweight='bold', fontsize=14)
        
        stats_text = f"""
Partitioning Method: Regional (Spatial)
Total Agents: {n_agents}
Total Samples: {sum(len(X_agent) for X_agent, _ in agent_data_splits)}
Input Space Bounds:
  X1: [{x1_bounds[0]:.3f}, {x1_bounds[1]:.3f}]
  X2: [{x2_bounds[0]:.3f}, {x2_bounds[1]:.3f}]
  X3: [{x3_bounds[0]:.3f}, {x3_bounds[1]:.3f}]

Agent Sample Counts:
"""
        
        for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
            percentage = (len(X_agent) / sum(len(X_a) for X_a, _ in agent_data_splits)) * 100
            stats_text += f"  Agent {i+1}: {len(X_agent)} samples ({percentage:.1f}%)\n"
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    elif input_dim >= 4:
        # Higher dimensions: show pairwise plots - similar to plot_quantum_gp_data
        n_plots = min(6, input_dim * (input_dim - 1) // 2)  # Max 6 pairwise plots
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes]
        
        plot_idx = 0
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                if plot_idx >= n_plots:
                    break
                    
                row = plot_idx // cols
                col = plot_idx % cols
                
                if rows > 1:
                    ax = axes[row][col]
                else:
                    ax = axes[col]
                
                # Plot each agent's data
                for agent_idx, (X_agent, Y_agent) in enumerate(agent_data_splits):
                    ax.scatter(X_agent[:, i], X_agent[:, j], c=[colors[agent_idx]], s=20, alpha=0.7,
                              label=f'Agent {agent_idx+1} ({len(X_agent)} samples)')
                
                ax.set_xlabel(f'X{i+1}')
                ax.set_ylabel(f'X{j+1}')
                ax.set_title(f'X{i+1} vs X{j+1} (colored by Agent)')
                if plot_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                
                plot_idx += 1
                
            if plot_idx >= n_plots:
                break
        
        # Hide unused subplots
        for idx in range(plot_idx, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row][col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(f'{title} ({input_dim}D Input)')
        plt.tight_layout()
        plt.show()
    
    # Print data distribution statistics
    print(f"\nAgent Data Distribution Summary:")
    total_samples = sum(len(X_agent) for X_agent, _ in agent_data_splits)
    for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
        percentage = (len(X_agent) / total_samples) * 100
        print(f"  Agent {i+1}: {len(X_agent)} samples ({percentage:.1f}%)")
        if input_dim <= 3:
            # Show spatial bounds for low-dimensional data
            bounds = []
            for dim in range(input_dim):
                min_val = X_agent[:, dim].min()
                max_val = X_agent[:, dim].max()
                bounds.append(f"X{dim+1}: [{min_val:.3f}, {max_val:.3f}]")
            print(f"    Spatial bounds: {', '.join(bounds)}")
    
    # Save plot if requested
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"agent_data_distribution_{input_dim}d_{n_agents}agents.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    plt.show()

def process_agent_training(agent_data):
    """
    Helper function for ProcessPoolExecutor that recreates Agent and trains it.
    This avoids pickling issues with complex quantum objects.
    Now uses a process-global quantum kernel for maximum efficiency.
    
    Args:
        agent_data: tuple containing (agent_id, X_sub, Y_sub, num_qubits, noise_std, rho, L, z, psi_i, use_parameter_shift, num_features, num_layers, num_workers, shift_value, encoding_type, kernel_type, measurement, riemannian_lr, riemannian_method, riemannian_beta, outer_kernel, outer_kernel_params, regularization)
    
    Returns:
        tuple: (theta_i, psi_i, nll_loss, condition_number, nll_components)
    """
    agent_id, X_sub, Y_sub, num_qubits, noise_std, rho, L, z, psi_i, use_parameter_shift, num_features, num_layers, num_workers, shift_value, encoding_type, kernel_type, measurement, riemannian_lr, riemannian_method, riemannian_beta, outer_kernel, outer_kernel_params, regularization, noise_lower_bound = agent_data
    
    # Get quantum kernel for this process (created only once per process, reused across iterations)
    q_kernel = get_or_create_process_kernel(num_qubits, num_features, num_layers, use_parameter_shift, encoding_type, kernel_type, measurement, outer_kernel, outer_kernel_params, regularization)
    
    # Print parameter shift usage information
    if use_parameter_shift:
        # print(f"Process {os.getpid()} - Agent {agent_id} - Using parameter shift (parallel evaluation enabled)")
        pass
    else:
        # print(f"Process {os.getpid()} - Agent {agent_id} - Using autodiff")
        pass
    
    # Always use the Riemannian agent path.
    agent = RiemannianAgent(
        agent_id=agent_id,
        X_sub=X_sub,
        Y_sub=Y_sub,
        num_qubits=num_qubits,
        noise_std=noise_std,
        rho=rho,
        L=L,
        q_kernel=q_kernel,
        use_parameter_shift=use_parameter_shift,
        num_workers=num_workers,
        shift_value=shift_value,
        num_layers=num_layers,
        encoding_type=encoding_type,
        kernel_type=kernel_type,
        measurement=measurement,
        outer_kernel=outer_kernel,
        outer_kernel_params=outer_kernel_params,
        regularization=regularization,
        riemannian_lr=riemannian_lr,
        riemannian_method=riemannian_method,
        riemannian_beta=riemannian_beta,
        noise_lower_bound=noise_lower_bound
    )
    
    # Train and update the agent
    return agent.train_and_update(z, psi_i)

# ===========================
# PREDICTION METHOD HELPERS
# ===========================

def extract_noise_level_quantum(agent_state):
    """
    Extract noise level from a stored quantum agent state.
    
    Args:
        agent_state: Dict containing stored agent covariance state
        
    Returns:
        Noise variance (scalar float)
    """
    try:
        # noise_level is stored in the agent_state dict
        if 'noise_std' in agent_state:
            noise_std = agent_state['noise_std']
            return noise_std ** 2
        elif 'noise_var' in agent_state:
            return agent_state['noise_var']
        else:
            # Fallback: estimate from kernel if available
            print(f"    WARNING: Noise level not found in agent_state, using default 1e-6")
            return 1e-6
    except (KeyError, AttributeError) as e:
        print(f"    WARNING: Error extracting noise level ({e}), using default 1e-6")
        return 1e-6

def predict_rbcm_aggregation_quantum(agent_states, X_test, noise_std, X_train=None, y_train_std=None, consensus_theta=None):
    """
    Make predictions using rBCM (Robust Bayesian Committee Machine) aggregation for quantum GPs.
    
    Implements the rBCM method from the reference paper with aggregation in NOISY (target) space,
    exactly matching the classical baseline and reference paper formulation.
    
    FAIRNESS FIXES:
    ===============
    Concern 1 (Noise Flexibility): Classical extracts per-agent noise, quantum uses global noise_std.
      - Current: Quantum uses fixed global noise_std (applied uniformly to all agents).
      - Limitation: Less flexible than classical's per-agent noise tuning.
      - Mitigation: Both start from same noise_std, ensuring baseline fairness.
    
    Concern 2 (Normalization Scaling): FIXED ✓
      - Issue: Classical scales prior variance by y_train_std² when normalize_y=True.
      - Fix: Quantum now applies same scaling: sigma2_prior *= (y_train_std ** 2)
      - Result: Fair comparison when data is normalized.
    
    Concern 3 (Reference Agent for Prior): FIXED FOR ADMM ✓
      - ADMM consensus: Use trained consensus hyperparameters (theta) for GLOBAL prior variance.
      - This prior is shared by ALL agents (not per-agent).
      - FACT-GP (independent agents): Falls back to per-agent priors if consensus_theta is None.
      - Result: Fair comparison where agents compare against common reference.
    
    Paper Formulation (Equations 10-11):
    - Prior variance: σ²* = k(x*, x*) + σ²_ε (noisy space)
    - Entropy weights: βᵢ = 0.5[log σ²* - log σ²ᵢ(x*)]
    - Mean: μ_rBCM(x*) = σ²_rBCM(x*) · Σ(βᵢ σ⁻²ᵢ(x*) μᵢ(x*))
    - Precision: σ⁻²_rBCM(x*) = Σ(βᵢ σ⁻²ᵢ(x*)) + (1 - Σβᵢ) σ⁻²*
    
    All computations occur in NOISY space (target predictive distribution space), not latent space.
    This matches sklearn's GaussianProcessRegressor behavior with WhiteKernel.
    
    Args:
        agent_states: List of stored quantum agent states
        X_test: Test input data (N_test, D)
        noise_std: Noise standard deviation
        X_train: Training data for prior variance computation (optional)
        y_train_std: Target standard deviation (for Concern 2: fair scaling with classical baseline)
        consensus_theta: Consensus hyperparameters from ADMM training (for ADMM/apxGP methods)
                        If provided, uses global prior variance. If None, falls back to per-agent priors (FACT-GP).
    
    Returns:
        Tuple of (y_pred_mean, y_pred_std, prediction_time)
        where y_pred_std is the standard deviation in noisy space (includes observation noise)
    """
    if y_train_std is None:
        y_train_std = 1.0
    
    print(f"\n{'='*60}")
    print("rBCM PREDICTION AGGREGATION (Quantum GP)")
    print(f"{'='*60}")
    print(f"  Number of local agents: {len(agent_states)}")
    print(f"  Test data: {X_test.shape}")
    print(f"  Aggregation: Entropy-based weighting with prior variance injection")
    print(f"  Observation noise std: {noise_std:.6f}")
    print(f"  Training target std (for scaling fairness): {y_train_std:.6f}")
    
    start_time = time.time()
    
    n_agents = len(agent_states)
    n_test = X_test.shape[0]
    noise_var = noise_std ** 2
    
    # CONCERN 3 FIX (ADMM): Use GLOBAL prior variance from consensus hyperparameters
    # For ADMM-trained agents: all agents share consensus theta, so single global prior
    # For FACT-GP (independent agents): consensus_theta is None, fall back to per-agent priors
    
    if consensus_theta is not None:
        # ADMM/apxGP: Use global prior computed from consensus hyperparameters
        print(f"\n  Computing GLOBAL prior variance from consensus hyperparameters (ADMM/apxGP)...")
        
        try:
            # Get the reference quantum kernel and assign consensus parameters
            reference_agent_state = agent_states[0]
            q_kernel_consensus = reference_agent_state['q_kernel']
            q_kernel_consensus.assign_parameters(consensus_theta)
            
            # Compute global prior using consensus kernel
            K_test_test_global = q_kernel_consensus.evaluate(X_test, X_test)
            k_diag_global = np.diag(K_test_test_global)
            sigma2_prior_global = (k_diag_global + noise_var) * (y_train_std ** 2)
            
            # Use same global prior for all agents
            sigma2_prior_per_agent = np.tile(sigma2_prior_global, (n_agents, 1))  # Shape: (n_agents, n_test)
            
            print(f"    Global prior (scaled by y_std²={y_train_std**2:.6f}): min={np.min(sigma2_prior_global):.6f}, mean={np.mean(sigma2_prior_global):.6f}, max={np.max(sigma2_prior_global):.6f}")
            print(f"    This GLOBAL prior is shared by all {n_agents} agents")
            
        except Exception as e:
            print(f"    WARNING: Consensus prior computation failed ({e}), falling back to per-agent priors")
            consensus_theta = None  # Force fallback
    
    if consensus_theta is None:
        # FACT-GP or fallback: Compute PER-AGENT prior variance using each agent's own kernel
        # This is more principled: agent i compares its uncertainty to its own prior (not a consensus prior)
        # Each agent independently weighs its prediction based on how confident its own prior is
        print(f"\n  Computing PER-AGENT prior variance (FACT-GP or fallback)...")
        
        sigma2_prior_per_agent = []  # List of (n_test,) arrays, one per agent
        
        for i, agent_state in enumerate(agent_states):
            q_kernel = agent_state['q_kernel']
            # CONCERN 2 FIX (Approach 1): Use per-agent y_train_std, not global
            y_agent_std = agent_state['y_train_std']
            try:
                K_test_test_i = q_kernel.evaluate(X_test, X_test)
                k_diag_i = np.diag(K_test_test_i)
                # Agent i's prior variance: σ²_i,* = k_i(x*,x*) + σ²_ε
                sigma2_prior_i_normalized = k_diag_i + noise_var
                # Scale by agent-specific y_train_std² (CONCERN 2 FIX: per-agent scaling)
                sigma2_prior_i = sigma2_prior_i_normalized * (y_agent_std ** 2)
                sigma2_prior_per_agent.append(sigma2_prior_i)
                
                if i == 0:
                    print(f"    Agent 1 prior (scaled by y_std²={y_agent_std**2:.6f}): min={np.min(sigma2_prior_i):.6f}, mean={np.mean(sigma2_prior_i):.6f}, max={np.max(sigma2_prior_i):.6f}")
            except Exception as e:
                print(f"    WARNING: Agent {i+1} prior variance computation failed: {e}")
                sigma2_prior_per_agent.append(np.ones(n_test))
        
        sigma2_prior_per_agent = np.array(sigma2_prior_per_agent)  # Shape: (n_agents, n_test)
        print(f"    Per-agent prior variance computed for all {n_agents} agents")
        print(f"    Prior variance range across all agents: min={np.min(sigma2_prior_per_agent):.6f}, max={np.max(sigma2_prior_per_agent):.6f}")
    
    print(f"\n  Computing predictions from {n_agents} local agents...")
    local_means = []
    local_vars = []  # Will store NOISY variances (latent + noise)
    
    for i, agent_state in enumerate(agent_states):
        agent_id = agent_state['agent_id']
        X_agent = agent_state['X_agent']
        alpha = agent_state['alpha']
        L = agent_state['L']
        q_kernel = agent_state['q_kernel']
        # CONCERN 2 FIX (Approach 1): Use per-agent y_train_std
        y_agent_std = agent_state['y_train_std']
        
        try:
            # Compute K(X_test, X_agent)
            K_test_agent = q_kernel.evaluate(X_test, X_agent)
            
            # Local mean: K_test_agent @ alpha
            mu_i = K_test_agent @ alpha
            local_means.append(mu_i)
            
            # Local latent variance: diag(K_test_test) - ||solve(L, K_test_agent.T)||²
            K_test_test_i = q_kernel.evaluate(X_test, X_test)
            
            v = np.linalg.solve(L, K_test_agent.T)
            sigma2_i_latent = np.diag(K_test_test_i) - np.sum(v**2, axis=0)
            sigma2_i_latent = np.maximum(sigma2_i_latent, 1e-10)
            
            # Convert latent variance to NOISY variance: σ²ᵢ = σ²ᵢ_latent + σ²_ε
            # This gives us the full predictive variance, matching sklearn.GaussianProcessRegressor behavior
            sigma2_i = sigma2_i_latent + noise_var
            
            # CRITICAL FIX: Scale local variance by y_agent_std² to match prior scaling
            # Classical GP: gp.predict() returns variance already scaled to original Y scale
            # Quantum GP: kernel variance is in normalized space, must scale to original space
            # Without this, entropy weights β are massively distorted (log(scaled_prior) - log(unscaled_local))
            # CONCERN 2 FIX (Approach 1): Use per-agent y_train_std, not global
            sigma2_i_scaled = sigma2_i * (y_agent_std ** 2)
            local_vars.append(sigma2_i_scaled)
            
            print(f"    Agent {agent_id+1}: mean ∈ [{mu_i.min():.4f}, {mu_i.max():.4f}], noisy std (scaled by y_std²={y_agent_std**2:.6f}) ∈ [{np.sqrt(sigma2_i_scaled.min()):.4f}, {np.sqrt(sigma2_i_scaled.max()):.4f}]")
            
        except Exception as e:
            print(f"    WARNING: Agent {agent_id+1} prediction failed ({e}), using per-agent prior variance as fallback")
            local_means.append(np.zeros(n_test))
            local_vars.append(np.maximum(sigma2_prior_per_agent[i], 1e-10))
    
    # Convert to arrays: shape (n_agents, n_test)
    local_means = np.array(local_means)
    local_vars = np.array(local_vars)  # NOISY variances for entropy computation and aggregation
    
    # ========== VECTORIZED ENTROPY-WEIGHTED AGGREGATION IN NOISY SPACE ==========
    # Directly aggregate in noisy (target predictive) space, matching the paper and classical baseline
    print(f"\n  Computing entropy-based weights (vectorized, PER-AGENT PRIORS, in noisy space)...")
    
    # Compute entropy weights: βᵢ = 0.5[log σ²_i,* - log σ²ᵢ]
    # CONCERN 3 FIX: Each agent i compares its prior σ²_i,* to its uncertainty σ²ᵢ
    # This is more fair and doesn't depend on any single reference agent
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = 0.5 * (np.log(sigma2_prior_per_agent) - np.log(local_vars))
        beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    
    sum_beta = np.sum(beta, axis=0)
    print(f"  Entropy weights β statistics (per-agent priors):")
    print(f"    min={np.min(beta):.6f}, mean={np.mean(beta):.6f}, max={np.max(beta):.6f}")
    print(f"  Sum Σβᵢ statistics (BEFORE normalization):")
    print(f"    min={np.min(sum_beta):.6f}, mean={np.mean(sum_beta):.6f}, max={np.max(sum_beta):.6f}")
    
    # ISSUE #2 FIX: Normalize entropy weights to ensure Σβᵢ ≤ 1 (paper constraint)
    # Prevents negative prior weights and ensures valid probabilistic interpretation
    # Reference: classical_gp_comparison.py lines 1719-1722
    beta_sum_clipped = np.minimum(sum_beta, 1.0)
    scale_factor = np.where(sum_beta > 1e-10, beta_sum_clipped / (sum_beta + 1e-10), 1.0)
    beta = beta * scale_factor[np.newaxis, :]
    sum_beta = np.sum(beta, axis=0)  # Recompute after normalization
    
    print(f"  Sum Σβᵢ statistics (AFTER normalization):")
    print(f"    min={np.min(sum_beta):.6f}, mean={np.mean(sum_beta):.6f}, max={np.max(sum_beta):.6f}")
    print(f"  Prior weight (1 - Σβᵢ):")
    prior_weight = 1 - sum_beta
    print(f"    min={np.min(prior_weight):.6f}, mean={np.mean(prior_weight):.6f}, max={np.max(prior_weight):.6f}")
    
    # Precision-weighted aggregation (vectorized) in noisy space
    # σ⁻²ᵢ from noisy variances
    sigma2_inv = 1.0 / (local_vars + 1e-10)
    
    # Weighted sum: Σ(βᵢ σ⁻²ᵢ μᵢ)
    weighted_sum = np.sum(beta * sigma2_inv * local_means, axis=0)
    
    # Total precision: Σ(βᵢ σ⁻²ᵢ) + (1 - Σβᵢ) σ⁻²*
    precision_weighted = np.sum(beta * sigma2_inv, axis=0)
    
    # ISSUE #1 FIX: Compute correct prior precision based on method
    # For ADMM (consensus_theta provided): Use GLOBAL prior precision (single value)
    # For FACT-GP (consensus_theta is None): Use AVERAGE prior precision (per-agent approach)
    # Reference: classical_gp_comparison.py line 1727 for correct ADMM formula
    sigma2_prior_inv = 1.0 / (sigma2_prior_per_agent + 1e-10)  # Shape: (n_agents, n_test)
    
    if consensus_theta is not None:
        # ADMM/apxGP: All agents use same global prior, so compute global prior precision directly
        # σ⁻²* = 1 / σ²* (single global value, broadcast across all agents)
        global_prior_precision = (1 - sum_beta) / (sigma2_prior_global + 1e-10)
        prior_precision = global_prior_precision
        print(f"    Using GLOBAL prior precision for ADMM (all agents share consensus prior)")
    else:
        # FACT-GP or fallback: Each agent compares to its own prior, use average
        # σ⁻²* = mean across agents of (1 / σ²_i,*)
        avg_prior_precision = np.mean(sigma2_prior_inv, axis=0)
        prior_precision = (1 - sum_beta) * avg_prior_precision
        print(f"    Using AVERAGE prior precision for per-agent priors (FACT-GP)")
    
    total_precision = precision_weighted + prior_precision
    
    # Mean aggregation (same in both spaces)
    numerator = weighted_sum
    y_pred_mean = np.where(
        total_precision > 1e-10,
        numerator / total_precision,
        np.mean(local_means, axis=0)
    )
    
    # Variance aggregation in noisy space: σ²_rBCM = 1 / σ⁻²_rBCM
    max_agent_var = np.max(local_vars, axis=0)
    y_pred_var = np.where(
        total_precision > 1e-10,
        1.0 / (total_precision + 1e-10),
        max_agent_var  # Conservative: use max agent variance as fallback
    )
    
    y_pred_std = np.sqrt(np.maximum(y_pred_var, 1e-10))
    
    prediction_time = time.time() - start_time
    
    print(f"\n  Prediction completed in: {prediction_time:.4f}s")
    print(f"  Aggregated prediction statistics (in noisy space):")
    print(f"    Mean: {np.mean(y_pred_mean):.6f} ± {np.std(y_pred_mean):.6f}")
    print(f"    Noisy variance (includes observation noise): min={np.min(y_pred_var):.6f}, mean={np.mean(y_pred_var):.6f}, max={np.max(y_pred_var):.6f}")
    print(f"    Noisy std: {np.mean(y_pred_std):.6f}")
    
    return y_pred_mean, y_pred_std, prediction_time


def train_fulldata_gp_quantum(X_train, Y_train, X_test, quantum_kernel_params, 
                             num_qubits, num_layers, noise_std,
                             use_parameter_shift=True, encoding_type='yz_cx',
                             kernel_type='fidelity', measurement='XYZ',
                             outer_kernel='gaussian', outer_kernel_params=None, 
                             regularization=None, y_train_std=None):
    """
    Train a GP on the full dataset using consensus hyperparameters (full-data method).
    
    This method trains a single GP using all training data with the consensus hyperparameters,
    providing a baseline for comparison with distributed methods.
    
    Args:
        X_train: Full training input data (N_train, D)
        Y_train: Full training output data (N_train,)
        X_test: Test input data (N_test, D)
        quantum_kernel_params: Trained consensus parameters
        num_qubits: Number of qubits
        num_layers: Number of layers
        noise_std: Noise standard deviation
        use_parameter_shift: Use parameter shift rule
        encoding_type: Encoding circuit type
        kernel_type: Quantum kernel type
        measurement: Measurement operator
        outer_kernel: Outer kernel type
        outer_kernel_params: Outer kernel parameters
        regularization: Regularization technique
        y_train_std: Target standard deviation (for fair scaling with classical baseline)
    
    Returns:
        Tuple of (y_pred_mean, y_pred_std, prediction_time)
    """
    if y_train_std is None:
        y_train_std = 1.0
    print(f"\n{'='*60}")
    print("FULL-DATA GP PREDICTION METHOD")
    print(f"{'='*60}")
    print(f"  Training data: {X_train.shape}")
    print(f"  Test data: {X_test.shape}")
    print(f"  Using consensus hyperparameters on full dataset")
    
    start_time = time.time()
    
    # Get input dimension
    input_dim = X_train.shape[1] if X_train.ndim > 1 else 1
    
    # Create quantum kernel with consensus parameters
    print(f"\n  Creating quantum kernel with consensus parameters...")
    q_kernel = create_quantum_kernel(
        num_qubits=num_qubits,
        num_features=input_dim,
        num_layers=num_layers,
        use_parameter_shift=use_parameter_shift,
        encoding_type=encoding_type,
        kernel_type=kernel_type,
        measurement=measurement,
        outer_kernel=outer_kernel,
        outer_kernel_params=outer_kernel_params,
        regularization=regularization
    )
    
    # Set the consensus parameters
    q_kernel.assign_parameters(quantum_kernel_params)
    
    # Compute kernel matrices
    print(f"  Computing kernel matrices...")
    
    try:
        # K(X_train, X_train)
        K_train_train = q_kernel.evaluate(X_train, X_train)
        # K(X_test, X_train)
        K_test_train = q_kernel.evaluate(X_test, X_train)
        # K(X_test, X_test)
        K_test_test = q_kernel.evaluate(X_test, X_test)
        
        print(f"    Kernel matrices computed successfully")
    except Exception as e:
        print(f"    ERROR computing kernel matrices: {e}")
        raise
    
    # Add noise for numerical stability
    K_train_train_noisy = K_train_train + (noise_std ** 2) * np.eye(K_train_train.shape[0])
    jitter = 1e-6
    K_train_train_noisy += jitter * np.eye(K_train_train.shape[0])
    
    # Check condition number
    cond_num = np.linalg.cond(K_train_train_noisy)
    print(f"    Training kernel condition number: {cond_num:.2e}")
    
    if cond_num > 1e12:
        print(f"    WARNING: Training kernel matrix is ill-conditioned!")
    
    # Solve for predictions using GP predictive equations
    print(f"  Computing GP predictions...")
    
    try:
        # Solve using Cholesky decomposition
        L_train = np.linalg.cholesky(K_train_train_noisy)
        alpha = np.linalg.solve(L_train, Y_train)
        alpha = np.linalg.solve(L_train.T, alpha)
        
        # Predictive mean: mu_test = K_test_train @ alpha
        y_pred_mean = K_test_train @ alpha
        
        # Predictive variance: sigma²_test = diag(K_test_test) - ||solve(L_train, K_test_train.T)||²
        v = np.linalg.solve(L_train, K_test_train.T)
        y_pred_var = np.diag(K_test_test) - np.sum(v**2, axis=0)
        y_pred_var = np.maximum(y_pred_var, 1e-10)
        
        print(f"    Cholesky decomposition successful")
        
    except np.linalg.LinAlgError as e:
        print(f"    Cholesky decomposition failed: {e}")
        print(f"    Falling back to direct matrix inversion...")
        
        try:
            K_train_inv = np.linalg.inv(K_train_train_noisy)
            alpha = K_train_inv @ Y_train
            y_pred_mean = K_test_train @ alpha
            y_pred_var = np.diag(K_test_test - K_test_train @ K_train_inv @ K_test_train.T)
            y_pred_var = np.maximum(y_pred_var, 1e-10)
            print(f"    Fallback prediction completed")
        except np.linalg.LinAlgError as e2:
            print(f"    ERROR: Both Cholesky and direct inversion failed: {e2}")
            raise RuntimeError("Kernel matrix is singular. Cannot make predictions.")
    
    # CRITICAL FIX: y_pred_var computed above is LATENT variance (function uncertainty) in NORMALIZED space.
    # Must: (1) Scale to original Y scale by y_train_std²
    #       (2) Add observation noise (also scaled to original Y scale) to get TARGET variance for NLPD evaluation
    # This matches sklearn's GaussianProcessRegressor.predict(return_std=True) behavior.
    # The scaling order is critical: σ²_target = (σ²_latent_normalized * y_train_std²) + (σ²_ε_normalized * y_train_std²)
    y_pred_var_latent_scaled = y_pred_var * (y_train_std ** 2)
    noise_var_scaled = (noise_std * y_train_std) ** 2
    y_pred_var_target = y_pred_var_latent_scaled + noise_var_scaled
    y_pred_std_target = np.sqrt(np.maximum(y_pred_var_target, 1e-10))
    
    prediction_time = time.time() - start_time
    
    print(f"  Prediction completed in: {prediction_time:.4f}s")
    print(f"  Prediction statistics:")
    print(f"    Mean range: [{y_pred_mean.min():.4f}, {y_pred_mean.max():.4f}]")
    print(f"    Latent variance (before scaling): [{y_pred_var.min():.6f}, {y_pred_var.max():.6f}]")
    print(f"    Latent variance (after scaling by y_train_std²={y_train_std**2:.6f}): [{y_pred_var_latent_scaled.min():.6f}, {y_pred_var_latent_scaled.max():.6f}]")
    print(f"    Latent std range (normalized): [{np.sqrt(y_pred_var).min():.4f}, {np.sqrt(y_pred_var).max():.4f}]")
    print(f"  CONCERN 3 FIX: Scaled noise from {noise_std**2:.6f} to {noise_var_scaled:.6f}")
    print(f"    Target std range (for NLPD): [{y_pred_std_target.min():.4f}, {y_pred_std_target.max():.4f}]")
    
    return y_pred_mean, y_pred_std_target, prediction_time

def make_predictions_with_method(prediction_method, agent_states, X_train, Y_train, X_test, 
                                 quantum_kernel_params, num_qubits, num_layers, noise_std,
                                 use_parameter_shift=True, encoding_type='yz_cx',
                                 kernel_type='fidelity', measurement='XYZ',
                                 outer_kernel='gaussian', outer_kernel_params=None, 
                                 regularization=None, y_train_std=None):
    """
    Make predictions using the specified aggregation method.
    
    Routes to one of three prediction methods based on `prediction_method`:
    - 'independent': FACT-GP independent agent aggregation (current/default)
    - 'full-data': Train GP on full dataset with consensus parameters
    - 'rbcm': Robust Bayesian Committee Machine entropy-weighted aggregation
    
    Args:
        prediction_method: Method to use ('independent', 'full-data', 'rbcm')
        agent_states: Stored quantum agent states (used by 'independent' and 'rbcm')
        X_train: Full training data (used by 'full-data')
        Y_train: Full training labels (used by 'full-data')
        X_test: Test data
        quantum_kernel_params: Consensus hyperparameters
        num_qubits: Number of qubits
        num_layers: Number of layers in encoding circuit
        noise_std: Noise standard deviation
        use_parameter_shift: Use parameter shift rule
        encoding_type: Encoding circuit type
        kernel_type: Quantum kernel type
        measurement: Measurement operator
        outer_kernel: Outer kernel type
        outer_kernel_params: Outer kernel parameters
        regularization: Regularization technique
        y_train_std: Target standard deviation (for fair scaling with classical baseline)
    
    Returns:
        Tuple of (y_pred_mean, y_pred_std, prediction_time, prediction_method_used)
    """
    # Default y_train_std to 1.0 if not provided
    if y_train_std is None:
        y_train_std = 1.0
    
    # Extract circuit parameters and optimized noise from extended quantum_kernel_params
    # quantum_kernel_params = [theta_1, ..., theta_k, sigma_n^2]
    if len(quantum_kernel_params) > 1:
        potential_noise_var = quantum_kernel_params[-1]
        if potential_noise_var < 10.0:  # Reasonable variance bound
            quantum_kernel_params_circuit = quantum_kernel_params[:-1]
            noise_var_optimized = quantum_kernel_params[-1]
            noise_std_optimized = np.sqrt(noise_var_optimized)
        else:
            quantum_kernel_params_circuit = quantum_kernel_params
            noise_std_optimized = noise_std
    else:
        quantum_kernel_params_circuit = quantum_kernel_params
        noise_std_optimized = noise_std
    
    print(f"\n{'='*70}")
    print(f"PREDICTION WITH METHOD: {prediction_method.upper()}")
    print(f"{'='*70}")
    
    if prediction_method.lower() == 'gpoe':
        # gPoE: Generalized Product of Experts with precision-weighted aggregation (β_i = 1/M)
        # NOTE: predict_gPoE_quantum returns (mean, std_target) where std_target includes noise
        y_pred_mean, y_pred_std = predict_gPoE_quantum(
            agent_states=agent_states,
            X_test=X_test,
            noise_std=noise_std_optimized,
            y_train_std=y_train_std
        )
        prediction_time = 0  # Approximate (included in function output)
        method_used = "gPoE (Precision-Weighted Aggregation)"

        
    elif prediction_method.lower() == 'full-data':
        # Full-data: Train GP on entire dataset with consensus parameters
        y_pred_mean, y_pred_std, prediction_time = train_fulldata_gp_quantum(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            quantum_kernel_params=quantum_kernel_params_circuit,
            num_qubits=num_qubits,
            num_layers=num_layers,
            noise_std=noise_std_optimized,
            use_parameter_shift=use_parameter_shift,
            encoding_type=encoding_type,
            kernel_type=kernel_type,
            measurement=measurement,
            outer_kernel=outer_kernel,
            outer_kernel_params=outer_kernel_params,
            regularization=regularization,
            y_train_std=y_train_std
        )
        method_used = "Full-Data"
        
    elif prediction_method.lower() == 'rbcm':
        # rBCM: Entropy-weighted aggregation of agent predictions
        # For ADMM-trained agents: use consensus hyperparameters for GLOBAL prior variance
        y_pred_mean, y_pred_std, prediction_time = predict_rbcm_aggregation_quantum(
            agent_states=agent_states,
            X_test=X_test,
            noise_std=noise_std_optimized,
            X_train=X_train,
            y_train_std=y_train_std,
            consensus_theta=quantum_kernel_params_circuit  # Use consensus hyperparameters for global prior
        )
        method_used = "rBCM"
        
    else:
        raise ValueError(f"Unknown prediction method: {prediction_method}. "
                        f"Must be one of: 'gPoE', 'full-data', 'rbcm'")
    
    return y_pred_mean, y_pred_std, prediction_time, method_used

def compute_and_store_agent_covariance_states(agent_data_splits, quantum_kernel_params, num_qubits, 
                                             num_layers, noise_std, use_parameter_shift=True, 
                                             encoding_type='yz_cx', kernel_type='fidelity', 
                                             measurement='XYZ', outer_kernel='gaussian', 
                                             outer_kernel_params=None, regularization=None):
    """
    Compute and store local covariance information for each agent after ADMM training.
    This matches FACT-GP approach: each agent stores its local trained state for later prediction.
    
    Args:
        agent_data_splits: List of (X_agent, Y_agent) tuples for each agent
        quantum_kernel_params: Consensus hyperparameters (z_best) from ADMM
        num_qubits: Number of qubits
        num_layers: Number of layers in encoding circuit
        noise_std: Observation noise standard deviation
        use_parameter_shift: Whether to use parameter shift rule
        encoding_type: Type of encoding circuit
        kernel_type: Type of quantum kernel
        measurement: Measurement operator for ProjectedQuantumKernel
        outer_kernel: Outer kernel type for ProjectedQuantumKernel
        outer_kernel_params: Parameters for the outer kernel
        regularization: Regularization technique for ProjectedQuantumKernel
    
    Returns:
        list: List of dictionaries containing stored agent states:
              Each dict has: {'X_agent': X, 'Y_agent': Y, 'alpha': alpha, 'L': L, 'agent_id': i}
    """
    print(f"\n{'='*60}")
    print("STORING LOCAL COVARIANCE STATES FOR EACH AGENT")
    print(f"{'='*60}")
    
    # Extract circuit parameters and optimized noise from extended quantum_kernel_params
    # quantum_kernel_params = [theta_1, ..., theta_k, sigma_n^2]
    if len(quantum_kernel_params) > 1:
        # Check if extended format (has noise variance)
        potential_noise_var = quantum_kernel_params[-1]
        if potential_noise_var < 10.0:  # Reasonable variance bound
            quantum_kernel_params_circuit = quantum_kernel_params[:-1]
            noise_var_optimized = quantum_kernel_params[-1]
            noise_std_optimized = np.sqrt(noise_var_optimized)
            print(f"Extended hyperparameters detected. Using circuit params and optimized noise: {noise_std_optimized:.6f}")
        else:
            quantum_kernel_params_circuit = quantum_kernel_params
            noise_std_optimized = noise_std
    else:
        quantum_kernel_params_circuit = quantum_kernel_params
        noise_std_optimized = noise_std
    
    agent_states = []
    
    for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
        print(f"\nAgent {i+1}/{len(agent_data_splits)}:")
        print(f"  Data shape: X={X_agent.shape}, Y={Y_agent.shape}")
        
        # Compute per-agent target std (CONCERN 2 FIX: Approach 1)
        # Each agent standardizes its own data independently, matching classical GP behavior
        y_agent_std = np.std(Y_agent) if len(Y_agent) > 0 else 1.0
        print(f"  Per-agent target std: {y_agent_std:.6f}")
        
        # Get input dimension
        input_dim = X_agent.shape[1] if X_agent.ndim > 1 else 1
        
        # Create quantum kernel with consensus parameters
        q_kernel = create_quantum_kernel(
            num_qubits=num_qubits,
            num_features=input_dim,
            num_layers=num_layers,
            use_parameter_shift=use_parameter_shift,
            encoding_type=encoding_type,
            kernel_type=kernel_type,
            measurement=measurement,
            outer_kernel=outer_kernel,
            outer_kernel_params=outer_kernel_params,
            regularization=regularization
        )
        
        # Set consensus parameters (circuit only)
        q_kernel.assign_parameters(quantum_kernel_params_circuit)
        
        # Compute K(X_agent, X_agent)
        try:
            K_agent = q_kernel.evaluate(X_agent, X_agent)
            print(f"  K(X_agent, X_agent) computed: {K_agent.shape}")
        except Exception as e:
            print(f"  ❌ Error computing kernel: {e}")
            raise
        
        # Add noise and jitter for numerical stability (use optimized noise if available)
        K_agent_noisy = K_agent + (noise_std_optimized**2) * np.eye(K_agent.shape[0])
        jitter = 1e-6
        K_agent_noisy += jitter * np.eye(K_agent.shape[0])
        
        # Compute Cholesky decomposition
        try:
            L = np.linalg.cholesky(K_agent_noisy)
            print(f"  Cholesky decomposition successful")
        except np.linalg.LinAlgError as e:
            print(f"  ❌ Cholesky decomposition failed: {e}")
            raise
        
        # Compute alpha = (K + noise*I)^{-1} @ Y
        try:
            alpha = np.linalg.solve(L, Y_agent)
            alpha = np.linalg.solve(L.T, alpha)
            print(f"  Alpha coefficients computed: {alpha.shape}")
        except np.linalg.LinAlgError as e:
            print(f"  ❌ Failed to compute alpha: {e}")
            raise
        
        # Store agent state
        agent_state = {
            'agent_id': i,
            'X_agent': X_agent,
            'Y_agent': Y_agent,
            'alpha': alpha,
            'L': L,
            'K_agent': K_agent,
            'q_kernel': q_kernel,
            'y_train_std': y_agent_std  # Per-agent target std (CONCERN 2 FIX)
        }
        agent_states.append(agent_state)
        print(f"  ✓ Agent {i+1} state stored")
    
    print(f"\n✅ All {len(agent_states)} agent covariance states stored successfully")
    print(f"{'='*60}\n")
    
    return agent_states

def predict_gPoE_quantum(agent_states, X_test, noise_std, y_train_std=None):
    """
    Make predictions using QUANTUM gPoE (Generalized Product of Experts).
    
    Implements gPoE with equal weights β_i = 1/M precision-weighted aggregation.
    
    Paper formula (equations 8-9):
    - Precision: σ^-2_gPoE(x*) = Σ(β_i σ_i^-2(x*))  where β_i = 1/M
    - Mean: μ_gPoE(x*) = σ²_gPoE(x*) * Σ(β_i σ_i^-2(x*) μ_i(x*))
    
    CRITICAL: Returns full TARGET variance (latent + observation noise) for correct
    NLPD evaluation.
    
    Args:
        agent_states: List of stored agent states from compute_and_store_agent_covariance_states()
        X_test: Test input data (N_test, D)
        noise_std: Observation noise standard deviation
        y_train_std: Target standard deviation (for fair scaling with classical baseline)
    
    Returns:
        tuple: (y_pred_mean, y_pred_std_target) where y_pred_std_target includes observation noise
    """
    if y_train_std is None:
        y_train_std = 1.0
    
    print(f"\nMaking gPoE (PRECISION-WEIGHTED) predictions using stored covariance states...")
    print(f"  Test data: {X_test.shape}")
    print(f"  Number of agents: {len(agent_states)}")
    print(f"  Aggregation: gPoE with equal weights β_i = 1/{len(agent_states)}")
    print(f"  Training target std (for scaling fairness): {y_train_std:.6f}")
    
    # Pre-compute K_test_test ONCE
    print(f"  Pre-computing K_test_test (will be reused for all agents)...")
    agent_ref = agent_states[0]
    q_kernel_ref = agent_ref['q_kernel']
    try:
        K_test_test_shared = q_kernel_ref.evaluate(X_test, X_test)
        k_diag_test_test = np.diag(K_test_test_shared)
        print(f"    K_test_test shape: {K_test_test_shared.shape}")
    except Exception as e:
        print(f"    WARNING: Could not pre-compute K_test_test: {e}")
        K_test_test_shared = None
        k_diag_test_test = None
    
    local_means = []
    local_variances = []
    
    # Collect predictions and variances from each agent
    M = len(agent_states)
    for agent_state in agent_states:
        agent_id = agent_state['agent_id']
        X_agent = agent_state['X_agent']
        alpha = agent_state['alpha']
        L = agent_state['L']
        q_kernel = agent_state['q_kernel']
        
        print(f"\n  Agent {agent_id+1} prediction...")
        
        # Compute K(X_test, X_agent)
        try:
            K_test_agent = q_kernel.evaluate(X_test, X_agent)
        except Exception as e:
            print(f"    ❌ Error computing kernel: {e}")
            raise
        
        # Local mean: K_test_agent @ alpha
        y_mean_local = K_test_agent @ alpha
        local_means.append(y_mean_local)
        
        # Local latent variance
        try:
            if K_test_test_shared is not None:
                k_diag = k_diag_test_test
            else:
                K_test_test_i = q_kernel.evaluate(X_test, X_test)
                k_diag = np.diag(K_test_test_i)
            
            v = np.linalg.solve(L, K_test_agent.T)
            y_var_local = k_diag - np.sum(v**2, axis=0)
            y_var_local = np.maximum(y_var_local, 1e-10)
            
            local_variances.append(y_var_local)
            
        except Exception as e:
            print(f"    ❌ Error computing variance: {e}")
            raise
    
    # gPoE PRECISION-WEIGHTED AGGREGATION (from paper equations 8-9)
    print(f"\nAggregating using gPoE precision-weighting...")
    
    local_means = np.array(local_means)      # Shape: (M, n_test)
    local_variances = np.array(local_variances)  # Shape: (M, n_test)
    
    # CONCERN 3 FIX: Scale local variances by y_train_std² to match target-space scales
    # Quantum kernels operate in normalized space, must scale to original target scale
    # for fair precision-weighting against classical GPs
    local_variances_scaled = local_variances * (y_train_std ** 2)
    
    # gPoE weight: β_i = 1/M
    beta_i = 1.0 / M
    
    # Compute precisions: σ_i^-2 = 1 / σ_i² (using SCALED variances)
    precisions = 1.0 / (local_variances_scaled + 1e-10)  # Shape: (M, n_test)
    
    # Weighted precisions: β_i * σ_i^-2
    weighted_precisions = beta_i * precisions  # Shape: (M, n_test)
    
    # Total precision: σ^-2_gPoE = Σ(β_i σ_i^-2)
    total_precision = np.sum(weighted_precisions, axis=0)  # Shape: (n_test,)
    
    # Weighted sum for mean: Σ(β_i σ_i^-2 μ_i)
    weighted_sum = np.sum(beta_i * precisions * local_means, axis=0)  # Shape: (n_test,)
    
    # Final predictions (in latent space after aggregation)
    y_pred_var_latent = 1.0 / (total_precision + 1e-10)  # σ²_gPoE
    y_pred_mean = y_pred_var_latent * weighted_sum  # μ_gPoE
    
    # CONCERN 3 FIX: Convert to target variance (add observation noise, scaled to original space)
    # noise_std was in normalized space, must scale by y_train_std² for target-space NLPD
    noise_var_scaled = (noise_std * y_train_std) ** 2
    y_pred_var_target = y_pred_var_latent + noise_var_scaled
    y_pred_std_target = np.sqrt(np.maximum(y_pred_var_target, 1e-10))
    
    print(f"  Latent variance range: [{np.min(y_pred_var_latent):.6f}, {np.max(y_pred_var_latent):.6f}]")
    print(f"  Target variance range (latent + scaled noise): [{np.min(y_pred_var_target):.6f}, {np.max(y_pred_var_target):.6f}]")
    print(f"  Final mean range: [{np.min(y_pred_mean):.4f}, {np.max(y_pred_mean):.4f}]")
    print(f"  Final target std range: [{np.min(y_pred_std_target):.4f}, {np.max(y_pred_std_target):.4f}]")
    print(f"  CONCERN 3 FIX: Scaled local variances by y_train_std²={y_train_std**2:.6f}")
    print(f"  CONCERN 3 FIX: Scaled noise from {noise_std**2:.6f} to {noise_var_scaled:.6f}")
    
    return y_pred_mean, y_pred_std_target

def predict_quantum_gp_with_agent_states(agent_states, X_test, noise_std, y_train_std=None):
    """
    Make predictions using stored agent covariance states (FACT-GP approach).
    Each agent makes local predictions independently, then results are aggregated.
    
    CRITICAL: Returns full TARGET variance (latent + observation noise) for correct
    uncertainty quantification and NLPD evaluation in downstream evaluation pipeline.
    
    PERFORMANCE OPTIMIZATION: K_test_test is computed once and reused across all agents
    since X_test and kernel parameters are identical for all agents.
    
    Args:
        agent_states: List of stored agent states from compute_and_store_agent_covariance_states()
        X_test: Test input data (N_test, D)
        noise_std: Observation noise standard deviation
        y_train_std: Target standard deviation (for fair scaling with classical baseline)
    
    Returns:
        tuple: (y_pred_mean, y_pred_std_target) where y_pred_std_target includes observation noise
               for correct NLPD calculation: std_target = √(latent_var + noise_var)
    """
    if y_train_std is None:
        y_train_std = 1.0
    
    print(f"\nMaking FACT-GP predictions using stored covariance states...")
    print(f"  Test data: {X_test.shape}")
    print(f"  Number of agents: {len(agent_states)}")
    print(f"  Training target std (for scaling fairness): {y_train_std:.6f}")
    
    # Pre-compute K_test_test ONCE to avoid redundant evaluations
    # Since all agents use the same X_test and consensus kernel params,
    # we only need to compute this matrix once and reuse its diagonal
    print(f"  Pre-computing K_test_test (will be reused for all agents)...")
    agent_ref = agent_states[0]
    q_kernel_ref = agent_ref['q_kernel']
    try:
        K_test_test_shared = q_kernel_ref.evaluate(X_test, X_test)
        k_diag_test_test = np.diag(K_test_test_shared)
        print(f"    K_test_test shape: {K_test_test_shared.shape}, will reuse {len(agent_states)} times")
    except Exception as e:
        print(f"    WARNING: Could not pre-compute K_test_test: {e}")
        K_test_test_shared = None
        k_diag_test_test = None
    
    local_means = []
    local_variances = []
    
    # Collect predictions from each agent
    for agent_state in agent_states:
        agent_id = agent_state['agent_id']
        X_agent = agent_state['X_agent']
        alpha = agent_state['alpha']
        L = agent_state['L']
        q_kernel = agent_state['q_kernel']
        # CONCERN 2 FIX (Approach 1): Use per-agent y_train_std
        y_agent_std = agent_state['y_train_std']
        
        print(f"\n  Predicting from Agent {agent_id+1}...")
        
        # Compute K(X_test, X_agent) for this agent
        try:
            K_test_agent = q_kernel.evaluate(X_test, X_agent)
            print(f"    K(X_test, X_agent) computed: {K_test_agent.shape}")
        except Exception as e:
            print(f"    ❌ Error computing kernel: {e}")
            raise
        
        # Compute local mean: K_test_agent @ alpha
        y_mean_local = K_test_agent @ alpha
        local_means.append(y_mean_local)
        
        # Compute local latent variance efficiently using stored L and pre-computed K_test_test
        # variance = diag(K_test_test) - ||solve(L, K_test_agent.T)||²
        try:
            # Use pre-computed K_test_test if available, otherwise compute for this agent
            if K_test_test_shared is not None:
                k_diag = k_diag_test_test
            else:
                K_test_test_i = q_kernel.evaluate(X_test, X_test)
                k_diag = np.diag(K_test_test_i)
            
            # Solve L @ v = K_test_agent.T for v
            v = np.linalg.solve(L, K_test_agent.T)
            
            # Latent Variance = diag(K_test_test) - ||v||²
            y_var_local = k_diag - np.sum(v**2, axis=0)
            y_var_local = np.maximum(y_var_local, 1e-10)  # Ensure non-negative
            
            local_variances.append(y_var_local)
            print(f"    Local mean range: [{y_mean_local.min():.4f}, {y_mean_local.max():.4f}]")
            print(f"    Local latent std range: [{np.sqrt(y_var_local.min()):.4f}, {np.sqrt(y_var_local.max()):.4f}]")
            
        except Exception as e:
            print(f"    ❌ Error computing variance: {e}")
            raise
    
    # Aggregate predictions using FACT-GP formula
    print(f"\nAggregating predictions from {len(agent_states)} agents...")
    
    # Step 1: Simple arithmetic mean of predictions
    y_pred_mean = np.mean(local_means, axis=0)
    print(f"  Mean aggregation: {y_pred_mean.shape}")
    
    # Step 2: Aggregate uncertainties with per-agent scaling (matching classical approach)
    # CONCERN 2 FIX: Apply per-agent y_train_std² scaling before aggregation, matching classical
    # This ensures each agent's variance contribution is scaled by ITS OWN training data standard deviation
    scaled_local_variances = []
    for i, var in enumerate(local_variances):
        y_agent_std = agent_states[i]['y_train_std']
        scaled_local_variances.append(var * (y_agent_std ** 2))
    scaled_local_variances = np.array(scaled_local_variances)
    
    # epistemic_variance = mean of (per-agent scaled) local variances
    epistemic_variance = np.mean(scaled_local_variances, axis=0)
    
    # disagreement_variance = variance across local predictions (model disagreement)
    disagreement_variance = np.var(local_means, axis=0)
    
    # Total latent variance = epistemic + disagreement (FACT-GP formula)
    y_pred_var_latent = epistemic_variance + disagreement_variance
    
    # CRITICAL FIX: Convert to full TARGET variance for NLPD evaluation
    # σ²_target = σ²_latent_scaled + σ²_ε_scaled
    # CONCERN 3 FIX: Scale noise_std by global y_train_std for target-space NLPD
    # (latent variance already scaled by per-agent y_train_std² factors)
    noise_var_scaled = (noise_std * y_train_std) ** 2
    y_pred_var_target = y_pred_var_latent + noise_var_scaled
    y_pred_std_target = np.sqrt(np.maximum(y_pred_var_target, 1e-10))
    
    print(f"\n  Aggregation complete:")
    print(f"    Epistemic variance (mean of scaled local variances): {np.mean(epistemic_variance):.6f}")
    print(f"    Disagreement variance: {np.mean(disagreement_variance):.6f}")
    print(f"  CONCERN 3 FIX: Scaled noise from {noise_std**2:.6f} to {noise_var_scaled:.6f}")
    print(f"  Final target variance range: [{np.min(y_pred_var_target):.6f}, {np.max(y_pred_var_target):.6f}]")
    
    print(f"  Epistemic latent variance range (after per-agent scaling): [{epistemic_variance.min():.6f}, {epistemic_variance.max():.6f}]")
    print(f"  Disagreement variance range: [{disagreement_variance.min():.6f}, {disagreement_variance.max():.6f}]")
    print(f"  Total latent variance range (epistemic + disagreement): [{y_pred_var_latent.min():.6f}, {y_pred_var_latent.max():.6f}]")
    print(f"  Target variance (latent + noise) range: [{y_pred_var_target.min():.6f}, {y_pred_var_target.max():.6f}]")
    print(f"  Final mean range: [{y_pred_mean.min():.4f}, {y_pred_mean.max():.4f}]")
    print(f"  Final target std range (for NLPD): [{y_pred_std_target.min():.4f}, {y_pred_std_target.max():.4f}]")
    
    return y_pred_mean, y_pred_std_target

def predict_quantum_gp(X_train, Y_train, X_test, quantum_kernel_params, 
                      num_qubits, num_layers, noise_std, 
                      use_parameter_shift=True, encoding_type='yz_cx', 
                      kernel_type='fidelity', measurement='XYZ', outer_kernel='gaussian', outer_kernel_params=None, regularization=None):
    """
    Make predictions using a trained quantum Gaussian Process.
    
    Args:
        X_train: Training input data (N_train, D)
        Y_train: Training output data (N_train,)
        X_test: Test input data (N_test, D)
        quantum_kernel_params: Trained quantum kernel parameters
        num_qubits: Number of qubits
        num_layers: Number of layers in encoding circuit
        noise_std: Observation noise standard deviation
        use_parameter_shift: Whether to use parameter shift rule
        encoding_type: Type of encoding circuit
        kernel_type: Type of quantum kernel
        measurement: Measurement operator for ProjectedQuantumKernel
        outer_kernel: Outer kernel type for ProjectedQuantumKernel
        outer_kernel_params: Parameters for the outer kernel
        regularization: Regularization technique for ProjectedQuantumKernel
    
    Returns:
        tuple: (y_pred_mean, y_pred_var, K_train_train, K_test_train, K_test_test)
    """
    print(f"Making predictions with quantum GP...")
    print(f"  Training data: {X_train.shape}")
    print(f"  Test data: {X_test.shape}")
    print(f"  Kernel parameters: {quantum_kernel_params}")
    
    # Get input dimension
    input_dim = X_train.shape[1] if X_train.ndim > 1 else 1
    
    # Create quantum kernel with trained parameters
    q_kernel = create_quantum_kernel(
        num_qubits=num_qubits,
        num_features=input_dim, 
        num_layers=num_layers,
        use_parameter_shift=use_parameter_shift,
        encoding_type=encoding_type,
        kernel_type=kernel_type,
        measurement=measurement,
        outer_kernel=outer_kernel,
        outer_kernel_params=outer_kernel_params,
        regularization=regularization
    )
    
    # Set the trained parameters
    q_kernel.assign_parameters(quantum_kernel_params)
    
    # Compute kernel matrices
    print("Computing kernel matrices...")
    
    # K(X_train, X_train)
    start_time = time.time()
    K_train_train = q_kernel.evaluate(X_train, X_train)
    print(f"  K_train_train computation: {time.time() - start_time:.3f}s")
    
    # K(X_test, X_train)  
    start_time = time.time()
    K_test_train = q_kernel.evaluate(X_test, X_train)
    print(f"  K_test_train computation: {time.time() - start_time:.3f}s")
    
    # K(X_test, X_test) for predictive variance
    start_time = time.time()
    K_test_test = q_kernel.evaluate(X_test, X_test)
    print(f"  K_test_test computation: {time.time() - start_time:.3f}s")
    
    # Add noise to training kernel matrix for numerical stability
    K_train_train_noisy = K_train_train + (noise_std**2) * np.eye(K_train_train.shape[0])
    
    # Add small jitter for numerical stability
    jitter = 1e-6
    K_train_train_noisy += jitter * np.eye(K_train_train.shape[0])
    
    # Check kernel matrix condition number
    cond_num = np.linalg.cond(K_train_train_noisy)
    print(f"  Training kernel condition number: {cond_num:.2e}")
    
    if cond_num > 1e12:
        print("  Warning: Training kernel matrix is ill-conditioned!")
    
    # Solve for predictions using GP predictive equations
    print("Computing GP predictions...")
    
    try:
        # Solve K_train_train_noisy @ alpha = Y_train
        # Using Cholesky decomposition for numerical stability
        L_train = np.linalg.cholesky(K_train_train_noisy)
        alpha = np.linalg.solve(L_train, Y_train)
        alpha = np.linalg.solve(L_train.T, alpha)
        
        # Predictive mean: mu_test = K_test_train @ alpha
        y_pred_mean = K_test_train @ alpha
        
        # Predictive variance: sigma²_test = K_test_test - K_test_train @ K_train_train^{-1} @ K_test_train^T
        # Efficient computation: v = solve(L_train, K_test_train.T)
        v = np.linalg.solve(L_train, K_test_train.T)
        y_pred_var = np.diag(K_test_test) - np.sum(v**2, axis=0)
        
        # Ensure variance is non-negative (numerical issues can cause small negative values)
        y_pred_var = np.maximum(y_pred_var, 1e-10)
        
        print(f"  Prediction completed successfully")
        print(f"  Mean prediction range: [{y_pred_mean.min():.4f}, {y_pred_mean.max():.4f}]")
        print(f"  Predictive std range: [{np.sqrt(y_pred_var.min()):.4f}, {np.sqrt(y_pred_var.max()):.4f}]")
        
    except np.linalg.LinAlgError as e:
        print(f"  Cholesky decomposition failed: {e}")
        print("  Falling back to direct matrix inversion...")
        
        # Fallback: direct matrix inversion (less numerically stable)
        try:
            K_train_inv = np.linalg.inv(K_train_train_noisy)
            alpha = K_train_inv @ Y_train
            y_pred_mean = K_test_train @ alpha
            y_pred_var = np.diag(K_test_test - K_test_train @ K_train_inv @ K_test_train.T)
            y_pred_var = np.maximum(y_pred_var, 1e-10)
            print(f"  Fallback prediction completed")
        except np.linalg.LinAlgError as e2:
            print(f"  Matrix inversion also failed: {e2}")
            raise RuntimeError("Both Cholesky and direct inversion failed. Kernel matrix is singular.")
    
    return y_pred_mean, y_pred_var, K_train_train, K_test_train, K_test_test

def k_fold_cross_validation_consensus(X_train, Y_train, consensus_params, num_qubits, num_layers, noise_std, 
                                    k_folds=5, use_parameter_shift=True, encoding_type='yz_cx', 
                                    kernel_type='fidelity', measurement='XYZ', outer_kernel='gaussian', 
                                    outer_kernel_params=None, regularization=None, random_seed=42):
    """
    Perform k-fold cross-validation to evaluate consensus hyperparameters (z) using NLPD score.
    This evaluates the global consensus parameters, not individual agent parameters.
    
    Args:
        X_train: Combined training input data from all agents
        Y_train: Combined training output data from all agents  
        consensus_params: Consensus hyperparameters (z) to evaluate
        k_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducible fold splitting
    
    Returns:
        dict: CV results with mean NLPD, std NLPD, and fold-wise results
    """
    from sklearn.model_selection import KFold
    
    # Extract circuit parameters and optimized noise from extended consensus_params
    # consensus_params = [theta_1, ..., theta_k, sigma_n^2]
    if len(consensus_params) > 1:
        # Check if this looks like extended format (last dim different from noise_std scale)
        potential_noise_var = consensus_params[-1]
        # If consensus_params looks extended, extract parts
        if potential_noise_var < 10.0 and potential_noise_var != noise_std:  # Reasonable bounds for variance
            consensus_params_circuit = consensus_params[:-1]
            noise_var_optimized = consensus_params[-1]
            noise_std_optimized = np.sqrt(noise_var_optimized)
        else:
            # Original format
            consensus_params_circuit = consensus_params
            noise_std_optimized = noise_std
    else:
        consensus_params_circuit = consensus_params
        noise_std_optimized = noise_std
    
    # Create k-fold splits
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    fold_nlpds = []
    fold_r2s = []
    fold_rmses = []
    
    input_dim = X_train.shape[1] if X_train.ndim > 1 else 1
    
    print(f"    Performing {k_folds}-fold CV on consensus params (circuit): {consensus_params_circuit}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        try:
            # Split data for this fold
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            Y_fold_train, Y_fold_val = Y_train[train_idx], Y_train[val_idx]
            
            # Make predictions on validation fold using predict_quantum_gp
            y_pred_mean, y_pred_var, _, _, _ = predict_quantum_gp(
                X_train=X_fold_train,
                Y_train=Y_fold_train,
                X_test=X_fold_val,
                quantum_kernel_params=consensus_params_circuit,
                num_qubits=num_qubits,
                num_layers=num_layers,
                noise_std=noise_std_optimized,
                use_parameter_shift=use_parameter_shift,
                encoding_type=encoding_type,
                kernel_type=kernel_type,
                measurement=measurement,
                outer_kernel=outer_kernel,
                outer_kernel_params=outer_kernel_params,
                regularization=regularization
            )
            
            # Calculate NLPD for this fold
            residuals = Y_fold_val - y_pred_mean
            eps = 1e-10
            Y_pred_var_safe = np.maximum(y_pred_var, eps)
            
            log_2pi = np.log(2 * np.pi)
            nlpd_per_point = 0.5 * log_2pi + 0.5 * np.log(Y_pred_var_safe) + 0.5 * (residuals**2 / Y_pred_var_safe)
            fold_nlpd = np.mean(nlpd_per_point)
            
            # Also calculate R² and RMSE for additional metrics
            fold_r2 = r2_score(Y_fold_val, y_pred_mean)
            fold_rmse = np.sqrt(mean_squared_error(Y_fold_val, y_pred_mean))
            
            fold_nlpds.append(fold_nlpd)
            fold_r2s.append(fold_r2)
            fold_rmses.append(fold_rmse)
            
            print(f"      Fold {fold_idx+1}: NLPD={fold_nlpd:.4f}, R²={fold_r2:.4f}, RMSE={fold_rmse:.4f}")
            
        except Exception as e:
            print(f"      Warning: Fold {fold_idx+1} failed: {e}")
            # Use a penalty score for failed folds
            fold_nlpds.append(float('inf'))
            fold_r2s.append(-float('inf'))
            fold_rmses.append(float('inf'))
    
    # Calculate cross-validation statistics
    valid_nlpds = [nlpd for nlpd in fold_nlpds if not np.isinf(nlpd)]
    
    if len(valid_nlpds) >= k_folds // 2:  # At least half the folds must succeed
        mean_nlpd = np.mean(valid_nlpds)
        std_nlpd = np.std(valid_nlpds)
        mean_r2 = np.mean([r2 for r2, nlpd in zip(fold_r2s, fold_nlpds) if not np.isinf(nlpd)])
        mean_rmse = np.mean([rmse for rmse, nlpd in zip(fold_rmses, fold_nlpds) if not np.isinf(nlpd)])
    else:
        # Too many folds failed
        mean_nlpd = float('inf')
        std_nlpd = float('inf')
        mean_r2 = -float('inf')
        mean_rmse = float('inf')
    
    return {
        'mean_nlpd': mean_nlpd,
        'std_nlpd': std_nlpd,
        'mean_r2': mean_r2,
        'mean_rmse': mean_rmse,
        'fold_nlpds': fold_nlpds,
        'fold_r2s': fold_r2s,
        'fold_rmses': fold_rmses,
        'valid_folds': len(valid_nlpds),
        'total_folds': k_folds
    }

def k_fold_cross_validation_local(agent_id, X_local, Y_local, consensus_params, num_qubits, num_layers, 
                                  noise_std, k_folds=5, use_parameter_shift=True, encoding_type='yz_cx', 
                                  kernel_type='fidelity', measurement='XYZ', outer_kernel='gaussian', 
                                  outer_kernel_params=None, regularization=None, random_seed=42):
    """
    Perform k-fold cross-validation on LOCAL agent data (distributed approach).
    This evaluates consensus hyperparameters using ONLY the agent's local data.
    No data sharing - respects distributed constraint.
    
    Args:
        agent_id: Agent identifier
        X_local: Local training input data for this agent only
        Y_local: Local training output data for this agent only
        consensus_params: Consensus hyperparameters (z) to evaluate
        k_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducible fold splitting
    
    Returns:
        dict: CV results with mean NLPD, std NLPD, and fold-wise results
    """
    from sklearn.model_selection import KFold
    
    # Extract circuit parameters and optimized noise from extended consensus_params
    # consensus_params = [theta_1, ..., theta_k, sigma_n^2]
    # Check if last element is a noise variance (small positive value, not an angle)
    if len(consensus_params) > 1:
        last_elem = consensus_params[-1]
        # Noise variance should be small (< 10) and different from a circuit parameter
        # Circuit params are typically in [0, 2π] or [-π, π]
        if 0 < last_elem < 10.0:
            # Extended format with noise: extract circuit params and noise variance
            consensus_params_circuit = consensus_params[:-1]
            noise_var_optimized = last_elem
            noise_std_optimized = np.sqrt(noise_var_optimized)
            print(f"    Extracted optimized noise: {noise_std_optimized:.6f} (from consensus_params[-1]={noise_var_optimized:.6f})")
        else:
            # Original format: only circuit parameters (last elem is circuit param, not noise)
            consensus_params_circuit = consensus_params
            noise_std_optimized = noise_std
    else:
        # Single parameter: treat as circuit parameter
        consensus_params_circuit = consensus_params
        noise_std_optimized = noise_std
    
    # Create k-fold splits on LOCAL data only
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    fold_nlpds = []
    fold_r2s = []
    fold_rmses = []
    
    input_dim = X_local.shape[1] if X_local.ndim > 1 else 1
    
    # Important: CV uses only this agent's local data
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_local)):
        try:
            # Split LOCAL data for this fold
            X_fold_train, X_fold_val = X_local[train_idx], X_local[val_idx]
            Y_fold_train, Y_fold_val = Y_local[train_idx], Y_local[val_idx]
            
            # Make predictions using consensus hyperparameters on this agent's local fold
            y_pred_mean, y_pred_var, _, _, _ = predict_quantum_gp(
                X_train=X_fold_train,
                Y_train=Y_fold_train,
                X_test=X_fold_val,
                quantum_kernel_params=consensus_params_circuit,
                num_qubits=num_qubits,
                num_layers=num_layers,
                noise_std=noise_std_optimized,
                use_parameter_shift=use_parameter_shift,
                encoding_type=encoding_type,
                kernel_type=kernel_type,
                measurement=measurement,
                outer_kernel=outer_kernel,
                outer_kernel_params=outer_kernel_params,
                regularization=regularization
            )
            
            # Calculate NLPD for this fold
            residuals = Y_fold_val - y_pred_mean
            eps = 1e-10
            Y_pred_var_safe = np.maximum(y_pred_var, eps)
            
            log_2pi = np.log(2 * np.pi)
            nlpd_per_point = 0.5 * log_2pi + 0.5 * np.log(Y_pred_var_safe) + 0.5 * (residuals**2 / Y_pred_var_safe)
            fold_nlpd = np.mean(nlpd_per_point)
            
            # Also calculate R² and RMSE for additional metrics
            fold_r2 = r2_score(Y_fold_val, y_pred_mean)
            fold_rmse = np.sqrt(mean_squared_error(Y_fold_val, y_pred_mean))
            
            fold_nlpds.append(fold_nlpd)
            fold_r2s.append(fold_r2)
            fold_rmses.append(fold_rmse)
            
        except Exception as e:
            # Use a penalty score for failed folds
            fold_nlpds.append(float('inf'))
            fold_r2s.append(-float('inf'))
            fold_rmses.append(float('inf'))
    
    # Calculate cross-validation statistics
    valid_nlpds = [nlpd for nlpd in fold_nlpds if not np.isinf(nlpd)]
    
    if len(valid_nlpds) >= k_folds // 2:  # At least half the folds must succeed
        mean_nlpd = np.mean(valid_nlpds)
        std_nlpd = np.std(valid_nlpds)
        mean_r2 = np.mean([r2 for r2, nlpd in zip(fold_r2s, fold_nlpds) if not np.isinf(nlpd)])
        mean_rmse = np.mean([rmse for rmse, nlpd in zip(fold_rmses, fold_nlpds) if not np.isinf(nlpd)])
    else:
        # Too many folds failed
        mean_nlpd = float('inf')
        std_nlpd = float('inf')
        mean_r2 = -float('inf')
        mean_rmse = float('inf')
    
    return {
        'agent_id': agent_id,
        'mean_nlpd': mean_nlpd,
        'std_nlpd': std_nlpd,
        'mean_r2': mean_r2,
        'mean_rmse': mean_rmse,
        'fold_nlpds': fold_nlpds,
        'fold_r2s': fold_r2s,
        'fold_rmses': fold_rmses,
        'valid_folds': len(valid_nlpds),
        'total_folds': k_folds
    }

def aggregate_local_cv_results(cv_results_list):
    """
    Aggregate cross-validation results from all agents.
    Each agent evaluated consensus params on their own local data.
    
    Args:
        cv_results_list: List of CV result dictionaries from each agent
    
    Returns:
        dict: Aggregated CV results (mean across agents)
    """
    # Extract NLPD scores from all agents
    all_nlpds = []
    all_r2s = []
    all_agent_ids = []
    
    for cv_result in cv_results_list:
        if not np.isinf(cv_result['mean_nlpd']):
            all_nlpds.append(cv_result['mean_nlpd'])
            all_r2s.append(cv_result['mean_r2'])
            all_agent_ids.append(cv_result['agent_id'])
    
    if len(all_nlpds) > 0:
        # Aggregate by averaging across agents
        aggregated_nlpd = np.mean(all_nlpds)
        aggregated_nlpd_std = np.std(all_nlpds)  # Std dev across agents
        aggregated_r2 = np.mean(all_r2s)
    else:
        aggregated_nlpd = float('inf')
        aggregated_nlpd_std = float('inf')
        aggregated_r2 = -float('inf')
    
    return {
        'aggregated_nlpd': aggregated_nlpd,
        'aggregated_nlpd_std': aggregated_nlpd_std,
        'aggregated_r2': aggregated_r2,
        'agent_nlpds': all_nlpds,
        'agent_ids': all_agent_ids,
        'num_valid_agents': len(all_nlpds),
        'total_agents': len(cv_results_list),
        'individual_results': cv_results_list
    }

def evaluate_predictions(Y_true, Y_pred, Y_pred_var=None, dataset_type="Test"):
    """
    Evaluate prediction quality using various metrics.
    
    Args:
        Y_true: True target values
        Y_pred: Predicted mean values  
        Y_pred_var: Predicted variances (optional)
        dataset_type: Name for the dataset being evaluated
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print(f"\n=== {dataset_type} Set Evaluation ===")
    
    # Basic regression metrics
    mse = mean_squared_error(Y_true, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_true, Y_pred)
    r2 = r2_score(Y_true, Y_pred)
    
    # Additional metrics
    residuals = Y_true - Y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    max_error = np.max(np.abs(residuals))
    
    print(f"Regression Metrics:")
    print(f"  Mean Squared Error (MSE):     {mse:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  Mean Absolute Error (MAE):    {mae:.6f}")
    print(f"  R² Score:                     {r2:.6f}")
    print(f"  Max Absolute Error:           {max_error:.6f}")
    
    print(f"\nResidual Analysis:")
    print(f"  Mean Residual:                {mean_residual:.6f}")
    print(f"  Std Residual:                 {std_residual:.6f}")
    
    # Uncertainty quantification metrics (if variance provided)
    if Y_pred_var is not None:
        Y_pred_std = np.sqrt(Y_pred_var)
        
        # Coverage probabilities (what fraction of true values fall within predicted intervals)
        # 1σ interval (68% confidence)
        within_1sigma = np.mean(np.abs(residuals) <= Y_pred_std)
        # 2σ interval (95% confidence)  
        within_2sigma = np.mean(np.abs(residuals) <= 2 * Y_pred_std)
        
        # Average uncertainty
        mean_uncertainty = np.mean(Y_pred_std)
        
        # Normalized root mean squared error in terms of uncertainty (calibration quality)
        normalized_rmse_uncertainty = np.sqrt(np.mean((residuals / Y_pred_std)**2))
        
        # Negative Log Predictive Density (NLPD)
        # NLPD = -log p(y_true | y_pred_mean, y_pred_var) for Gaussian predictions
        # For Gaussian: -log p(y | μ, σ²) = 0.5 * log(2π) + 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
        
        # Add small epsilon for numerical stability to avoid log(0) or division by 0
        eps = 1e-10
        Y_pred_var_safe = np.maximum(Y_pred_var, eps)
        
        log_2pi = np.log(2 * np.pi)
        nlpd_per_point = 0.5 * log_2pi + 0.5 * np.log(Y_pred_var_safe) + 0.5 * (residuals**2 / Y_pred_var_safe)
        nlpd = np.mean(nlpd_per_point)
        
        print(f"\nUncertainty Quantification:")
        print(f"  Mean Predictive Std:          {mean_uncertainty:.6f}")
        print(f"  Coverage within 1σ:           {within_1sigma:.3f} (expected: 0.68)")
        print(f"  Coverage within 2σ:           {within_2sigma:.3f} (expected: 0.95)")
        print(f"  Normalized RMSE (Uncertainty): {normalized_rmse_uncertainty:.6f} (good if ≈ 1.0)")
        print(f"  Negative Log Predictive Density: {nlpd:.6f} (lower is better)")
        
        # NLPD interpretation
        nlpd_quality = ""
        if nlpd < 0.5:
            nlpd_quality = "Excellent uncertainty quantification"
        elif nlpd < 1.0:
            nlpd_quality = "Good uncertainty quantification"
        elif nlpd < 2.0:
            nlpd_quality = "Fair uncertainty quantification"
        else:
            nlpd_quality = "Poor uncertainty quantification"
        print(f"  NLPD Quality Assessment:      {nlpd_quality}")
        
        # Calibration quality
        if within_1sigma > 0.5 and within_2sigma > 0.8:
            uncertainty_quality = "Good"
        elif within_1sigma > 0.4 and within_2sigma > 0.7:
            uncertainty_quality = "Fair" 
        else:
            uncertainty_quality = "Poor"
        print(f"  Uncertainty Calibration:      {uncertainty_quality}")
        
        print(f"\nNote: NLPD (Negative Log Predictive Density) measures how well the predicted")
        print(f"      uncertainty matches the actual prediction errors. Lower NLPD indicates")
        print(f"      better calibrated uncertainty estimates and higher prediction confidence.")
    
    # Overall performance assessment
    if r2 > 0.9:
        performance = "Excellent"
    elif r2 > 0.7:
        performance = "Good"
    elif r2 > 0.5:
        performance = "Fair"
    else:
        performance = "Poor"
    
    print(f"\nOverall Performance: {performance}")
    
    # Range-normalized RMSE (always available, regardless of uncertainty)
    y_range = Y_true.max() - Y_true.min()
    normalized_rmse_range = rmse / y_range if y_range > 0 else float('inf')
    print(f"Range-normalized RMSE:        {normalized_rmse_range:.6f} (lower is better)")
    
    # Create metrics dictionary
    metrics = {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'r2': r2,
        'max_error': max_error,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'normalized_rmse_range': normalized_rmse_range,  # Always available
        'performance': performance
    }
    
    if Y_pred_var is not None:
        metrics.update({
            'mean_uncertainty': mean_uncertainty,
            'within_1sigma': within_1sigma,
            'within_2sigma': within_2sigma,
            'normalized_rmse_uncertainty': normalized_rmse_uncertainty,  # Uncertainty-based calibration
            'nlpd': nlpd,
            'uncertainty_quality': uncertainty_quality
        })
    
    return metrics

def plot_predictions(X_test, Y_true, Y_pred, Y_pred_var=None, X_train=None, Y_train=None, 
                    title="Quantum GP Predictions", save_plot=False, output_dir="plots", config=None, nlpd_info=None):
    """
    Plot prediction results for different input dimensions.
    
    Args:
        X_test: Test input data
        Y_true: True test outputs
        Y_pred: Predicted test outputs
        Y_pred_var: Predicted variances (optional)
        X_train: Training inputs (optional, for context)
        Y_train: Training outputs (optional, for context)
        title: Plot title
        save_plot: Whether to save plot
        output_dir: Directory to save plots
        config: Configuration dictionary to display on plot (optional)
        nlpd_info: Dictionary with NLPD information for enhanced display (optional)
    """
    input_dim = X_test.shape[1] if X_test.ndim > 1 else 1
    Y_pred_std = np.sqrt(Y_pred_var) if Y_pred_var is not None else None
    
    # Calculate NLPD for display if variance is available
    nlpd_display = None
    if Y_pred_var is not None:
        residuals = Y_true - Y_pred
        
        # Add small epsilon for numerical stability to avoid log(0) or division by 0
        eps = 1e-10
        Y_pred_var_safe = np.maximum(Y_pred_var, eps)
        
        log_2pi = np.log(2 * np.pi)
        nlpd_per_point = 0.5 * log_2pi + 0.5 * np.log(Y_pred_var_safe) + 0.5 * (residuals**2 / Y_pred_var_safe)
        nlpd_display = np.mean(nlpd_per_point)
    
    if input_dim == 1:
        # 1D plotting with uncertainty bands
        fig, (ax_main, ax_config) = plt.subplots(1, 2, figsize=(16, 6), 
                                                gridspec_kw={'width_ratios': [3, 1]})
        
        # Sort test data for nice line plots
        sort_idx = np.argsort(X_test[:, 0])
        X_sorted = X_test[sort_idx, 0]
        Y_true_sorted = Y_true[sort_idx]
        Y_pred_sorted = Y_pred[sort_idx]
        
        # Plot training data if provided
        if X_train is not None and Y_train is not None:
            ax_main.scatter(X_train[:, 0], Y_train, c='lightblue', alpha=0.6, s=20, label='Training Data')
        
        # Plot true test function
        ax_main.scatter(X_test[:, 0], Y_true, c='red', alpha=0.7, s=30, label='True Test Data')
        
        # Plot predictions
        ax_main.plot(X_sorted, Y_pred_sorted, 'b-', linewidth=2, label='GP Prediction')
        
        # Plot uncertainty bands if available
        if Y_pred_std is not None:
            Y_pred_std_sorted = Y_pred_std[sort_idx]
            ax_main.fill_between(X_sorted, 
                           Y_pred_sorted - 2*Y_pred_std_sorted,
                           Y_pred_sorted + 2*Y_pred_std_sorted,
                           alpha=0.2, color='blue', label='95% Confidence')
            ax_main.fill_between(X_sorted,
                           Y_pred_sorted - Y_pred_std_sorted, 
                           Y_pred_sorted + Y_pred_std_sorted,
                           alpha=0.3, color='blue', label='68% Confidence')
        
        ax_main.set_xlabel('X')
        ax_main.set_ylabel('Y')
        ax_main.set_title(title)
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # Add configuration text box
        if config is not None:
            config_text = ""
            
            # Add regular configuration information
            for key, value in config.items():
                config_text += f"{key}: {value}\n"
            
            ax_config.text(0.05, 0.95, config_text, transform=ax_config.transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax_config.set_title('Configuration', fontsize=10, fontweight='bold')
            ax_config.axis('off')
        else:
            ax_config.axis('off')
        
    elif input_dim == 2:
        # 2D plotting with subplots
        fig = plt.figure(figsize=(20, 5))
        
        # True values
        ax1 = fig.add_subplot(141, projection='3d')
        scatter1 = ax1.scatter(X_test[:, 0], X_test[:, 1], Y_true, c=Y_true, cmap='viridis', s=20)
        ax1.set_title('True Values')
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2') 
        ax1.set_zlabel('Y')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5)
        
        # Predictions
        ax2 = fig.add_subplot(142, projection='3d')
        scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1], Y_pred, c=Y_pred, cmap='viridis', s=20)
        ax2.set_title('Predictions')
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_zlabel('Y')
        plt.colorbar(scatter2, ax=ax2, shrink=0.5)
        
        # Residuals
        ax3 = fig.add_subplot(143, projection='3d')
        residuals = Y_true - Y_pred
        scatter3 = ax3.scatter(X_test[:, 0], X_test[:, 1], residuals, c=residuals, cmap='RdBu', s=20)
        ax3.set_title('Residuals')
        ax3.set_xlabel('X1')
        ax3.set_ylabel('X2')
        ax3.set_zlabel('Residual')
        plt.colorbar(scatter3, ax=ax3, shrink=0.5)
        
        # Configuration text
        ax4 = fig.add_subplot(144)
        if config is not None:
            config_text = ""
            
            for key, value in config.items():
                config_text += f"{key}: {value}\n"
            
            ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Configuration', fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
    else:
        # Higher dimensions: correlation plot and residuals
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Prediction vs true correlation
        axes[0].scatter(Y_true, Y_pred, alpha=0.6, s=20)
        axes[0].plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Prediction Correlation')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = Y_true - Y_pred
        axes[1].scatter(Y_pred, residuals, alpha=0.6, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Configuration text
        if config is not None:
            config_text = ""
            
            for key, value in config.items():
                config_text += f"{key}: {value}\n"
            
            axes[2].text(0.05, 0.95, config_text, transform=axes[2].transAxes,
                        fontsize=8, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[2].set_title('Configuration', fontsize=10, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'{title} ({input_dim}D Input)')
        plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique identifier based on timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = os.path.join(output_dir, f"predictions_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to: {filename}")
    
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parallel Distributed QGPR with ADMM')
    parser.add_argument('--differentiation', choices=['parameter_shift', 'autodiff'], default='autodiff',
                       help='Quantum differentiation method: parameter_shift (Qiskit) or autodiff (PennyLane) (default: autodiff)')
    parser.add_argument('--n-agents', type=int, default=4,
                       help='Number of agents (default: 4)')
    parser.add_argument('--num-qubits', type=int, default=3,
                       help='Number of qubits (default: 3)')
    parser.add_argument('--num-layers', type=int, default=1,
                       help='Number of layers in the encoding circuit (default: 1)')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='Maximum ADMM iterations (default: 100)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='ADMM convergence tolerance (default: 1e-6)')
    parser.add_argument('--rho', type=float, default=100.0,
                       help='ADMM penalty parameter rho (default: 100.0)')
    parser.add_argument('--L', type=float, default=100.0,
                       help='Lipschitz constant L for all agents (default: 100.0)')
    parser.add_argument('--input-dim', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                       help='Input dimensionality: 1-6D (default: 1)')
    parser.add_argument('--n-dataset', type=int, default=100,
                       help='Number of dataset samples (default: 100)')
    parser.add_argument('--partition', choices=['regional', 'random', 'sequential'], default='regional',
                       help='Data partitioning method: regional (spatial/local), random, or sequential (default: regional)')
    parser.add_argument('--data-percentage', type=float, default=1.0,
                       help='Percentage of local data to give to each agent (0.0 to 1.0, default: 1.0 = 100%%)')
    parser.add_argument('--noise-std', type=float, default=0.1,
                       help='Noise standard deviation (default: 0.1)')
    parser.add_argument('--noise-lower-bound', type=float, default=0.1,
                       help='Lower bound for noise hyperparameter optimization (default: 0.1)')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split ratio (default: 0.1)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers for parameter shift evaluation (default: None, uses all available CPUs)')
    parser.add_argument('--shift-value', type=float, default=np.pi/8,
                       help='Shift value for parameter shift rule (default: π/8 ≈ 0.393, optimal value)')
    
    # Dataset generation arguments
    parser.add_argument('--classical-dataset', action='store_true',
                       help='Use classical dataset generation instead of quantum (default: quantum)')
    parser.add_argument('--real-world-dataset', type=str, default='srtm',
                       choices=['sst', 'sea_surface_temperature', 'robot_push', 'robot', 'push', 'srtm_elevation', 'srtm', 'elevation'],
                       help='Use real-world dataset instead of synthetic. Options: sst/sea_surface_temperature (2D), robot_push/robot/push (3D), srtm_elevation/srtm/elevation (2D) (default: srtm)')
    parser.add_argument('--srtm-region', type=str, default='maharashtra',
                       choices=['maharashtra', 'great_lakes', 'oregon_coast', 'washington_coast'],
                       help='SRTM elevation region from Attentive Kernels paper (default: maharashtra)')
    parser.add_argument('--use-srtm-preprocessed', action='store_true', default=False,
                       help='Use preprocessed .npy SRTM files instead of raw HGT files (default: False)')
    parser.add_argument('--dataset-max-samples', type=int, default=1000,
                       help='Maximum number of samples to load from real-world datasets (default: 1000)')
    parser.add_argument('--dataset-subsample', type=int, default=10,
                       help='Subsampling factor for real-world datasets (default: 10)')
    parser.add_argument('--dataset-normalize', action='store_true', default=True,
                       help='Normalize real-world dataset features and targets (default: True)')
    parser.add_argument('--dataset-only', action='store_true',
                       help='Stop after dataset generation without training')
    parser.add_argument('--save-dataset', action='store_true',
                       help='Save the generated dataset to CSV file (default: False)')
    parser.add_argument('--dataset-name', type=str, default='quantum_dataset',
                       help='Name for saved dataset (default: quantum_dataset)')
    parser.add_argument('--data-range', nargs=2, type=float, default=[-2.0, 2.0],
                       help='Range for input data generation (default: -2.0 2.0)')
    parser.add_argument('--encoding', 
                       choices=['chebyshev', 'yz_cx', 'hubregtsen', 'kyriienko', 'multi_control', 'layered', 'random', 'highdim'], 
                       default='hubregtsen',
                       help='Encoding circuit type: chebyshev (has arccos), yz_cx (safer, no arccos), hubregtsen (research), kyriienko (research), multi_control (complex), layered (customizable), random (randomized), highdim (high-dimensional) (default: hubregtsen)')
    parser.add_argument('--kernel-type', 
                       choices=['fidelity', 'projected'], 
                       default='projected',
                       help='Quantum kernel type: fidelity (state fidelity) or projected (observable measurements) (default: projected)')
    parser.add_argument('--measurement', type=str, default='XYZ',
                       help='Measurement operator for ProjectedQuantumKernel: Can be a string like "XYZ", "X", "Y", "Z", "XX", "YY", "ZZ", etc., or a list of Pauli strings for multi-qubit measurements (default: XYZ)')
    parser.add_argument('--outer-kernel', type=str, default='gaussian',
                       choices=['gaussian', 'matern', 'expsinesquared', 'rationalquadratic', 'dotproduct', 'pairwisekernel'],
                       help='Outer kernel type for ProjectedQuantumKernel: gaussian, matern (default), expsinesquared, rationalquadratic, dotproduct, pairwisekernel')
    parser.add_argument('--outer-kernel-gamma', type=float, default=1.0,
                       help='Gamma parameter for Gaussian outer kernel (default: 1.0)')
    parser.add_argument('--outer-kernel-length-scale', type=float, default=1.0,
                       help='Length scale parameter for Matern and RationalQuadratic kernels (default: 1.0)')
    parser.add_argument('--outer-kernel-nu', type=float, default=1.5,
                       help='Nu parameter for Matern kernel (default: 1.5)')
    parser.add_argument('--outer-kernel-alpha', type=float, default=1.0,
                       help='Alpha parameter for RationalQuadratic kernel (default: 1.0)')
    parser.add_argument('--outer-kernel-sigma', type=float, default=1.0,
                       help='Sigma parameter for DotProduct kernel (default: 1.0)')
    parser.add_argument('--outer-kernel-periodicity', type=float, default=1.0,
                       help='Periodicity parameter for ExpSineSquared kernel (default: 1.0)')
    parser.add_argument('--regularization', type=str, default=None,
                       choices=['thresholding', 'tikhonov', None],
                       help='Regularization technique for ProjectedQuantumKernel: thresholding, tikhonov, or None (default: None)')
    parser.add_argument('--prediction-method', type=str, default='gPoE',
                       choices=['gPoE', 'full-data', 'rbcm'],
                       help='Prediction aggregation method for distributed quantum GP: '
                            'independent (FACT-GP simple aggregation), '
                            'full-data (retrain on full dataset), or '
                            'rbcm (Robust Bayesian Committee Machine with entropy weighting) '
                            '(default: gPoE)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting the generated data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for ground truth parameters and data splitting (default: 42). Ensures same quantum parameters across runs.')
    parser.add_argument('--data-seed', type=int, default=None,
                       help='Random seed for X data generation (default: None). If None, X values differ each run. If set, X values are reproducible.')
    parser.add_argument('--kernel-params', type=float, nargs='+', default=None,
                       help='Specific kernel parameters to use as ground truth (e.g., --kernel-params 0.576 2.450 1.875 1.401 0.314 1.443). If not provided, random parameters are generated.')
    
    # Riemannian optimization arguments
    parser.add_argument('--riemannian-lr', type=float, default=0.015,
                       help='Initial learning rate for Riemannian optimizer (default: 0.015, balanced optimization)')
    parser.add_argument('--riemannian-method', 
                       choices=['gradient_descent', 'momentum', 'conjugate_gradient'], 
                       default='gradient_descent',
                       help='Riemannian optimization method (default: gradient_descent)')
    parser.add_argument('--riemannian-beta', type=float, default=0.9,
                       help='Beta parameter for momentum/conjugate gradient methods (default: 0.9)')
    parser.add_argument('--riemannian-rho', type=float, default=1.0,
                       help='ADMM penalty parameter for Riemannian optimization (default: 1.0)')
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--max-step-size', type=float, default=0.1,
                       help='Maximum step size (default: 0.1)')
    
    # Cross-validation arguments
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--cv-patience', type=int, default=5,
                       help='Early stopping patience for CV-based optimization (default: 5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.data_percentage <= 1.0):
        raise ValueError(f"data_percentage must be between 0.0 and 1.0, got {args.data_percentage}")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Process outer kernel parameters
    outer_kernel = args.outer_kernel
    outer_kernel_params = {}
    
    if outer_kernel == 'gaussian':
        outer_kernel_params = {'gamma': args.outer_kernel_gamma}
    elif outer_kernel == 'matern':
        outer_kernel_params = {
            'length_scale': args.outer_kernel_length_scale,
            'nu': args.outer_kernel_nu
        }
    elif outer_kernel == 'expsinesquared':
        outer_kernel_params = {
            'length_scale': args.outer_kernel_length_scale,
            'periodicity': args.outer_kernel_periodicity
        }
    elif outer_kernel == 'rationalquadratic':
        outer_kernel_params = {
            'length_scale': args.outer_kernel_length_scale,
            'alpha': args.outer_kernel_alpha
        }
    elif outer_kernel == 'dotproduct':
        outer_kernel_params = {'sigma_0': args.outer_kernel_sigma}
    elif outer_kernel == 'pairwisekernel':
        # PairwiseKernel uses default parameters
        outer_kernel_params = {}
    
    print(f"Using outer kernel: {outer_kernel}")
    if outer_kernel_params:
        print(f"Outer kernel parameters: {outer_kernel_params}")
    if args.regularization is not None:
        print(f"Using regularization technique: {args.regularization}")
    else:
        print("No regularization applied")
    print()
    
    # Initialize dataset-related variables (fix UnboundLocalError)
    dataset_name = None  # Will be set based on dataset type
    srtm_data_seed = args.seed  # Default seed
    
    # Check dataset type and load accordingly
    if args.real_world_dataset:
        # === Real-World Dataset Mode ===
        print("=== Real-World Dataset Mode ===")
        
        # Get dataset info
        dataset_info = get_dataset_info()
        dataset_key = args.real_world_dataset.lower()
        
        # Find matching dataset key
        dataset_name = None
        for key, info in dataset_info.items():
            if dataset_key in [key] + [alt.lower() for alt in key.split('_')]:
                dataset_name = key
                break
        
        # Special handling for SRTM dataset aliases
        if dataset_key in ['srtm', 'elevation', 'srtm_elevation']:
            dataset_name = 'srtm_elevation'
        
        if dataset_name is None:
            available = list(dataset_info.keys())
            raise ValueError(f"Unknown real-world dataset '{args.real_world_dataset}'. Available: {available}")
        
        info = dataset_info[dataset_name]
        print(f"Loading {info['name']} dataset...")
        print(f"  Dimensions: {info['dimensions']}D")
        print(f"  Input: {info['input_desc']}")
        print(f"  Output: {info['output_desc']}")
        print(f"  Source: {info['source']}")
        print(f"  Max samples: {args.dataset_max_samples}")
        
        # Special handling for dataset-specific parameters
        if dataset_name == 'sst':
            print(f"  Subsample factor: {args.dataset_subsample}")
        elif dataset_name == 'srtm_elevation':
            print(f"  Region: {args.srtm_region}")
            print(f"  Use preprocessed: {args.use_srtm_preprocessed}")
            print(f"  Subsample factor: {args.dataset_subsample}")
        
        print(f"  Normalize: {args.dataset_normalize}")
        print(f"  Plot dataset: {not args.no_plot}")
        
        # Generate time-based seed for SRTM dataset (like quantum GP)
        if dataset_name == 'srtm_elevation':
            srtm_data_seed = int(time.time() * 1000) % 2**32
            print(f"  SRTM data seed: {srtm_data_seed} (time-based, changes each run)")
        else:
            srtm_data_seed = args.seed  # Use fixed seed for other datasets
            print(f"  Data seed: {srtm_data_seed} (fixed)")
        print()
        
        # Load the dataset
        try:
            kwargs = {
                'normalize': args.dataset_normalize,
                'max_samples': args.dataset_max_samples,
                'random_state': srtm_data_seed,  # Use time-based seed for SRTM
                'save_plot': not args.no_plot  # Pass plotting flag
            }
            
            if dataset_name == 'sst':
                kwargs['subsample_factor'] = args.dataset_subsample
            elif dataset_name == 'srtm_elevation':
                kwargs['region'] = args.srtm_region
                kwargs['subsample_factor'] = args.dataset_subsample
                kwargs['use_preprocessed'] = args.use_srtm_preprocessed
                # Pass encoding circuit type so data gets scaled to correct bounds
                kwargs['encoding_type'] = args.encoding
                # For Chebyshev circuits, pass the nonlinearity parameter
                if args.encoding == 'chebyshev':
                    kwargs['encoding_nonlinearity'] = 'arccos'  # Default for Chebyshev
            
            X_full, Y_full = load_real_world_dataset(dataset_name, **kwargs)
            
            # Override input_dim with actual dataset dimension
            actual_input_dim = X_full.shape[1]
            if args.input_dim != actual_input_dim:
                print(f"Note: Overriding --input-dim from {args.input_dim} to {actual_input_dim} (dataset dimension)")
                args.input_dim = actual_input_dim
            
            # No ground truth quantum parameters for real-world dataset
            ground_truth_params = None
            
        except Exception as e:
            print(f"Error loading real-world dataset '{args.real_world_dataset}': {e}")
            raise
        
    elif args.classical_dataset:
        # === Classical Training Mode ===
        print("=== Classical Dataset Training Mode ===")
        dataset_name = 'classical'  # Set dataset name for consistency
        if args.data_seed is not None:
            print(f"  Data seed (X values): {args.data_seed} (fixed)")
        else:
            print(f"  Data seed (X values): None (random each run)")
        
        # Calculate total samples needed for classical generation
        total_samples = int(args.n_dataset / (1 - args.test_split))  # Ensure we get approximately n_dataset after split
        
        # Generate full dataset first, then split (recommended approach)
        print(f"Generating {args.input_dim}D classical dataset with {total_samples} total samples...")
        X_full, Y_full = generate_data_numpy(total_samples, args.input_dim, args.noise_std, args.data_seed)
        
        # No ground truth quantum parameters for classical dataset
        ground_truth_params = None
        
    else:
        # === Default: Quantum Dataset Generation Mode ===
        print("=== Quantum Dataset Generation Mode ===")
        dataset_name = 'quantum'  # Set dataset name for consistency
        print(f"Configuration:")
        print(f"  Number of samples: {args.n_dataset}")
        print(f"  Input dimension: {args.input_dim}D")
        print(f"  Number of qubits: {args.num_qubits}")
        print(f"  Number of layers: {args.num_layers}")
        print(f"  Data range: {args.data_range}")
        print(f"  Encoding type: {args.encoding}")
        print(f"  Noise std: {args.noise_std}")
        print(f"  Differentiation: {args.differentiation}")
        print(f"  Random seed (ground truth params): {args.seed}")
        if args.kernel_params:
            print(f"  Custom kernel parameters: {args.kernel_params}")
        if args.data_seed is not None:
            print(f"  Data seed (X values): {args.data_seed} (fixed)")
        else:
            print(f"  Data seed (X values): None (random each run)")
        print()
        
        # Recommend sample sizes similar to dataset.ipynb
        recommended_samples = {
            1: 1000,
            2: 32400,  # 180x180 grid
            3: 16900,  # From dataset.ipynb
            4: 32400,  # From dataset.ipynb
            5: 16900,
            6: 32400   # From dataset.ipynb
        }
        
        if args.n_dataset != recommended_samples.get(args.input_dim, args.n_dataset):
            print(f"Note: Recommended sample size for {args.input_dim}D: {recommended_samples.get(args.input_dim)}")
        
        # Generate quantum GP data
        start_time = time.time()
        
        X_synthetic, Y_synthetic, ground_truth_params = generate_quantum_gp_data(
            num_samples=args.n_dataset,
            input_dim=args.input_dim,
            num_qubits=args.num_qubits,
            num_layers=args.num_layers,
            data_range=tuple(args.data_range),
            noise_std=args.noise_std,
            use_parameter_shift=(args.differentiation == 'parameter_shift'),
            kernel_params=np.array(args.kernel_params) if args.kernel_params else None,  # Pass custom kernel parameters
            encoding_type=args.encoding,
            kernel_type=args.kernel_type,
            measurement=args.measurement,
            outer_kernel=outer_kernel,
            outer_kernel_params=outer_kernel_params,
            regularization=args.regularization,
            data_seed=args.data_seed,  # Use command line data seed (None = random each time)
            param_seed=args.seed  # Use command line seed for consistent ground truth parameters
        )
        
        total_time = time.time() - start_time
        print(f"\nQuantum dataset generation time: {total_time:.4f}s")
        
        # Use the generated dataset for training
        X_full, Y_full = X_synthetic, Y_synthetic
    
    # === Common Dataset Post-Processing ===
    # Plot the data (for all dataset types) - but skip if real-world dataset already plotted
    if not args.no_plot and not args.real_world_dataset:  # Real-world datasets plot themselves
        if args.classical_dataset:
            plot_title = f"Classical Dataset ({args.input_dim}D)"
        else:
            plot_title = f"Quantum GP Data ({args.input_dim}D, {args.num_qubits} qubits)"
        
        try:
            plot_quantum_gp_data(X_full, Y_full, plot_title)
        except Exception as e:
            print(f"Warning: Could not plot data: {e}")
    elif args.real_world_dataset:
        print(f"Real-world dataset plotting completed during loading.")
    
    # Save the dataset if requested (for all dataset types)
    if args.save_dataset:
        try:
            dataset_filename = save_quantum_dataset(X_full, Y_full, args.dataset_name)
            print(f"Dataset saved to: {dataset_filename}")
        except Exception as e:
            print(f"Warning: Could not save dataset: {e}")
    
    # Print dataset summary
    print(f"\nDataset loaded/generated successfully!")
    print(f"Dataset: {X_full.shape[0]} samples, {X_full.shape[1]}D input")
    print(f"Input range: [{X_full.min():.3f}, {X_full.max():.3f}]")
    print(f"Output range: [{Y_full.min():.3f}, {Y_full.max():.3f}]")
    
    # Check if we should stop after dataset generation/loading
    if args.dataset_only:
        print("Stopping after dataset loading (--dataset-only flag)")
        return  # Exit without training
    
    print(f"\n{'='*50}")
    if args.real_world_dataset:
        print("Continuing to training mode with real-world dataset...")
    elif args.classical_dataset:
        print("Continuing to training mode with classical dataset...")
    else:
        print("Continuing to training mode with generated quantum dataset...")
    print(f"{'='*50}")
    
    # === Common Training Logic ===
    # Configuration
    use_parameter_shift = (args.differentiation == 'parameter_shift')
    n_agents = args.n_agents
    num_qubits = args.num_qubits
    num_layers = args.num_layers
    maxiter = args.max_iter
    tolADMM = args.tolerance
    input_dim = args.input_dim
    partition_method = args.partition
    noise_std = args.noise_std
    test_split = args.test_split
    num_workers = args.num_workers
    shift_value = args.shift_value
    
    # ADMM parameters from command line
    rho = args.rho
    L = args.L * np.ones(n_agents)  # Use same L for all agents
    
    # Riemannian optimization configuration
    riemannian_lr = args.riemannian_lr
    riemannian_method = args.riemannian_method
    riemannian_beta = args.riemannian_beta
    
    print(f"Configuration:")
    print("  Gradient method: riemannian")
    print(f"  Quantum differentiation: {args.differentiation}")
    print(f"  Parallel parameter evaluation: {'Enabled' if use_parameter_shift else 'Disabled'}")
    print("  Riemannian optimization: Enabled")
    print(f"    - Method: {riemannian_method}")
    print(f"    - Learning rate: {riemannian_lr}")
    print(f"    - Beta: {riemannian_beta}")
    print(f"  Cross-validation: {args.cv_folds}-fold CV with NLPD metric")
    print(f"    - Early stopping patience: {args.cv_patience} iterations")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of qubits: {num_qubits}")
    print(f"  Number of layers: {num_layers}")
    if not args.classical_dataset:
        print(f"  Training samples: {X_full.shape[0]} (from generated quantum dataset)")
    else:
        print(f"  Total samples generated: {X_full.shape[0]}")
    print(f"  Test split ratio: {test_split}")
    print(f"  Partition method: {partition_method}")
    print(f"  Noise std: {noise_std}")
    print(f"  Number of agents: {n_agents}")
    print(f"  Max iterations: {maxiter}")
    print(f"  Tolerance: {tolADMM}")
    print(f"  ADMM penalty parameter (rho): {rho}")
    print(f"  Lipschitz constant (L): {L[0]} (same for all agents)")
    if use_parameter_shift:
        print(f"  Parallel workers: {num_workers if num_workers else 'Auto (all CPUs)'}")
    print()
    
    # Split into train/test using sklearn (same as pxpGP_train.py)
    # Get indices to show train/test split in plot
    indices = np.arange(len(X_full))
    # Use the same seed as data sampling for consistent variation across experiments
    split_seed = srtm_data_seed if dataset_name == 'srtm_elevation' else args.seed
    X_train, X_test, Y_train, Y_test, train_indices, test_indices = train_test_split(
        X_full, Y_full, indices,
        test_size=test_split, 
        random_state=split_seed,  # Use same seed as data sampling for variation
        shuffle=True
    )
    
    print(f"After split - Training: X={X_train.shape}, Y={Y_train.shape}")
    print(f"After split - Test: X={X_test.shape}, Y={Y_test.shape}")
    
    # Compute training target std for fair scaling with classical baseline (Concern 2)
    y_train_std = np.std(Y_train) if len(Y_train) > 0 else 1.0
    print(f"Training target std: {y_train_std:.6f}")
    
    # Plot the data with train/test distinction
    if not args.no_plot:
        plot_quantum_gp_data(X_full, Y_full, 
                           f"Quantum GP Data with Train/Test Split ({args.input_dim}D, {args.num_qubits} qubits)",
                           train_indices=train_indices, test_indices=test_indices)
    
    # Split training data among agents
    print(f"Splitting training data among {n_agents} agents using {partition_method} method...")
    if args.data_percentage < 1.0:
        print(f"Each agent will receive {args.data_percentage*100:.1f}% of their local data (randomly sampled)")
    agent_data_splits = split_data_numpy(X_train, Y_train, n_agents, partition_method, args.data_percentage, args.seed)
    
    # Print agent data distribution
    for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
        print(f"  Agent {i+1}: {X_agent.shape[0]} samples")
    
    # Plot agent data distribution
    plot_title = f"Agent Data Distribution ({partition_method.title()} Partitioning)"
    if args.data_percentage < 1.0:
        plot_title += f" - {args.data_percentage*100:.1f}% Sampling"
    plot_agent_data_distribution(agent_data_splits, title=plot_title, save_plot=args.save_dataset)

    # Initialize quantum kernel once for all agents
    print("Initializing quantum kernel once for all agents...")
    kernel_init_start = time.time()
    q_kernel_template = create_quantum_kernel(num_qubits, input_dim, num_layers, use_parameter_shift, args.encoding, args.kernel_type, args.measurement, outer_kernel, outer_kernel_params, args.regularization)  # Use same encoding, kernel type, measurement and outer kernel as dataset generation
    kernel_init_time = time.time() - kernel_init_start
    print(f"Quantum kernel initialization took: {kernel_init_time:.4f}s")

    # Number of hyperparameters for the quantum kernel
    # Note: ProjectedQuantumKernel.num_parameters is None until first evaluation,
    # so we get it from the encoding circuit instead
    n_hyperparameters = q_kernel_template.num_parameters
    if n_hyperparameters is None:
        n_hyperparameters = q_kernel_template.encoding_circuit.num_parameters
    print(f"Encoding circuit parameters: {n_hyperparameters}")

    # Initialize the ADMM variables
    # Extended to include noise variance optimization
    # theta, psi, z now have shape (n_agents, n_hyperparameters + 1)
    # where the last element is the noise variance sigma_n^2
    n_params_extended = n_hyperparameters + 1  # +1 for noise variance
    theta = np.round(np.random.rand(n_agents, n_hyperparameters), 4)
    psi = np.round(np.random.rand(n_agents, n_hyperparameters), 4)
    
    # Initialize noise variance to args.noise_std (used for data generation)
    # This ensures consistency: data was generated with noise_std, so optimizer should start there
    initial_noise_var = noise_std  # Use args.noise_std instead of hardcoded 0.1
    noise_variance_init = np.full((n_agents, 1), initial_noise_var)
    noise_variance_init = np.round(np.clip(noise_variance_init, args.noise_lower_bound, 1.0), 4)
    
    theta = np.concatenate([theta, noise_variance_init], axis=1)  # Shape: (n_agents, n_hyperparameters + 1)
    psi = np.concatenate([psi, np.zeros((n_agents, 1))], axis=1)  # Shape: (n_agents, n_hyperparameters + 1)
    
    print(f"Extended parameter dimension: {n_params_extended} (circuit: {n_hyperparameters}, noise: 1)")
    print(f"Initialized noise variance (target={initial_noise_var:.6f}): {np.round(noise_variance_init.flatten(), 4)}")
    print(f"Noise variance bounds: [{args.noise_lower_bound}, 1.0]")
    
    # rho > 0: penalty parameter constant term of the augmented Lagrangian
    # L_i > 0: Lipschitz constant of the gradient of the loss function, different for each agent
    # rho and L are hyperparameters that can be tuned (set via command line arguments)
    iter = 0

    agents = []
    agent_data_list = []  # Store agent initialization data for ProcessPoolExecutor
    
    for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
        # Ensure X_agent has the right shape for GP (N, D)
        if X_agent.ndim == 1:
            X_agent = X_agent.reshape(-1, 1)
        
        # Always use the Riemannian agent with pre-initialized kernel.
        agent = RiemannianAgent(
            agent_id=f"agent_{i+1}",
            X_sub=X_agent,
            Y_sub=Y_agent,
            num_qubits=num_qubits,
            noise_std=noise_std,
            rho=rho,
            L=L[i],
            q_kernel=q_kernel_template,
            use_parameter_shift=use_parameter_shift,
            num_workers=num_workers,
            shift_value=shift_value,
            num_layers=num_layers,
            encoding_type=args.encoding,
            kernel_type=args.kernel_type,
            measurement=args.measurement,
            outer_kernel=outer_kernel,
            outer_kernel_params=outer_kernel_params,
            regularization=args.regularization,
            riemannian_lr=riemannian_lr,
            riemannian_method=riemannian_method,
            riemannian_beta=riemannian_beta,
            noise_lower_bound=args.noise_lower_bound
        )
        agents.append(agent)
        
        # For process execution: store initialization data (avoids pickling quantum objects)
        agent_data_list.append((f"agent_{i+1}", X_agent, Y_agent, num_qubits, noise_std, rho, L[i]))

    # Initialize z as the average of theta and psi on the Riemannian manifold.
    theta_circuit = theta[:, :-1]
    psi_circuit = psi[:, :-1]
    theta_noise = theta[:, -1]  # Last column: noise variance
    psi_noise = psi[:, -1]      # Last column: noise variance

    manifold, _, riemannian_admm = create_riemannian_framework(
        num_parameters=n_hyperparameters,
        learning_rate=riemannian_lr,
        rho=rho,
        method=riemannian_method,
        gradient_clip_norm=args.gradient_clip_norm,
        max_step_size=args.max_step_size
    )
    z_circuit = np.round(riemannian_admm.update_z(theta_circuit, psi_circuit), 4)
    z_noise = np.round(np.mean(theta_noise + psi_noise / rho), 4)
    z_noise = np.clip(z_noise, args.noise_lower_bound, 1.0)

    z = np.concatenate([z_circuit, [z_noise]])
    print(f"Initialized z_circuit using Riemannian ADMM on {manifold.name}")
    print(f"Initialized z_noise using arithmetic mean: {z_noise:.6f}")
    # z = np.random.rand(n_hyperparameters)  # Random initialization for z

    print("Starting ADMM optimization with riemannian gradients...")
    
    # Initialize NLL loss tracking for analysis
    nll_loss_history = []  # Store NLL losses for each iteration
    
    # Print initial hyperparameters
    print(f"\nInitial hyperparameters:")
    if ground_truth_params is not None:
        print(f"Ground truth params (circuit only): {ground_truth_params}")
        print(f"Initial ADMM z (circuit): {z[:-1]}")
        print(f"Initial ADMM z (noise):   {z[-1]:.6f}")
        
        # Calculate initial distance to ground truth (circuit parameters only)
        initial_euclidean = np.linalg.norm(z[:-1] - ground_truth_params)
        print(f"Initial ||z_circuit - ground_truth||: {initial_euclidean:.6f}")
        
        initial_error = initial_euclidean
    else:
        print(f"Initial ADMM z (circuit): {z[:-1]}")
        print(f"Initial ADMM z (noise):   {z[-1]:.6f}")
        print("(No ground truth available for classical dataset)")
    print(f"Initial agent params (theta):")
    for i, theta_i in enumerate(theta):
        print(f"  Agent {i+1}: {theta_i}")
    print()

    # CV-based solution tracking (realistic approach - no ground truth needed)
    k_folds = args.cv_folds  # Use command line argument
    cv_patience = args.cv_patience  # Use command line argument
    z_best_cv = None
    cv_best = float('inf')
    patience_counter = 0
    cv_score_history = []
    
    # Keep ground truth tracking only for comparison/analysis (not for optimization)
    z_best = None
    error_best = float('inf')
    error_history = []
    
    print(f"Starting ADMM optimization with {k_folds}-fold cross-validation on consensus parameters...")
    print(f"Using NLPD as validation metric (lower is better)")
    print(f"Early stopping patience: {cv_patience} iterations")
    print(f"Note: Ground truth parameters used only for analysis, not optimization")
    
    admm_start_time = time.time()

    while True:
        iter += 1
        print(f"\n=== ADMM Iteration {iter} ===")
        iteration_start = time.time()
        
        # CORRECT ADMM ORDER: 1. Update z first (consensus update)
        # Separate circuit parameters and noise variance
        z_old = z.copy()
        
        # Extract and update circuit parameters using Riemannian ADMM
        theta_circuit = theta[:, :-1]
        psi_circuit = psi[:, :-1]
        theta_noise = theta[:, -1]
        psi_noise = psi[:, -1]

        manifold, _, riemannian_admm = create_riemannian_framework(
            num_parameters=n_hyperparameters,
            learning_rate=riemannian_lr,
            rho=rho,
            method=riemannian_method,
            gradient_clip_norm=args.gradient_clip_norm,
            max_step_size=args.max_step_size
        )
        z_circuit = np.round(riemannian_admm.update_z(theta_circuit, psi_circuit), 4)
        z_noise = np.round(np.mean(theta_noise + psi_noise / rho), 4)
        z_noise = np.clip(z_noise, args.noise_lower_bound, 1.0)

        z = np.concatenate([z_circuit, [z_noise]])
        
        print(f"Updated consensus z (circuit): {z[:-1]}")
        print(f"Updated consensus z (noise): {z[-1]:.6f}")
        
        # CORRECT ADMM ORDER: 2. Update theta and psi using new z
        # Use ProcessPoolExecutor to run agents in parallel
        # ProcessPoolExecutor creates separate processes, avoiding PennyLane's threading issues
        with ProcessPoolExecutor() as executor:
            # Prepare data for each process (includes NEW z and OLD psi_i for each agent)
            process_args = []
            for i, agent_data in enumerate(agent_data_list):
                # Extend agent_data with current z, psi_i, and Riemannian configuration
                full_agent_data = agent_data + (z, psi[i], use_parameter_shift, input_dim, num_layers, num_workers, shift_value, args.encoding, args.kernel_type, args.measurement, riemannian_lr, riemannian_method, riemannian_beta, outer_kernel, outer_kernel_params, args.regularization, args.noise_lower_bound)
                process_args.append(full_agent_data)
            
            # Execute in parallel processes
            results = list(executor.map(process_agent_training, process_args))
        # results[i] is a tuple with (theta_i, psi_i, nll_loss, condition_number, nll_components, gradient_augmented)

        # Extract theta, psi, nll_loss, condition numbers, and NLL components from the results
        nll_losses = []  # Store NLL losses for analysis
        condition_numbers = []  # Store condition numbers for analysis
        nll_components_list = []  # Store NLL components for correlation analysis
        for i, result in enumerate(results):
            theta_i, psi_i, nll_loss, cond_num, nll_comp, grad_augmented = result
            
            theta[i] = np.round(theta_i, 4)
            psi[i] = np.round(psi_i, 4)
            nll_losses.append(nll_loss)
            condition_numbers.append(cond_num)
            nll_components_list.append(nll_comp)

        # Print NLL losses for all agents
        print(f"\nNLL Loss Analysis - Iteration {iter}:")
        total_nll = 0.0
        valid_losses = []
        for i, nll_loss in enumerate(nll_losses):
            print(f"  Agent {i+1}: {nll_loss:.8f}")
            if not np.isinf(nll_loss) and not np.isnan(nll_loss):
                valid_losses.append(nll_loss)
                total_nll += nll_loss
        
        if valid_losses:
            avg_nll = total_nll / len(valid_losses)
            min_nll = min(valid_losses)
            max_nll = max(valid_losses)
            print(f"  Summary: Total={total_nll:.6f}, Average={avg_nll:.6f}, Min={min_nll:.6f}, Max={max_nll:.6f}")
            
            # Store NLL losses and condition numbers for analysis
            nll_loss_history.append({
                'iteration': iter,
                'agent_losses': nll_losses.copy(),
                'condition_numbers': condition_numbers.copy(),
                'nll_components': nll_components_list.copy(),
                'total_nll': total_nll,
                'avg_nll': avg_nll,
                'min_nll': min_nll,
                'max_nll': max_nll
            })
        else:
            print(f"  Warning: All NLL losses are invalid (inf/nan)")
            nll_loss_history.append({
                'iteration': iter,
                'agent_losses': nll_losses.copy(),
                'condition_numbers': condition_numbers.copy(),
                'nll_components': nll_components_list.copy(),
                'total_nll': float('inf'),
                'avg_nll': float('inf'),
                'min_nll': float('inf'),
                'max_nll': float('inf')
            })
            
        # Detailed NLL Component Analysis
        print(f"\nNLL Component Analysis - Iteration {iter}:")
        log_det_terms = []
        quadratic_terms = []
        constant_terms = []
        
        for i, nll_comp in enumerate(nll_components_list):
            if isinstance(nll_comp, dict) and all(k in nll_comp for k in ['log_det_term', 'quadratic_term', 'constant_term']):
                log_det = nll_comp['log_det_term']
                quadratic = nll_comp['quadratic_term']
                constant = nll_comp['constant_term']
                
                print(f"  Agent {i+1}: LogDet={log_det:.6f}, Quadratic={quadratic:.6f}, Constant={constant:.6f}")
                
                if not (np.isinf(log_det) or np.isnan(log_det)):
                    log_det_terms.append(log_det)
                if not (np.isinf(quadratic) or np.isnan(quadratic)):
                    quadratic_terms.append(quadratic)
                if not (np.isinf(constant) or np.isnan(constant)):
                    constant_terms.append(constant)
            else:
                print(f"  Agent {i+1}: Invalid NLL components")
        
        # Component summaries
        if log_det_terms:
            print(f"  LogDet Summary: Avg={np.mean(log_det_terms):.6f}, Min={np.min(log_det_terms):.6f}, Max={np.max(log_det_terms):.6f}")
        if quadratic_terms:
            print(f"  Quadratic Summary: Avg={np.mean(quadratic_terms):.6f}, Min={np.min(quadratic_terms):.6f}, Max={np.max(quadratic_terms):.6f}")
        if constant_terms:
            print(f"  Constant Summary: Avg={np.mean(constant_terms):.6f}, Min={np.min(constant_terms):.6f}, Max={np.max(constant_terms):.6f}")
        
        # Kernel Matrix Condition Numbers for this iteration
        print(f"Kernel Matrix Condition Numbers for Expressivity - Iteration {iter}:")
        for i, cond_num in enumerate(condition_numbers):
            if cond_num < 1e12:
                status = "Good"
            elif cond_num < 1e15:
                status = "Moderate"
            else:
                status = "Poor"
            print(f"  Agent {i+1}: {cond_num:.2e} ({status})")
        
        avg_cond = np.mean(condition_numbers)
        max_cond = np.max(condition_numbers)
        min_cond = np.min(condition_numbers)
        print(f"  Summary: Avg={avg_cond:.2e}, Min={min_cond:.2e}, Max={max_cond:.2e}")
        print()

        # Cross-Validation on Consensus Parameters - DISTRIBUTED APPROACH
        # Each agent evaluates on LOCAL data only (no data sharing)
        print(f"\nCross-Validation Analysis - Iteration {iter} (DISTRIBUTED, LOCAL DATA)")
        print(f"  Evaluating consensus parameters: {z}")
        print(f"  Each agent evaluates on their own local data (no data sharing)")
        
        try:
            # Perform local CV for each agent in parallel
            cv_results_local = []
            
            for i, (X_agent, Y_agent) in enumerate(agent_data_splits):
                cv_result = k_fold_cross_validation_local(
                    agent_id=f"agent_{i+1}",
                    X_local=X_agent,
                    Y_local=Y_agent,
                    consensus_params=z,
                    num_qubits=num_qubits, 
                    num_layers=num_layers, 
                    noise_std=noise_std,
                    k_folds=k_folds, 
                    use_parameter_shift=use_parameter_shift,
                    encoding_type=args.encoding, 
                    kernel_type=args.kernel_type,
                    measurement=args.measurement, 
                    outer_kernel=outer_kernel,
                    outer_kernel_params=outer_kernel_params, 
                    regularization=args.regularization,
                    random_seed=args.seed + iter  # Different seed each iteration
                )
                cv_results_local.append(cv_result)
            
            # Aggregate results across agents
            cv_aggregated = aggregate_local_cv_results(cv_results_local)
            
            cv_score = cv_aggregated['aggregated_nlpd']
            cv_std = cv_aggregated['aggregated_nlpd_std']
            cv_r2 = cv_aggregated['aggregated_r2']
            
            # Print per-agent results
            print(f"\n  Per-Agent CV Results (Local Data):")
            for i, agent_result in enumerate(cv_results_local):
                agent_status = "✅" if not np.isinf(agent_result['mean_nlpd']) else "❌"
                print(f"    {agent_status} Agent {i+1}: NLPD={agent_result['mean_nlpd']:.4f}±{agent_result['std_nlpd']:.4f}, R²={agent_result['mean_r2']:.4f}")
            
            # Print aggregated results
            if not np.isinf(cv_score):
                status = "✅ Good" if cv_score < 2.0 else "⚠️ Moderate" if cv_score < 5.0 else "❌ Poor"
            else:
                status = "❌ Failed"
            
            print(f"\n  Aggregated CV Results (Distributed):")
            print(f"    Aggregated NLPD: {cv_score:.4f}±{cv_std:.4f} (mean±std across {cv_aggregated['num_valid_agents']} agents)")
            print(f"    Aggregated R²:   {cv_r2:.4f}")
            print(f"    Status:          {status}")
            
            # Track best CV score for early stopping
            if cv_score < cv_best:
                cv_best = cv_score
                z_best_cv = z.copy()
                patience_counter = 0
                print(f"    🎯 New best CV score! Saved consensus parameters.")
            else:
                patience_counter += 1
                print(f"    No improvement. Patience: {patience_counter}/{cv_patience}")
            
            cv_score_history.append({
                'iteration': iter,
                'consensus_cv_score': cv_score,
                'cv_score_std': cv_std,
                'cv_r2': cv_r2,
                'num_valid_agents': cv_aggregated['num_valid_agents'],
                'total_agents': cv_aggregated['total_agents'],
                'distributed_local_cv': True,
                'consensus_params': z.copy(),
                'agent_cv_results': cv_results_local
            })
            
        except Exception as e:
            print(f"  ❌ CV evaluation failed: {e}")
            patience_counter += 1
            cv_score_history.append({
                'iteration': iter,
                'consensus_cv_score': float('inf'),
                'cv_score_std': float('inf'),
                'cv_r2': -float('inf'),
                'num_valid_agents': 0,
                'total_agents': len(agent_data_splits),
                'distributed_local_cv': True,
                'consensus_params': z.copy(),
                'agent_cv_results': []
            })
        
        # Calculate convergence metrics
        theta_z_norms = np.linalg.norm(z - theta, axis=1)
        max_norm = np.max(theta_z_norms)
        z_change = np.linalg.norm(z - z_old)
        
        iteration_time = time.time() - iteration_start
        print(f"Iteration {iter} took: {iteration_time:.4f}s")
        print(f"Max ||z - theta_i||: {max_norm:.6f}")
        print(f"||z_new - z_old||: {z_change:.6f}")
        
        # Print hyperparameters comparison
        print(f"\nConsensus Parameters after iteration {iter}:")
        print(f"Current consensus z (circuit): {z[:-1]}")
        print(f"Current consensus z (noise):   {z[-1]:.6f}")
        print(f"Best CV score so far: {cv_best:.6f}")
        if z_best_cv is not None:
            print(f"Best CV parameters: {z_best_cv}")
        
        # Original ground truth comparison (if available)
        if ground_truth_params is not None:
            print(f"\nGround Truth Comparison:")
            print(f"Ground truth params: {ground_truth_params}")
            
            # Use fast Euclidean distance during optimization for speed
            # Only compare circuit parameters (z[:-1]) with ground_truth_params
            # Use optimized Riemannian distance - much faster than manifold.distance()
            param_error = fast_riemannian_distance(z[:-1], ground_truth_params)
            euclidean_error = np.linalg.norm(z[:-1] - ground_truth_params)
            print(f"Riemannian distance ||z_circuit - ground_truth||: {param_error:.6f}")
            print(f"Euclidean distance (for comparison): {euclidean_error:.6f}")
                
            # Best solution tracking (ground truth)
            error_history.append(float(np.round(param_error,4)))
            if param_error < error_best:
                error_best = param_error
                z_best = z.copy()
        else:
            print(f"\n(No ground truth available for classical dataset)")
            param_error = float('inf')  # Set to infinity for classical datasets
            
        print(f"Individual agent params (theta):")
        for i, theta_i in enumerate(theta):
            print(f"  Agent {i+1}: {theta_i}")
        print()  # Add empty line for better readability

        # Modified convergence criteria
        consensus_converged = np.all(theta_z_norms < tolADMM)
        early_stopping = patience_counter >= cv_patience
        max_iterations = iter >= maxiter
        
        if consensus_converged:
            print(f"\n✅ Converged: Consensus reached after {iter} iterations!")
            break
        elif early_stopping:
            print(f"\n⏰ Early stopping: No CV improvement for {cv_patience} iterations")
            print(f"Using best consensus parameters from iteration {iter - patience_counter}")
            z = z_best_cv.copy()  # Use best CV parameters
            break
        elif max_iterations:
            print(f"\n⏱️ Max iterations reached: {maxiter}")
            if z_best_cv is not None:
                z = z_best_cv.copy()  # Use best CV parameters
            break

    total_time = time.time() - admm_start_time
    print(f"\nTotal ADMM optimization time: {total_time:.4f}s")
    print(f"Average time per iteration: {total_time/iter:.4f}s")
    print("Gradient method used: riemannian")
    print(f"Quantum differentiation method: {args.differentiation}")
    print(f"Parallel parameter evaluation: {'Enabled' if use_parameter_shift else 'Disabled'}")
    
    # Final hyperparameters summary with CV-based optimization
    print(f"\n{'='*50}")
    print("FINAL HYPERPARAMETERS SUMMARY (CV-based)")
    print(f"{'='*50}")
    print(f"🎯 PRIMARY OPTIMIZATION METHOD: Cross-Validation (Realistic)")
    print(f"Best CV-NLPD score: {cv_best:.6f}")
    print(f"Final consensus params (circuit): {z[:-1]}")
    print(f"Final consensus params (noise):   {z[-1]:.6f}")
    if z_best_cv is not None:
        print(f"Best CV params (circuit): {z_best_cv[:-1]}")
        print(f"Best CV params (noise):   {z_best_cv[-1]:.6f}")
        print(f"✅ CV-optimized parameters will be used for prediction")
    else:
        print(f"⚠️  No CV-optimized parameters available, using final iteration")
        
    # Ground truth comparison (for analysis only, not optimization)
    if ground_truth_params is not None:
        print(f"\n📊 GROUND TRUTH ANALYSIS (for comparison only):")
        print(f"Ground truth params: {ground_truth_params}")
        print(f"Best ADMM (z circuit): {z_best[:-1]}")
        print(f"Best ||z_circuit - ground_truth||: {error_best:.6f}")
        
        # Calculate final error using appropriate distance metric (circuit params only)
        final_error = fast_riemannian_distance(z[:-1], ground_truth_params)
        euclidean_final_error = np.linalg.norm(z[:-1] - ground_truth_params)
        print(f"Final Riemannian distance: {final_error:.6f}")
        print(f"Final Euclidean distance:  {euclidean_final_error:.6f}")
            
        print(f"Parameter recovery: {'🎯 EXCELLENT!' if final_error < 1.0 else 'Good' if final_error < 3.0 else 'Needs improvement'}")
        print(f"Error history: {error_history}")
        print(f"Note: Ground truth comparison is for analysis only")
    else:
        print(f"\n(No ground truth available for classical dataset)")
        
    print(f"\nFinal agent params (theta) - consensus check:")
    for i, theta_i in enumerate(theta):
        consensus_error = fast_riemannian_distance(z, theta_i)
        print(f"  Agent {i+1}: {theta_i} (||z - theta_{i+1}||: {consensus_error:.6f})")
    print(f"{'='*50}")
    
    # CV Score Evolution Analysis
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION SCORE EVOLUTION")
    print(f"{'='*50}")
    if cv_score_history:
        print(f"Total iterations: {len(cv_score_history)}")
        
        # Show first and last few iterations
        num_to_show = min(3, len(cv_score_history))
        print(f"\nFirst {num_to_show} iterations:")
        for i in range(num_to_show):
            cv_data = cv_score_history[i]
            print(f"  Iteration {cv_data['iteration']}: CV-NLPD={cv_data['consensus_cv_score']:.4f}±{cv_data['cv_score_std']:.4f}, R²={cv_data['cv_r2']:.4f}")
        
        if len(cv_score_history) > 6:
            print(f"  ...")
        
        if len(cv_score_history) > num_to_show:
            print(f"Last {num_to_show} iterations:")
            for i in range(max(num_to_show, len(cv_score_history) - num_to_show), len(cv_score_history)):
                cv_data = cv_score_history[i]
                print(f"  Iteration {cv_data['iteration']}: CV-NLPD={cv_data['consensus_cv_score']:.4f}±{cv_data['cv_score_std']:.4f}, R²={cv_data['cv_r2']:.4f}")
        
        # Show improvement
        if len(cv_score_history) > 1:
            initial_cv = cv_score_history[0]['consensus_cv_score']
            final_cv = cv_score_history[-1]['consensus_cv_score']
            if not np.isinf(initial_cv) and not np.isinf(final_cv):
                improvement = initial_cv - final_cv
                print(f"\nCV Score Improvement:")
                print(f"  Initial CV-NLPD: {initial_cv:.6f}")
                print(f"  Final CV-NLPD:   {final_cv:.6f}")
                print(f"  Improvement:     {improvement:.6f} ({'✅ Better' if improvement > 0 else '❌ Worse'})")
        
        print(f"  Best CV-NLPD: {cv_best:.6f}")
    else:
        print("No CV score history available")
    print(f"{'='*50}")
    
    # NLL Loss Convergence Analysis
    print(f"\n{'='*50}")
    print("NLL LOSS CONVERGENCE ANALYSIS")
    print(f"{'='*50}")
    if nll_loss_history:
        print(f"Total iterations: {len(nll_loss_history)}")
        # Show first few and last few iterations
        print(f"\nNLL Loss Evolution:")
        num_to_show = min(3, len(nll_loss_history))
        
        print(f"First {num_to_show} iterations:")
        for i in range(num_to_show):
            loss_data = nll_loss_history[i]
            print(f"  Iteration {loss_data['iteration']}: Avg={loss_data['avg_nll']:.6f}, Min={loss_data['min_nll']:.6f}, Max={loss_data['max_nll']:.6f}")
        
        if len(nll_loss_history) > 6:
            print(f"  ...")
        
        if len(nll_loss_history) > num_to_show:
            print(f"Last {num_to_show} iterations:")
            for i in range(max(num_to_show, len(nll_loss_history) - num_to_show), len(nll_loss_history)):
                loss_data = nll_loss_history[i]
                print(f"  Iteration {loss_data['iteration']}: Avg={loss_data['avg_nll']:.6f}, Min={loss_data['min_nll']:.6f}, Max={loss_data['max_nll']:.6f}")
        
        # Show initial vs final comparison
        initial_loss = nll_loss_history[0]
        final_loss = nll_loss_history[-1]
        print(f"\nLoss Reduction:")
        print(f"  Initial average NLL: {initial_loss['avg_nll']:.6f}")
        print(f"  Final average NLL:   {final_loss['avg_nll']:.6f}")
        if not np.isinf(initial_loss['avg_nll']) and not np.isinf(final_loss['avg_nll']):
            improvement = initial_loss['avg_nll'] - final_loss['avg_nll']
            print(f"  Improvement: {improvement:.6f} ({improvement/initial_loss['avg_nll']*100:.2f}%)")
        
        # Best overall loss
        valid_avg_losses = [h['avg_nll'] for h in nll_loss_history if not np.isinf(h['avg_nll'])]
        if valid_avg_losses:
            best_avg_loss = min(valid_avg_losses)
            best_iter = next(h['iteration'] for h in nll_loss_history if h['avg_nll'] == best_avg_loss)
            print(f"  Best average NLL: {best_avg_loss:.6f} (iteration {best_iter})")
    else:
        print("No NLL loss history available")
    print(f"{'='*50}")
    
    # NLL Loss vs Hyperparameter Error Comparison
    if ground_truth_params is not None and nll_loss_history and error_history:
        print(f"\n{'='*50}")
        print("NLL LOSS vs HYPERPARAMETER ERROR COMPARISON")
        print(f"{'='*50}")
        
        # Find iteration with lowest NLL loss
        valid_nll_data = [(i, h['avg_nll']) for i, h in enumerate(nll_loss_history) if not np.isinf(h['avg_nll']) and not np.isnan(h['avg_nll'])]
        
        if valid_nll_data:
            # Find iteration with lowest NLL
            min_nll_idx, min_nll_value = min(valid_nll_data, key=lambda x: x[1])
            min_nll_iter = nll_loss_history[min_nll_idx]['iteration']
            
            # Find iteration with lowest hyperparameter error
            min_error_idx = np.argmin(error_history)
            min_error_value = error_history[min_error_idx]
            min_error_iter = min_error_idx + 1  # iterations are 1-indexed
            
            print(f"Lowest NLL Loss:")
            print(f"  Iteration: {min_nll_iter}")
            print(f"  NLL Loss: {min_nll_value:.6f}")
            print(f"  Hyperparameter Error: {error_history[min_nll_idx]:.6f}")
            
            print(f"\nLowest Hyperparameter Error:")
            print(f"  Iteration: {min_error_iter}")
            print(f"  Hyperparameter Error: {min_error_value:.6f}")
            if min_error_idx < len(nll_loss_history):
                corresponding_nll = nll_loss_history[min_error_idx]['avg_nll']
                print(f"  NLL Loss: {corresponding_nll:.6f}")
            else:
                print(f"  NLL Loss: N/A (no data for this iteration)")
            
            # Check if they align
            alignment_check = min_nll_iter == min_error_iter
            print(f"\nAlignment Analysis:")
            print(f"  Do lowest NLL and lowest error occur at same iteration? {'✅ YES' if alignment_check else '❌ NO'}")
            
            if not alignment_check:
                nll_gap = abs(min_nll_value - (nll_loss_history[min_error_idx]['avg_nll'] if min_error_idx < len(nll_loss_history) else float('inf')))
                error_gap = abs(min_error_value - error_history[min_nll_idx])
                print(f"  NLL difference: {nll_gap:.6f}")
                print(f"  Error difference: {error_gap:.6f}")
                print(f"  Iteration difference: {abs(min_nll_iter - min_error_iter)} iterations")
            
            # Calculate correlation between NLL and error
            valid_iterations = min(len(nll_loss_history), len(error_history))
            nll_values = [nll_loss_history[i]['avg_nll'] for i in range(valid_iterations) 
                         if not np.isinf(nll_loss_history[i]['avg_nll']) and not np.isnan(nll_loss_history[i]['avg_nll'])]
            error_values = [error_history[i] for i in range(len(nll_values))]
            
            if len(nll_values) > 1 and len(error_values) > 1:
                correlation = np.corrcoef(nll_values, error_values)[0, 1]
                print(f"\nCorrelation Analysis:")
                print(f"  Pearson correlation (NLL vs Error): {correlation:.4f}")
                if correlation > 0.7:
                    print(f"  Strong positive correlation - Lower error tends to mean lower NLL ✅")
                elif correlation > 0.3:
                    print(f"  Moderate positive correlation")
                elif correlation > -0.3:
                    print(f"  Weak correlation")
                elif correlation > -0.7:
                    print(f"  Moderate negative correlation")
                else:
                    print(f"  Strong negative correlation - Lower error means higher NLL ❌")
                
                # Detailed NLL Component Correlation Analysis
                print(f"\n🔍 NLL Component Correlation Analysis:")
                print(f"  Analyzing which NLL term correlates best with hyperparameter error...")
                
                # Extract component values across iterations
                log_det_series = []
                quadratic_series = []
                constant_series = []
                
                for i in range(valid_iterations):
                    if i < len(nll_loss_history) and 'nll_components' in nll_loss_history[i]:
                        components = nll_loss_history[i]['nll_components']
                        # Average across agents for this iteration
                        valid_log_det = []
                        valid_quadratic = []
                        valid_constant = []
                        
                        for comp in components:
                            if isinstance(comp, dict) and all(k in comp for k in ['log_det_term', 'quadratic_term', 'constant_term']):
                                if not (np.isinf(comp['log_det_term']) or np.isnan(comp['log_det_term'])):
                                    valid_log_det.append(comp['log_det_term'])
                                if not (np.isinf(comp['quadratic_term']) or np.isnan(comp['quadratic_term'])):
                                    valid_quadratic.append(comp['quadratic_term'])
                                if not (np.isinf(comp['constant_term']) or np.isnan(comp['constant_term'])):
                                    valid_constant.append(comp['constant_term'])
                        
                        if valid_log_det:
                            log_det_series.append(np.mean(valid_log_det))
                        if valid_quadratic:
                            quadratic_series.append(np.mean(valid_quadratic))
                        if valid_constant:
                            constant_series.append(np.mean(valid_constant))
                
                # Compute correlations for each component
                if len(log_det_series) > 1 and len(log_det_series) == len(error_values[:len(log_det_series)]):
                    log_det_corr = np.corrcoef(log_det_series, error_values[:len(log_det_series)])[0, 1]
                    print(f"  📊 Log Determinant vs Error: {log_det_corr:.4f}", end="")
                    if abs(log_det_corr) > 0.7:
                        print(" (STRONG)")
                    elif abs(log_det_corr) > 0.3:
                        print(" (MODERATE)")
                    else:
                        print(" (WEAK)")
                else:
                    log_det_corr = float('nan')
                    print(f"  📊 Log Determinant vs Error: N/A (insufficient data)")
                
                if len(quadratic_series) > 1 and len(quadratic_series) == len(error_values[:len(quadratic_series)]):
                    quadratic_corr = np.corrcoef(quadratic_series, error_values[:len(quadratic_series)])[0, 1]
                    print(f"  📊 Quadratic Form vs Error: {quadratic_corr:.4f}", end="")
                    if abs(quadratic_corr) > 0.7:
                        print(" (STRONG)")
                    elif abs(quadratic_corr) > 0.3:
                        print(" (MODERATE)")
                    else:
                        print(" (WEAK)")
                else:
                    quadratic_corr = float('nan')
                    print(f"  📊 Quadratic Form vs Error: N/A (insufficient data)")
                
                if len(constant_series) > 1 and len(constant_series) == len(error_values[:len(constant_series)]):
                    constant_corr = np.corrcoef(constant_series, error_values[:len(constant_series)])[0, 1]
                    print(f"  📊 Constant Term vs Error: {constant_corr:.4f}", end="")
                    if abs(constant_corr) > 0.7:
                        print(" (STRONG)")
                    elif abs(constant_corr) > 0.3:
                        print(" (MODERATE)")
                    else:
                        print(" (WEAK)")
                else:
                    constant_corr = float('nan')
                    print(f"  📊 Constant Term vs Error: N/A (insufficient data)")
                
                # Determine which component correlates best
                valid_correlations = {}
                if not np.isnan(log_det_corr):
                    valid_correlations['Log Determinant'] = abs(log_det_corr)
                if not np.isnan(quadratic_corr):
                    valid_correlations['Quadratic Form'] = abs(quadratic_corr)
                if not np.isnan(constant_corr):
                    valid_correlations['Constant Term'] = abs(constant_corr)
                
                if valid_correlations:
                    best_component = max(valid_correlations, key=valid_correlations.get)
                    best_correlation = valid_correlations[best_component]
                    print(f"\n  🏆 BEST PREDICTOR: {best_component} (|correlation| = {best_correlation:.4f})")
                    
                    if best_correlation > 0.7:
                        print(f"     💡 {best_component} strongly predicts hyperparameter quality!")
                    elif best_correlation > 0.3:
                        print(f"     💡 {best_component} moderately predicts hyperparameter quality")
                    else:
                        print(f"     ⚠️  {best_component} weakly predicts hyperparameter quality")
                else:
                    print(f"\n  ❌ No valid component correlations found")
            
            # Summary recommendation
            print(f"\nRecommendation:")
            if alignment_check:
                print(f"  🎯 OPTIMAL: Lowest NLL and lowest hyperparameter error align perfectly!")
            elif abs(min_nll_iter - min_error_iter) <= 2:
                print(f"  ✅ GOOD: Lowest NLL and lowest error are close (within 2 iterations)")
            else:
                print(f"  ⚠️  CAUTION: Significant gap between lowest NLL and lowest error")
                print(f"     Consider using iteration {min_error_iter} parameters for better generalization")
        else:
            print("Insufficient valid NLL data for comparison")
        print(f"{'='*50}")
    
    # Optional: Test model performance on held-out test set
    # Note: This would require implementing a test function for your quantum GP
    # that can make predictions using the trained quantum kernel parameters
    print(f"\nTraining completed!")
    print(f"Final consensus parameters shape: {z.shape}")
    print(f"Training data used: {X_train.shape[0]} samples")
    print(f"Test data available: {X_test.shape[0]} samples")
    
    # === PREDICTION AND EVALUATION ON TEST SET ===
    print(f"\n{'='*60}")
    print("PREDICTION AND EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    # Determine which hyperparameters to use for prediction
    hyperparams_to_use = None
    hyperparams_source = ""
    
    # Use CV-based hyperparameter selection (realistic approach)
    if z_best_cv is not None:
        # Use CV-optimized hyperparameters (best validation performance)
        hyperparams_to_use = z_best_cv
        hyperparams_source = f"CV-optimized (best CV-NLPD: {cv_best:.6f})"
        print(f"Using CV-OPTIMIZED hyperparameters for prediction (best validation performance)")
    else:
        # Use final hyperparameters as fallback
        hyperparams_to_use = z
        hyperparams_source = "final (last iteration)"
        print(f"Using FINAL hyperparameters for prediction (last iteration)")
    
    print(f"Hyperparameters source: {hyperparams_source}")
    print(f"Hyperparameters values: {hyperparams_to_use}")
    
    # === STORE LOCAL COVARIANCE STATES FOR FACT-GP PREDICTION ===
    print(f"\n{'='*60}")
    print("COMPUTING LOCAL COVARIANCE STATES (FACT-GP)")
    print(f"{'='*60}")
    
    agent_states = compute_and_store_agent_covariance_states(
        agent_data_splits=agent_data_splits,
        quantum_kernel_params=hyperparams_to_use,
        num_qubits=num_qubits,
        num_layers=num_layers,
        noise_std=noise_std,
        use_parameter_shift=use_parameter_shift,
        encoding_type=args.encoding,
        kernel_type=args.kernel_type,
        measurement=args.measurement,
        outer_kernel=outer_kernel,
        outer_kernel_params=outer_kernel_params,
        regularization=args.regularization
    )
    
    # Make predictions on test set
    try:
        print(f"\nMaking predictions on test set...")
        prediction_start_time = time.time()
        
        # Use prediction method wrapper to route to appropriate aggregation method
        y_pred_mean, y_pred_std, prediction_time_method, prediction_method_used = make_predictions_with_method(
            prediction_method=args.prediction_method,
            agent_states=agent_states,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            quantum_kernel_params=hyperparams_to_use,
            num_qubits=num_qubits,
            num_layers=num_layers,
            noise_std=noise_std,
            use_parameter_shift=use_parameter_shift,
            encoding_type=args.encoding,
            kernel_type=args.kernel_type,
            measurement=args.measurement,
            outer_kernel=outer_kernel,
            outer_kernel_params=outer_kernel_params,
            regularization=args.regularization,
            y_train_std=y_train_std
        )
        y_pred_var = y_pred_std ** 2
        
        prediction_time = time.time() - prediction_start_time
        print(f"Total prediction time: {prediction_time:.4f}s")
        
        # ===== DIAGNOSTIC CHECKS: Verify quantum kernel can represent full target range =====
        print(f"\n{'='*70}")
        print("DIAGNOSTIC: Kernel Expressiveness Check")
        print(f"{'='*70}")
        print(f"Target variable statistics:")
        print(f"  Training Y range: [{Y_train.min():.6f}, {Y_train.max():.6f}]")
        print(f"  Test Y range: [{Y_test.min():.6f}, {Y_test.max():.6f}]")
        print(f"  Training Y std: {np.std(Y_train):.6f}")
        
        print(f"\nPrediction statistics:")
        print(f"  Predicted Y range: [{y_pred_mean.min():.6f}, {y_pred_mean.max():.6f}]")
        print(f"  Predicted Y std: {np.std(y_pred_mean):.6f}")
        
        # Calculate coverage
        train_range = Y_train.max() - Y_train.min()
        pred_range = y_pred_mean.max() - y_pred_mean.min()
        test_range = Y_test.max() - Y_test.min()
        coverage_vs_train = pred_range / train_range if train_range > 0 else 0
        coverage_vs_test = pred_range / test_range if test_range > 0 else 0
        
        print(f"\nRange Coverage Analysis:")
        print(f"  Training range: {train_range:.6f}")
        print(f"  Test range: {test_range:.6f}")
        print(f"  Prediction range: {pred_range:.6f}")
        print(f"  Coverage vs training: {coverage_vs_train*100:.1f}%")
        print(f"  Coverage vs test: {coverage_vs_test*100:.1f}%")
        
        if coverage_vs_train < 0.5:
            print(f"  ⚠️  WARNING: Predictions cover only {coverage_vs_train*100:.1f}% of training range")
            print(f"     This indicates kernel may be ill-conditioned or underfitting")
        
        print(f"{'='*70}\n")
        
        # Evaluate predictions
        print(f"\nEvaluating prediction quality...")
        test_metrics = evaluate_predictions(
            Y_true=Y_test,
            Y_pred=y_pred_mean,
            Y_pred_var=y_pred_var,
            dataset_type="Test"
        )
        
        # Also evaluate on training set for comparison (on subset to save time)
        print(f"\nEvaluating on training set for comparison...")
        # Sample 100 random points from training set for faster evaluation with many agents
        n_train_eval = min(100, len(X_train))
        train_eval_indices = np.random.choice(len(X_train), size=n_train_eval, replace=False)
        X_train_eval = X_train[train_eval_indices]
        Y_train_eval = Y_train[train_eval_indices]
        
        try:
            # Use prediction method wrapper for training set evaluation
            y_train_pred_mean, y_train_pred_std, _, _ = make_predictions_with_method(
                prediction_method=args.prediction_method,
                agent_states=agent_states,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_train_eval,
                quantum_kernel_params=hyperparams_to_use,
                num_qubits=num_qubits,
                num_layers=num_layers,
                noise_std=noise_std,
                use_parameter_shift=use_parameter_shift,
                encoding_type=args.encoding,
                kernel_type=args.kernel_type,
                measurement=args.measurement,
                outer_kernel=outer_kernel,
                outer_kernel_params=outer_kernel_params,
                regularization=args.regularization,
                y_train_std=y_train_std
            )
            y_train_pred_var = y_train_pred_std ** 2
            
            train_metrics = evaluate_predictions(
                Y_true=Y_train_eval,
                Y_pred=y_train_pred_mean,
                Y_pred_var=y_train_pred_var,
                dataset_type="Training (subset)"
            )
            
        except Exception as e:
            print(f"Warning: Could not evaluate training set predictions: {e}")
            train_metrics = None
        
        # === GROUND TRUTH vs BEST TRAINED HYPERPARAMETERS COMPARISON ===
        if ground_truth_params is not None and z_best is not None:
            print(f"\n{'='*60}")
            print("GROUND TRUTH vs BEST TRAINED HYPERPARAMETERS COMPARISON")
            print(f"{'='*60}")
            
            # Compute agent states using ground truth hyperparameters for comparison
            print(f"\nComputing agent states with GROUND TRUTH hyperparameters...")
            try:
                agent_states_gt = compute_and_store_agent_covariance_states(
                    agent_data_splits=agent_data_splits,
                    quantum_kernel_params=ground_truth_params,
                    num_qubits=num_qubits,
                    num_layers=num_layers,
                    noise_std=noise_std,
                    use_parameter_shift=use_parameter_shift,
                    encoding_type=args.encoding,
                    kernel_type=args.kernel_type,
                    measurement=args.measurement,
                    outer_kernel=outer_kernel,
                    outer_kernel_params=outer_kernel_params,
                    regularization=args.regularization
                )
            except Exception as e:
                print(f"Warning: Could not compute ground truth agent states: {e}")
                agent_states_gt = None
            
            # Make predictions with ground truth hyperparameters
            print(f"Making predictions with GROUND TRUTH hyperparameters...")
            try:
                gt_prediction_start = time.time()
                if agent_states_gt is not None:
                    # Use prediction method wrapper with ground truth agent states
                    y_pred_gt_mean, y_pred_gt_std, gt_pred_time, _ = make_predictions_with_method(
                        prediction_method=args.prediction_method,
                        agent_states=agent_states_gt,
                        X_train=X_train,
                        Y_train=Y_train,
                        X_test=X_test,
                        quantum_kernel_params=ground_truth_params,
                        num_qubits=num_qubits,
                        num_layers=num_layers,
                        noise_std=noise_std,
                        use_parameter_shift=use_parameter_shift,
                        encoding_type=args.encoding,
                        kernel_type=args.kernel_type,
                        measurement=args.measurement,
                        outer_kernel=outer_kernel,
                        outer_kernel_params=outer_kernel_params,
                        regularization=args.regularization,
                        y_train_std=y_train_std
                    )
                    y_pred_gt_var = y_pred_gt_std ** 2
                else:
                    # Fallback to original prediction function
                    y_pred_gt_mean, y_pred_gt_var, _, _, _ = predict_quantum_gp(
                        X_train=X_train,
                        Y_train=Y_train, 
                        X_test=X_test,
                        quantum_kernel_params=ground_truth_params,
                        num_qubits=num_qubits,
                        num_layers=num_layers,
                        noise_std=noise_std,
                        use_parameter_shift=use_parameter_shift,
                        encoding_type=args.encoding,
                        kernel_type=args.kernel_type,
                        measurement=args.measurement,
                        outer_kernel=outer_kernel,
                        outer_kernel_params=outer_kernel_params,
                        regularization=args.regularization
                    )
                gt_prediction_time = time.time() - gt_prediction_start
                print(f"Ground truth prediction time: {gt_prediction_time:.4f}s")
                
                # Evaluate ground truth predictions
                gt_metrics = evaluate_predictions(
                    Y_true=Y_test,
                    Y_pred=y_pred_gt_mean,
                    Y_pred_var=y_pred_gt_var,
                    dataset_type="Test (Ground Truth Hyperparams)"
                )
                
                # Compare metrics between ground truth and best trained
                print(f"\n{'='*60}")
                print("PREDICTION ACCURACY COMPARISON")
                print(f"{'='*60}")
                
                print(f"Ground Truth Hyperparameters: {ground_truth_params}")
                print(f"Best Trained Hyperparameters:  {z_best}")
                print(f"Hyperparameter Error (L2 norm): {error_best:.6f}")
                
                print(f"\nPrediction Metrics Comparison:")
                print(f"                                Ground Truth    Best Trained    Improvement")
                print(f"                                ------------    ------------    -----------")
                
                # R² Score comparison
                r2_improvement = gt_metrics['r2'] - test_metrics['r2']
                r2_improvement_pct = (r2_improvement / max(abs(test_metrics['r2']), 1e-10)) * 100
                print(f"R² Score:                      {gt_metrics['r2']:12.6f}    {test_metrics['r2']:12.6f}    {r2_improvement:+11.6f} ({r2_improvement_pct:+6.2f}%)")
                
                # RMSE comparison (lower is better)
                rmse_improvement = test_metrics['rmse'] - gt_metrics['rmse']  # Positive means GT is better
                rmse_improvement_pct = (rmse_improvement / max(gt_metrics['rmse'], 1e-10)) * 100
                print(f"RMSE:                          {gt_metrics['rmse']:12.6f}    {test_metrics['rmse']:12.6f}    {rmse_improvement:+11.6f} ({rmse_improvement_pct:+6.2f}%)")
                
                # MSE comparison (lower is better)
                mse_improvement = test_metrics['mse'] - gt_metrics['mse']  # Positive means GT is better
                mse_improvement_pct = (mse_improvement / max(gt_metrics['mse'], 1e-10)) * 100
                print(f"MSE:                           {gt_metrics['mse']:12.6f}    {test_metrics['mse']:12.6f}    {mse_improvement:+11.6f} ({mse_improvement_pct:+6.2f}%)")
                
                # MAE comparison (lower is better)
                mae_improvement = test_metrics['mae'] - gt_metrics['mae']  # Positive means GT is better
                mae_improvement_pct = (mae_improvement / max(gt_metrics['mae'], 1e-10)) * 100
                print(f"MAE:                           {gt_metrics['mae']:12.6f}    {test_metrics['mae']:12.6f}    {mae_improvement:+11.6f} ({mae_improvement_pct:+6.2f}%)")
                
                # Max Error comparison (lower is better)
                max_error_improvement = test_metrics['max_error'] - gt_metrics['max_error']  # Positive means GT is better
                max_error_improvement_pct = (max_error_improvement / max(gt_metrics['max_error'], 1e-10)) * 100
                print(f"Max Absolute Error:            {gt_metrics['max_error']:12.6f}    {test_metrics['max_error']:12.6f}    {max_error_improvement:+11.6f} ({max_error_improvement_pct:+6.2f}%)")
                
                # NLPD comparison (lower is better) - only if both have uncertainty quantification
                if 'nlpd' in gt_metrics and 'nlpd' in test_metrics:
                    nlpd_improvement = test_metrics['nlpd'] - gt_metrics['nlpd']  # Positive means GT is better
                    nlpd_improvement_pct = (nlpd_improvement / max(gt_metrics['nlpd'], 1e-10)) * 100
                    print(f"NLPD (Uncertainty Quality):   {gt_metrics['nlpd']:12.6f}    {test_metrics['nlpd']:12.6f}    {nlpd_improvement:+11.6f} ({nlpd_improvement_pct:+6.2f}%)")
                    has_nlpd_comparison = True
                else:
                    print(f"NLPD (Uncertainty Quality):   {'N/A':>12}    {'N/A':>12}    {'N/A':>11}     {'N/A':>6}")
                    has_nlpd_comparison = False
                
                # Range-based NRMSE comparison (lower is better) - always available
                if 'normalized_rmse_range' in gt_metrics and 'normalized_rmse_range' in test_metrics:
                    range_nrmse_improvement = test_metrics['normalized_rmse_range'] - gt_metrics['normalized_rmse_range']  # Positive means GT is better
                    range_nrmse_improvement_pct = (range_nrmse_improvement / max(gt_metrics['normalized_rmse_range'], 1e-10)) * 100
                    print(f"Range NRMSE:                   {gt_metrics['normalized_rmse_range']:12.6f}    {test_metrics['normalized_rmse_range']:12.6f}    {range_nrmse_improvement:+11.6f} ({range_nrmse_improvement_pct:+6.2f}%)")
                    has_range_nrmse_comparison = True
                else:
                    print(f"Range NRMSE:                   {'N/A':>12}    {'N/A':>12}    {'N/A':>11}     {'N/A':>6}")
                    has_range_nrmse_comparison = False
                
                # Uncertainty-based Normalized RMSE comparison (closer to 1.0 is better) - only if both have uncertainty quantification
                if 'normalized_rmse_uncertainty' in gt_metrics and 'normalized_rmse_uncertainty' in test_metrics:
                    # For uncertainty normalized RMSE, closer to 1.0 is better, so calculate how far each is from 1.0
                    gt_distance_from_1 = abs(gt_metrics['normalized_rmse_uncertainty'] - 1.0)
                    test_distance_from_1 = abs(test_metrics['normalized_rmse_uncertainty'] - 1.0)
                    nrmse_uncertainty_improvement = test_distance_from_1 - gt_distance_from_1  # Positive means GT is better (closer to 1.0)
                    nrmse_uncertainty_improvement_pct = (nrmse_uncertainty_improvement / max(gt_distance_from_1, 1e-10)) * 100
                    print(f"Uncertainty NRMSE (Calibr.):  {gt_metrics['normalized_rmse_uncertainty']:12.6f}    {test_metrics['normalized_rmse_uncertainty']:12.6f}    {-nrmse_uncertainty_improvement:+11.6f} ({-nrmse_uncertainty_improvement_pct:+6.2f}%)")
                    has_nrmse_comparison = True
                else:
                    print(f"Uncertainty NRMSE (Calibr.):  {'N/A':>12}    {'N/A':>12}    {'N/A':>11}     {'N/A':>6}")
                    has_nrmse_comparison = False
                
                # Summary analysis
                print(f"\n{'='*60}")
                print("HYPERPARAMETER IMPACT ANALYSIS")
                print(f"{'='*60}")
                
                # Determine overall performance impact
                significant_improvements = 0
                total_comparisons = 0
                
                metrics_analysis = [
                    ("R² Score", r2_improvement, r2_improvement_pct, "higher_better"),
                    ("RMSE", rmse_improvement, rmse_improvement_pct, "lower_better"), 
                    ("MSE", mse_improvement, mse_improvement_pct, "lower_better"),
                    ("MAE", mae_improvement, mae_improvement_pct, "lower_better"),
                    ("Max Error", max_error_improvement, max_error_improvement_pct, "lower_better")
                ]
                
                # Add NLPD to metrics analysis if available
                if has_nlpd_comparison:
                    metrics_analysis.append(("NLPD", nlpd_improvement, nlpd_improvement_pct, "lower_better"))
                
                # Add Range NRMSE to metrics analysis if available
                if has_range_nrmse_comparison:
                    metrics_analysis.append(("Range NRMSE", range_nrmse_improvement, range_nrmse_improvement_pct, "lower_better"))
                
                # Add Uncertainty Normalized RMSE to metrics analysis if available
                if has_nrmse_comparison:
                    metrics_analysis.append(("Uncertainty NRMSE", -nrmse_uncertainty_improvement, -nrmse_uncertainty_improvement_pct, "closer_to_1_better"))
                
                print(f"Impact Assessment:")
                for metric_name, improvement, improvement_pct, direction in metrics_analysis:
                    total_comparisons += 1
                    
                    if direction == "higher_better":
                        # For R², positive improvement means GT is better
                        if improvement > 0.01:  # Significant improvement threshold
                            significance = "🎯 SIGNIFICANT" 
                            significant_improvements += 1
                        elif improvement > 0.001:
                            significance = "✅ MODERATE"
                        elif improvement > -0.001:
                            significance = "➖ MINIMAL"
                        else:
                            significance = "❌ WORSE"
                    elif direction == "closer_to_1_better":
                        # For Normalized RMSE, closer to 1.0 is better
                        # improvement represents how much closer GT is to 1.0 than trained
                        if improvement > 0 and abs(improvement_pct) > 10:  # >10% closer to 1.0
                            significance = "🎯 SIGNIFICANT"
                            significant_improvements += 1
                        elif improvement > 0 and abs(improvement_pct) > 2:  # >2% closer to 1.0
                            significance = "✅ MODERATE"
                        elif abs(improvement_pct) <= 2:  # Within 2%
                            significance = "➖ MINIMAL"
                        else:
                            significance = "❌ WORSE"
                    else:
                        # For error metrics, positive improvement means GT is better (lower error)
                        if improvement > 0 and abs(improvement_pct) > 5:  # >5% improvement
                            significance = "🎯 SIGNIFICANT"
                            significant_improvements += 1
                        elif improvement > 0 and abs(improvement_pct) > 1:  # >1% improvement
                            significance = "✅ MODERATE"
                        elif abs(improvement_pct) <= 1:  # Within 1%
                            significance = "➖ MINIMAL"
                        else:
                            significance = "❌ WORSE"
                    
                    print(f"  {metric_name:15}: {significance} ({improvement_pct:+6.2f}%)")
                
                # Overall conclusion
                improvement_ratio = significant_improvements / total_comparisons
                print(f"\nOverall Hyperparameter Impact:")
                print(f"  Significant improvements: {significant_improvements}/{total_comparisons} metrics ({improvement_ratio*100:.1f}%)")
                
                if improvement_ratio >= 0.6:
                    conclusion = "🎯 CRITICAL: Ground truth hyperparameters provide substantially better predictions!"
                    recommendation = "The rotational parameter optimization is HIGHLY EFFECTIVE for prediction accuracy."
                elif improvement_ratio >= 0.4:
                    conclusion = "✅ IMPORTANT: Ground truth hyperparameters provide moderately better predictions."
                    recommendation = "The rotational parameter optimization has MODERATE IMPACT on prediction accuracy."
                elif improvement_ratio >= 0.2:
                    conclusion = "⚠️ MINOR: Ground truth hyperparameters provide slightly better predictions."
                    recommendation = "The rotational parameter optimization has LIMITED IMPACT on prediction accuracy."
                else:
                    conclusion = "➖ NEGLIGIBLE: Little difference between ground truth and trained hyperparameters."
                    recommendation = "The rotational parameter optimization has MINIMAL IMPACT on prediction accuracy."
                
                print(f"  Conclusion: {conclusion}")
                print(f"  Recommendation: {recommendation}")
                
                # Special focus on NLPD for quantum GP uncertainty quantification
                if has_nlpd_comparison:
                    print(f"\n{'='*60}")
                    print("NLPD ANALYSIS FOR QUANTUM GP UNCERTAINTY")
                    print(f"{'='*60}")
                    
                    print(f"NLPD measures the quality of uncertainty estimates in Gaussian Processes.")
                    print(f"For quantum GPs, this is particularly important because:")
                    print(f"  • Quantum circuits introduce complex parameter-dependent uncertainties")
                    print(f"  • Rotational parameters directly affect kernel smoothness and predictive variance")
                    print(f"  • NLPD captures both prediction accuracy AND uncertainty calibration")
                    
                    print(f"\nNLPD Results:")
                    print(f"  Ground Truth NLPD:    {gt_metrics['nlpd']:8.6f}")
                    print(f"  Trained NLPD:         {test_metrics['nlpd']:8.6f}")
                    print(f"  NLPD Improvement:     {nlpd_improvement:+8.6f} ({nlpd_improvement_pct:+6.2f}%)")
                    
                    if nlpd_improvement > 0.1:  # Significant NLPD improvement (GT is better)
                        nlpd_conclusion = "🎯 SIGNIFICANT: Ground truth provides much better uncertainty quantification!"
                        nlpd_interpretation = "Trained hyperparameters poorly capture quantum circuit uncertainty structure."
                    elif nlpd_improvement > 0.05:
                        nlpd_conclusion = "✅ MODERATE: Ground truth provides better uncertainty quantification."
                        nlpd_interpretation = "Trained hyperparameters partially capture quantum circuit uncertainty structure."
                    elif nlpd_improvement > -0.05:
                        nlpd_conclusion = "➖ SIMILAR: Comparable uncertainty quantification quality."
                        nlpd_interpretation = "Trained hyperparameters adequately capture quantum circuit uncertainty structure."
                    else:
                        nlpd_conclusion = "🎯 SURPRISING: Trained hyperparameters provide better uncertainty quantification!"
                        nlpd_interpretation = "Trained hyperparameters may have learned better uncertainty structure than ground truth."
                    
                    print(f"  NLPD Assessment: {nlpd_conclusion}")
                    print(f"  Interpretation:  {nlpd_interpretation}")
                    
                    # Add NLPD to comparison results
                    if 'comparison_results' not in locals():
                        comparison_results = {}
                    comparison_results.update({
                        'nlpd_improvement': nlpd_improvement,
                        'nlpd_improvement_pct': nlpd_improvement_pct,
                        'nlpd_conclusion': nlpd_conclusion,
                        'nlpd_interpretation': nlpd_interpretation
                    })
                
                # Additional analysis: correlation between hyperparameter error and prediction quality
                print(f"\nHyperparameter Error vs Prediction Quality:")
                print(f"  Hyperparameter L2 Error: {error_best:.6f}")
                print(f"  Primary Impact Metric (R²): {r2_improvement:+.6f}")
                print(f"  Primary Impact Metric (RMSE): {rmse_improvement:+.6f}")
                
                # Store comparison results for later use
                comparison_results = {
                    'gt_metrics': gt_metrics,
                    'trained_metrics': test_metrics,
                    'r2_improvement': r2_improvement,
                    'rmse_improvement': rmse_improvement,
                    'mse_improvement': mse_improvement,
                    'mae_improvement': mae_improvement,
                    'max_error_improvement': max_error_improvement,
                    'hyperparameter_error': error_best,
                    'significant_improvements': significant_improvements,
                    'improvement_ratio': improvement_ratio,
                    'conclusion': conclusion,
                    'recommendation': recommendation
                }
                
                # Add NLPD metrics if available
                if has_nlpd_comparison:
                    comparison_results.update({
                        'nlpd_improvement': nlpd_improvement,
                        'nlpd_improvement_pct': nlpd_improvement_pct,
                        'gt_nlpd': gt_metrics['nlpd'],
                        'trained_nlpd': test_metrics['nlpd'],
                        'has_nlpd': True
                    })
                else:
                    comparison_results['has_nlpd'] = False
                
                # Add Range NRMSE metrics if available
                if has_range_nrmse_comparison:
                    comparison_results.update({
                        'range_nrmse_improvement': range_nrmse_improvement,
                        'range_nrmse_improvement_pct': range_nrmse_improvement_pct,
                        'gt_range_nrmse': gt_metrics['normalized_rmse_range'],
                        'trained_range_nrmse': test_metrics['normalized_rmse_range'],
                        'has_range_nrmse': True
                    })
                else:
                    comparison_results['has_range_nrmse'] = False
                
                # Add Uncertainty Normalized RMSE metrics if available
                if has_nrmse_comparison:
                    comparison_results.update({
                        'nrmse_improvement': -nrmse_uncertainty_improvement,  # Use the negated value for display
                        'nrmse_improvement_pct': -nrmse_uncertainty_improvement_pct,
                        'gt_nrmse': gt_metrics['normalized_rmse_uncertainty'],
                        'trained_nrmse': test_metrics['normalized_rmse_uncertainty'],
                        'has_nrmse': True
                    })
                else:
                    comparison_results['has_nrmse'] = False
                
            except Exception as e:
                print(f"❌ Error during ground truth prediction: {e}")
                print(f"Could not compare with ground truth hyperparameters.")
                comparison_results = None
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠️ Ground truth hyperparameters not available for comparison.")
            comparison_results = None
        
        # Plot predictions
        if not args.no_plot:
            print(f"\nGenerating prediction plots...")
            
            # Prepare configuration dictionary for display
            config_dict = {
                'Dataset': 'Quantum GP' if not args.classical_dataset else 'Classical',
                'Input Dimension': f"{input_dim}D",
                'Training Samples': X_train.shape[0],
                'Test Samples': X_test.shape[0],
                'Agents': n_agents,
                'Data Percentage': f"{args.data_percentage*100:.1f}%" if args.data_percentage < 1.0 else "100%",
                'ADMM Iterations': iter,
                'Gradient Method': 'riemannian',
                'Differentiation': args.differentiation,
                'Encoding': args.encoding,
                'Kernel Type': args.kernel_type,
                'Prediction Method': args.prediction_method,
                'Qubits': num_qubits,
                'Layers': num_layers,
                'Noise Std': f"{noise_std:.3f}",
                'Shift Value': f"{shift_value:.4f}",
                'Random Seed': args.seed,
                'Riemannian': 'Yes',
                'ADMM ρ (rho)': f"{rho:.1f}",
                'Lipschitz Constants': f"[{', '.join([f'{l:.1f}' for l in L])}]" if len(L) <= 5 else f"[{L[0]:.1f}, ..., {L[-1]:.1f}] ({len(L)} agents)",
                'Test R² Score': f"{test_metrics['r2']:.4f}",
                'Test RMSE': f"{test_metrics['rmse']:.6f}",
                'Test MSE': f"{test_metrics['mse']:.6f}",
                'Max Abs Error': f"{test_metrics['max_error']:.6f}",
                'Performance': test_metrics['performance']
            }
            
            # Add range-based NRMSE (always available)
            if 'normalized_rmse_range' in test_metrics:
                # Check if we have ground truth comparison for range-based NRMSE
                if 'comparison_results' in locals() and comparison_results is not None and comparison_results.get('has_range_nrmse', False):
                    config_dict['Range NRMSE'] = f"Test:: {test_metrics['normalized_rmse_range']:.4f} | GT:: {comparison_results['gt_range_nrmse']:.4f}"
                else:
                    config_dict['Range NRMSE'] = f"{test_metrics['normalized_rmse_range']:.4f}"
            
            # Add NLPD (uncertainty quality) if available
            if 'nlpd' in test_metrics:
                # Check if we have ground truth comparison for NLPD
                if 'comparison_results' in locals() and comparison_results is not None and comparison_results.get('has_nlpd', False):
                    # Show both test and GT NLPD in one line
                    config_dict['NLPD'] = f"Test:: {test_metrics['nlpd']:.6f} | GT:: {comparison_results['gt_nlpd']:.6f}"
                else:
                    # Show only test NLPD if no GT available
                    config_dict['NLPD'] = f"{test_metrics['nlpd']:.6f}"
                
                # Add Uncertainty-based Normalized RMSE if available
                if 'normalized_rmse_uncertainty' in test_metrics:
                    # Check if we have ground truth comparison for Uncertainty NRMSE
                    if 'comparison_results' in locals() and comparison_results is not None and comparison_results.get('has_nrmse', False):
                        # Show both test and GT Uncertainty Normalized RMSE in one line
                        config_dict['Uncertainty NRMSE'] = f"Test:: {test_metrics['normalized_rmse_uncertainty']:.4f} | GT:: {comparison_results['gt_nrmse']:.4f}"
                    else:
                        # Show only test Uncertainty Normalized RMSE if no GT available
                        config_dict['Uncertainty NRMSE'] = f"{test_metrics['normalized_rmse_uncertainty']:.4f}"
                
                config_dict['Uncertainty Quality'] = test_metrics.get('uncertainty_quality', 'N/A')
            
            # Add ground truth and best ADMM hyperparameters if available
            if ground_truth_params is not None:
                config_dict['Ground Truth Params'] = f"[{', '.join([f'{p:.3f}' for p in ground_truth_params])}]"
                if z_best is not None:
                    config_dict['Best ADMM Params'] = f"[{', '.join([f'{p:.3f}' for p in z_best])}]"
                    config_dict['Best Error'] = f"{error_best:.6f}"
                
                # Add R² improvement if comparison was performed
                if 'comparison_results' in locals() and comparison_results is not None:
                    config_dict['R² GT vs Trained'] = f"{comparison_results['r2_improvement']:+.4f}"
                    config_dict['GT Impact'] = f"{comparison_results['significant_improvements']}/{comparison_results.get('total_comparison_metrics', 5)}"
            
            if args.kernel_type == 'projected':
                config_dict['Measurement'] = args.measurement
                config_dict['Outer Kernel'] = outer_kernel.title()
                if outer_kernel_params:
                    # Format outer kernel parameters nicely
                    param_strs = []
                    for key, value in outer_kernel_params.items():
                        if isinstance(value, float):
                            param_strs.append(f"{key}={value:.3f}")
                        else:
                            param_strs.append(f"{key}={value}")
                    config_dict['Outer Kernel Params'] = f"{{{', '.join(param_strs)}}}"
            
            config_dict['Riem. Method'] = riemannian_method
            config_dict['Riem. LR'] = f"{riemannian_lr:.4f}"
            
            # Prepare NLPD information for enhanced plotting
            nlpd_info = {}
            if 'comparison_results' in locals() and comparison_results is not None:
                if comparison_results.get('has_nlpd', False):
                    nlpd_info = {
                        'improvement': comparison_results['nlpd_improvement'],
                        'gt_nlpd': comparison_results['gt_nlpd'],
                        'trained_nlpd': comparison_results['trained_nlpd']
                    }
            
            plot_predictions(
                X_test=X_test,
                Y_true=Y_test,
                Y_pred=y_pred_mean,
                Y_pred_var=y_pred_var,
                X_train=X_train,
                Y_train=Y_train,
                title=f"Quantum GP Predictions ({input_dim}D, {args.encoding} encoding)",
                save_plot=True,
                output_dir="results",
                config=config_dict,
                nlpd_info=nlpd_info if nlpd_info else None
            )
        
        # Final Summary
        print(f"\n{'='*60}")
        print("FINAL TRAINING AND PREDICTION SUMMARY")
        print(f"{'='*60}")
        
        print(f"Training Configuration:")
        print(f"  Dataset: {'Quantum GP' if not args.classical_dataset else 'Classical'}")
        print(f"  Input dimension: {input_dim}D")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Number of agents: {n_agents}")
        print(f"  ADMM iterations: {iter}")
        print("  Gradient method: riemannian")
        print(f"  Quantum differentiation: {args.differentiation}")
        print(f"  Encoding: {args.encoding}")
        print(f"  Kernel type: {args.kernel_type}")
        if args.kernel_type == 'projected':
            print(f"  Measurement: {args.measurement}")
        print("  Riemannian optimization: Enabled")
        
        print(f"\nHyperparameter Optimization Results:")
        if ground_truth_params is not None:
            print(f"  Ground truth params: {ground_truth_params}")
            print(f"  Best found params:   {z_best}")
            print(f"  Final params:        {z}")
            print(f"  Best error:          {error_best:.6f}")
            final_error = np.linalg.norm(z[:-1] - ground_truth_params) 
            print(f"  Final error:         {final_error:.6f}")
            print(f"  Hyperparameter recovery: {'🎯 Excellent' if error_best < 1.0 else '✅ Good' if error_best < 3.0 else '⚠️ Moderate' if error_best < 5.0 else '❌ Poor'}")
        else:
            print(f"  Final consensus params: {z}")
            print(f"  (No ground truth available for error calculation)")
        
        print(f"\nPrediction Performance:")
        print(f"  Test R² Score:        {test_metrics['r2']:.4f} ({'🎯 Excellent' if test_metrics['r2'] > 0.9 else '✅ Good' if test_metrics['r2'] > 0.7 else '⚠️ Fair' if test_metrics['r2'] > 0.5 else '❌ Poor'})")
        print(f"  Test RMSE:           {test_metrics['rmse']:.6f}")
        print(f"  Test MAE:            {test_metrics['mae']:.6f}")
        if 'uncertainty_quality' in test_metrics:
            print(f"  Uncertainty Quality:  {test_metrics['uncertainty_quality']}")
        
        if train_metrics is not None:
            print(f"  Training R² Score:    {train_metrics['r2']:.4f}")
            overfitting_check = train_metrics['r2'] - test_metrics['r2']
            print(f"  Overfitting Check:    {overfitting_check:.4f} ({'🎯 No overfitting' if overfitting_check < 0.1 else '✅ Minimal' if overfitting_check < 0.2 else '⚠️ Moderate' if overfitting_check < 0.4 else '❌ Significant overfitting'})")
        
        # Add hyperparameter impact summary if comparison was performed
        if 'comparison_results' in locals() and comparison_results is not None:
            print(f"\nHyperparameter Impact on Prediction Accuracy:")
            print(f"  Ground Truth vs Best Trained:")
            print(f"    R² Improvement:      {comparison_results['r2_improvement']:+.6f}")
            print(f"    RMSE Improvement:    {comparison_results['rmse_improvement']:+.6f}")
            print(f"    Significant Improvements: {comparison_results['significant_improvements']}/6 metrics ({comparison_results['improvement_ratio']*100:.1f}%)")
            print(f"    Impact Assessment:   {comparison_results['conclusion'].split(': ')[1] if ': ' in comparison_results['conclusion'] else comparison_results['conclusion']}")
        
        print(f"\nComputational Performance:")
        print(f"  Total training time:  {total_time:.2f}s")
        print(f"  Prediction time:      {prediction_time:.2f}s")
        print(f"  Training time per iteration: {total_time/iter:.2f}s")
        
        print(f"\n🎉 Quantum GP training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during prediction and evaluation: {e}")
        print(f"Training completed but prediction failed.")
        print(f"Final hyperparameters available: {hyperparams_to_use}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()