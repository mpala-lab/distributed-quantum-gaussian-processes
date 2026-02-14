"""
Riemannian Agent for Quantum Gaussian Process Regression

This module implements an Agent that uses Riemannian optimization for handling
rotational hyperparameters in quantum circuits. The key advantages are:

1. Respects the natural geometry of rotation parameters
2. Better convergence properties on periodic parameter spaces  
3. Avoids issues with parameter wrapping and discontinuities
4. More stable optimization for quantum circuit training
"""

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
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import os

# Import our Riemannian optimization framework
from riemannian_optimizer import TorusManifold, RiemannianOptimizer, RiemannianADMM, create_riemannian_framework

def _evaluate_kernel_shifted_riemannian(args):
    """
    Helper function for parallel parameter shift evaluation with Riemannian parameters.
    Evaluates kernel with shifted parameters, ensuring they stay on the manifold.
    """
    kernel_data, X, params_shifted, manifold = args
    
    # Ensure parameters are on manifold
    params_on_manifold = manifold.wrap_to_manifold(params_shifted)
    
    # Track QPU job timing
    job_start = time.time()
    
    # Recreate kernel with same configuration based on encoding type
    encoding_type = kernel_data.get('encoding_type', 'yz_cx')
    kernel_type = kernel_data.get('kernel_type', 'fidelity')
    
    # Create encoding circuit based on type
    if encoding_type == 'chebyshev':
        enc_circ = ChebyshevPQC(kernel_data['num_qubits'], 
                               num_features=kernel_data['num_features'], 
                               num_layers=kernel_data['num_layers'])
    elif encoding_type == 'yz_cx':
        enc_circ = YZ_CX_EncodingCircuit(kernel_data['num_qubits'], 
                                        num_features=kernel_data['num_features'], 
                                        num_layers=kernel_data['num_layers'])
    elif encoding_type == 'hubregtsen':
        enc_circ = HubregtsenEncodingCircuit(kernel_data['num_qubits'], 
                                            num_features=kernel_data['num_features'], 
                                            num_layers=kernel_data['num_layers'])
    elif encoding_type == 'kyriienko':
        enc_circ = KyriienkoEncodingCircuit(kernel_data['num_qubits'], 
                                           num_features=kernel_data['num_features'], 
                                           num_layers=kernel_data['num_layers'])
    elif encoding_type == 'multi_control':
        enc_circ = MultiControlEncodingCircuit(kernel_data['num_qubits'], 
                                              num_features=kernel_data['num_features'], 
                                              num_layers=kernel_data['num_layers'])
    elif encoding_type == 'layered':
        enc_circ = LayeredEncodingCircuit(kernel_data['num_qubits'], 
                                         num_features=kernel_data['num_features'], 
                                         num_layers=kernel_data['num_layers'],
                                         gates=['RX', 'RY', 'RZ'])
    elif encoding_type == 'random':
        enc_circ = RandomEncodingCircuit(kernel_data['num_qubits'], 
                                        num_features=kernel_data['num_features'], 
                                        num_layers=kernel_data['num_layers'])
    elif encoding_type == 'highdim':
        enc_circ = HighDimEncodingCircuit(kernel_data['num_qubits'], 
                                         num_features=kernel_data['num_features'], 
                                         num_layers=kernel_data['num_layers'])
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    # Create quantum kernel based on type
    if kernel_type == 'fidelity':
        q_kernel = FidelityKernel(
            encoding_circuit=enc_circ,
            executor=Executor(kernel_data['executor_type']),
            parameter_seed=0,
            use_expectation=True,
            evaluate_duplicates="all"
        )
    elif kernel_type == 'projected':
        # Get outer kernel parameters with defaults
        outer_kernel = kernel_data.get('outer_kernel', 'gaussian')
        outer_kernel_params = kernel_data.get('outer_kernel_params', {})
        
        q_kernel = ProjectedQuantumKernel(
            encoding_circuit=enc_circ,
            measurement=kernel_data['measurement'],
            outer_kernel=outer_kernel,
            executor=Executor(kernel_data['executor_type']),
            parameter_seed=0,
            regularization=kernel_data.get('regularization', None),
            **outer_kernel_params
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Set parameters on manifold
    q_kernel._parameters = params_on_manifold
    
    # Evaluate kernel matrix (this is the actual QPU computation)
    qpu_eval_start = time.time()
    K = q_kernel.evaluate(X, X)
    qpu_eval_time = time.time() - qpu_eval_start
    
    job_total_time = time.time() - job_start
    
    return K


class RiemannianAgent:
    """
    Agent class for distributed quantum Gaussian process regression using Riemannian optimization.
    
    This agent treats quantum circuit parameters as living on a torus manifold and uses
    Riemannian optimization methods for more stable and efficient training.
    """
    
    def __init__(self, agent_id, X_sub, Y_sub, num_qubits, noise_std, rho, L, 
                 q_kernel=None, use_parameter_shift=False, num_workers=None, 
                 shift_value=np.pi/8, num_layers=2, combined_computation=True, 
                 encoding_type='yz_cx', kernel_type='fidelity', measurement='XYZ',
                 outer_kernel='gaussian', outer_kernel_params=None, regularization=None,
                 riemannian_lr=0.01, riemannian_method='gradient_descent', 
                 riemannian_beta=0.9):
        """
        Initialize Riemannian Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            X_sub: Training input data for this agent
            Y_sub: Training output data for this agent
            num_qubits: Number of qubits in quantum circuit
            noise_std: Noise standard deviation
            rho: ADMM penalty parameter
            L: Lipschitz constant
            q_kernel: Pre-initialized quantum kernel (optional)
            use_parameter_shift: Whether to use parameter shift rule
            num_workers: Number of parallel workers
            shift_value: Parameter shift value
            num_layers: Number of layers in encoding circuit
            combined_computation: Use combined kernel+derivative computation
            encoding_type: Type of encoding circuit
            kernel_type: Type of quantum kernel
            measurement: Measurement for ProjectedQuantumKernel
            outer_kernel: Outer kernel type for ProjectedQuantumKernel
            outer_kernel_params: Parameters for the outer kernel
            regularization: Regularization technique for ProjectedQuantumKernel
            riemannian_lr: Learning rate for Riemannian optimizer
            riemannian_method: Riemannian optimization method
            riemannian_beta: Beta parameter for momentum/conjugate gradient
        """
        self.agent_id = agent_id
        self.X_sub = X_sub
        self.Y_sub = Y_sub
        self.num_qubits = num_qubits
        self.noise_std = noise_std
        self.rho = rho
        self.L = L
        self.q_kernel = q_kernel
        self.use_parameter_shift = use_parameter_shift
        self.num_workers = num_workers
        self.shift_value = shift_value
        self.num_layers = num_layers
        self.combined_computation = combined_computation
        self.encoding_type = encoding_type
        self.kernel_type = kernel_type
        self.measurement = measurement
        self.outer_kernel = outer_kernel
        self.outer_kernel_params = outer_kernel_params
        self.regularization = regularization
        
        # Riemannian optimization parameters
        self.riemannian_lr = riemannian_lr
        self.riemannian_method = riemannian_method
        self.riemannian_beta = riemannian_beta
        
        # Initialize Riemannian framework (will be set up when we know num_parameters)
        self.manifold = None
        self.riemannian_optimizer = None
        self.riemannian_admm = None
        
    def _setup_riemannian_framework(self, num_parameters):
        """Set up Riemannian optimization framework once we know the number of parameters."""
        if self.manifold is None:
            self.manifold, self.riemannian_optimizer, self.riemannian_admm = create_riemannian_framework(
                num_parameters=num_parameters,
                learning_rate=self.riemannian_lr,
                rho=self.rho,
                method=self.riemannian_method
            )
            print(f"Agent {self.agent_id} - Initialized Riemannian framework: {self.manifold.name}")
    
    def _parallel_parameter_shift_kernel_and_derivatives_riemannian(self, X, params, h=None):
        """
        Compute both kernel matrix and derivatives using parallel parameter shift with Riemannian manifold.
        """
        if h is None:
            h = self.shift_value
            
        num_params = len(params)
        
        # Ensure params are on manifold
        params = self.manifold.wrap_to_manifold(params)
        
        # Determine executor type
        executor_type = "statevector_simulator"
        if hasattr(self.q_kernel, 'executor'):
            executor_str = str(self.q_kernel.executor).lower()
            if "qiskit" in executor_str or "statevector" in executor_str:
                executor_type = "statevector_simulator"
            else:
                executor_type = "pennylane"
        
        # Prepare kernel configuration data
        kernel_data = {
            'num_qubits': self.num_qubits,
            'num_features': X.shape[1],
            'num_layers': self.num_layers,
            'executor_type': executor_type,
            'encoding_type': self.encoding_type,
            'kernel_type': self.kernel_type,
            'measurement': self.measurement
        }
        
        # Prepare parameter shifts for parallel evaluation
        shift_args = []
        
        # Add original parameters for kernel matrix evaluation
        shift_args.append((kernel_data, X, params.copy(), self.manifold))
        
        for i in range(num_params):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += h
            shift_args.append((kernel_data, X, params_plus, self.manifold))
            
            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= h
            shift_args.append((kernel_data, X, params_minus, self.manifold))
        
        # Execute all evaluations in parallel
        print(f"Agent {self.agent_id} - 🔬 Riemannian: Submitting {len(shift_args)} quantum jobs...")
        qpu_shift_start = time.time()
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            kernel_results = list(executor.map(_evaluate_kernel_shifted_riemannian, shift_args))
        qpu_shift_time = time.time() - qpu_shift_start
        print(f"Agent {self.agent_id} - ⚡ Riemannian quantum execution: {qpu_shift_time:.4f}s")
        
        # Extract kernel matrix (first result)
        K = kernel_results[0]
        
        # Compute derivatives from remaining results
        dK_dparams = np.zeros((num_params, X.shape[0], X.shape[0]))
        
        for i in range(num_params):
            K_plus = kernel_results[1 + 2*i]
            K_minus = kernel_results[1 + 2*i + 1]
            dK_dparams[i] = (K_plus - K_minus) / (2.0 * h)
        
        return K, dK_dparams
    
    def _manual_projected_kernel_derivatives_riemannian(self, X, params, h=None):
        """Manual parameter shift for ProjectedQuantumKernel with Riemannian manifold."""
        if h is None:
            h = self.shift_value
            
        num_params = len(params)
        n_samples = X.shape[0]
        dK_dparams = np.zeros((num_params, n_samples, n_samples))
        
        # Ensure params are on manifold
        params = self.manifold.wrap_to_manifold(params)
        
        for i in range(num_params):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += h
            params_plus = self.manifold.wrap_to_manifold(params_plus)
            self.q_kernel._parameters = params_plus
            K_plus = self.q_kernel.evaluate(X, X)
            
            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= h
            params_minus = self.manifold.wrap_to_manifold(params_minus)
            self.q_kernel._parameters = params_minus
            K_minus = self.q_kernel.evaluate(X, X)
            
            # Compute derivative
            dK_dparams[i] = (K_plus - K_minus) / (2.0 * h)
        
        # Restore original parameters
        self.q_kernel._parameters = self.manifold.wrap_to_manifold(params)
        
        return dK_dparams
    
    def train_and_update(self, z, psi_i):
        """
        Train and update using Riemannian optimization.
        
        Args:
            z: Global parameter (will be projected to manifold)
            psi_i: Dual variable for this agent
            
        Returns:
            tuple: (theta_i, psi_i, nll_loss, condition_number)
        """
        start_time = time.time()
        print(f"Agent {self.agent_id} - Starting Riemannian train_and_update")
        
        # Initialize quantum kernel if needed
        if self.q_kernel is None:
            # Create quantum kernel (same as original agent)
            if self.encoding_type == 'chebyshev':
                enc_circ = ChebyshevPQC(self.num_qubits, num_features=self.X_sub.shape[1], num_layers=self.num_layers)
            elif self.encoding_type == 'yz_cx':
                enc_circ = YZ_CX_EncodingCircuit(self.num_qubits, num_features=self.X_sub.shape[1], num_layers=self.num_layers)
            elif self.encoding_type == 'hubregtsen':
                enc_circ = HubregtsenEncodingCircuit(self.num_qubits, num_features=self.X_sub.shape[1], num_layers=self.num_layers)
            elif self.encoding_type == 'kyriienko':
                enc_circ = KyriienkoEncodingCircuit(self.num_qubits, num_features=self.X_sub.shape[1], num_layers=self.num_layers)
            elif self.encoding_type == 'multi_control':
                enc_circ = MultiControlEncodingCircuit(self.num_qubits, num_features=self.X_sub.shape[1], num_layers=self.num_layers)
            elif self.encoding_type == 'layered':
                enc_circ = LayeredEncodingCircuit(self.num_qubits, num_features=self.X_sub.shape[1], num_layers=self.num_layers, gates=['RX', 'RY', 'RZ'])
            elif self.encoding_type == 'random':
                enc_circ = RandomEncodingCircuit(self.num_qubits, num_features=self.X_sub.shape[1], num_layers=self.num_layers)
            elif self.encoding_type == 'highdim':
                enc_circ = HighDimEncodingCircuit(self.num_qubits, num_features=self.X_sub.shape[1], num_layers=self.num_layers)
            else:
                raise ValueError(f"Unknown encoding type: {self.encoding_type}")
            
            if self.kernel_type == 'fidelity':
                q_kernel = FidelityKernel(
                    encoding_circuit=enc_circ, executor=Executor("pennylane"), parameter_seed=0, 
                    use_expectation=True, evaluate_duplicates="all"
                )
            elif self.kernel_type == 'projected':
                # Use outer kernel parameters with defaults
                outer_kernel_params = self.outer_kernel_params or {}
                
                q_kernel = ProjectedQuantumKernel(
                    encoding_circuit=enc_circ, 
                    measurement=self.measurement,
                    outer_kernel=self.outer_kernel,
                    executor=Executor("pennylane"), 
                    parameter_seed=0,
                    regularization=self.regularization,
                    **outer_kernel_params
                )
            else:
                raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        else:
            q_kernel = self.q_kernel
        
        # Set up Riemannian framework
        num_parameters = len(z)
        self._setup_riemannian_framework(num_parameters)
        
        # Project z to manifold
        z_manifold = self.manifold.wrap_to_manifold(z)
        q_kernel._parameters = z_manifold
        
        # Compute kernel matrix and derivatives
        block_start = time.time()
        
        if self.use_parameter_shift:
            print(f"Agent {self.agent_id} - Using Riemannian parameter shift")
            
            if self.combined_computation:
                C, dCdTheta = self._parallel_parameter_shift_kernel_and_derivatives_riemannian(self.X_sub, z_manifold)
            else:
                # Separate computation
                q_kernel._parameters = z_manifold
                C = q_kernel.evaluate(self.X_sub, self.X_sub)
                dCdTheta = self._parallel_parameter_shift_derivatives_riemannian(self.X_sub, z_manifold)
        else:
            print(f"Agent {self.agent_id} - Using standard derivatives with Riemannian manifold")
            
            if self.kernel_type == 'projected':
                q_kernel._parameters = z_manifold
                C = q_kernel.evaluate(self.X_sub, self.X_sub)
                dCdTheta = self._manual_projected_kernel_derivatives_riemannian(self.X_sub, z_manifold)
            else:
                results = q_kernel.evaluate_derivatives(self.X_sub, self.X_sub, values=["K", "dKdp"])
                C = results["K"]
                dCdTheta = results["dKdp"]
        
        block_time = time.time() - block_start
        print(f"Agent {self.agent_id} - Riemannian quantum computation: {block_time:.4f}s")
        
        # Add noise and solve system (same as original)
        C_noise = C + self.noise_std**2 * np.eye(C.shape[0])
        condition_number = np.linalg.cond(C)
        
        # Matrix inversion
        try:
            L = np.linalg.cholesky(C_noise)
            C_inv_y = np.linalg.solve(L.T, np.linalg.solve(L, self.Y_sub))
            identity = np.eye(C_noise.shape[0])
            C_inv = np.linalg.solve(L.T, np.linalg.solve(L, identity))
        except np.linalg.LinAlgError:
            try:
                from scipy.linalg import lu_factor, lu_solve
                LU, piv = lu_factor(C_noise)
                C_inv_y = lu_solve((LU, piv), self.Y_sub)
                identity = np.eye(C_noise.shape[0])
                C_inv = lu_solve((LU, piv), identity)
            except np.linalg.LinAlgError:
                C_inv = np.linalg.pinv(C_noise)
                C_inv_y = C_inv @ self.Y_sub
        
        # Compute gradient (same as original)
        outer_product = np.outer(C_inv_y, C_inv_y)
        bracket_matrix = C_inv - outer_product
        
        dLdTheta = 0.5 * np.array([
            np.sum(bracket_matrix * dCdTheta[i].T) for i in range(dCdTheta.shape[0])
        ])
        
        L_theta_i = np.round(dLdTheta, 4)
        
        # Compute NLL loss
        try:
            sign, log_det = np.linalg.slogdet(C_noise)
            if sign <= 0:
                log_det = np.log(np.linalg.det(C_noise + 1e-8 * np.eye(C_noise.shape[0])))
            
            # Compute individual NLL components for correlation analysis
            log_det_term = 0.5 * log_det
            quadratic_term = 0.5 * (self.Y_sub.T @ C_inv_y)
            constant_term = 0.5 * len(self.Y_sub) * np.log(2*np.pi)
            
            # Compute the total NLL loss
            nll_loss = log_det_term + quadratic_term + constant_term
            
            # Store individual components for analysis
            nll_components = {
                'log_det_term': float(log_det_term),
                'quadratic_term': float(quadratic_term), 
                'constant_term': float(constant_term),
                'total': float(nll_loss)
            }
            
        except Exception as e:
            print(f"Agent {self.agent_id} - Warning: Could not compute NLL loss: {e}")
            nll_loss = float('inf')
            # Create placeholder components for fallback
            nll_components = {
                'log_det_term': float('inf'),
                'quadratic_term': float('inf'),
                'constant_term': float('inf'),
                'total': float('inf')
            }
        
        print(f"Agent {self.agent_id} - Riemannian L_theta_i: {L_theta_i}")
        
        # RIEMANNIAN ADMM UPDATE
        print(f"Agent {self.agent_id} - Using Riemannian ADMM updates")
        
        # Update theta using Riemannian ADMM
        theta_i = self.riemannian_admm.update_theta(z_manifold, L_theta_i, psi_i, self.L, self.riemannian_optimizer)
        
        # Update psi using Riemannian ADMM  
        psi_i = self.riemannian_admm.update_psi(psi_i, theta_i, z_manifold)
        
        # Round for display
        theta_i = np.round(theta_i, 4)
        psi_i = np.round(psi_i, 4)
        
        total_time = time.time() - start_time
        print(f"Agent {self.agent_id} - Riemannian total time: {total_time:.4f}s")
        
        return theta_i, psi_i, nll_loss, condition_number, nll_components
