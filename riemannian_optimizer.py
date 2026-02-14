"""
Riemannian Optimization for Quantum Circuit Parameters

This module implements Riemannian optimization methods specifically designed for 
handling rotational hyperparameters in quantum circuits. The key insight is that
rotation angles naturally live on a torus manifold S^1 x S^1 x ... x S^1, and
standard Euclidean optimization can be inefficient or get stuck in local minima.

Key Features:
1. Torus manifold optimization for rotation parameters
2. Exponential and logarithmic maps for manifold operations
3. Retraction and vector transport operations
4. Riemannian gradient descent and conjugate gradient methods
5. Parallel transport for distributed optimization (ADMM)

References:
- Absil, P.-A., Mahony, R., & Sepulchre, R. (2009). Optimization algorithms on matrix manifolds.
- Boumal, N. (2023). An introduction to optimization on smooth manifolds.
- Quantum circuit optimization papers using Riemannian methods
"""

import numpy as np
from typing import Tuple, Optional, Callable
import time

def circular_mean(angles: np.ndarray, period: float = np.pi) -> np.ndarray:
    """
    Compute the circular mean (Karcher mean) for angles on a torus manifold.
    
    This is the geometrically correct way to average angles on S^1 x S^1 x ... x S^1.
    For each parameter dimension, we compute the mean direction using the
    arctangent of the sum of unit vectors.
    
    Args:
        angles: Array of shape (n_samples, n_params) containing angles to average
        period: Period of the circular space (default: π)
    
    Returns:
        Array of shape (n_params,) containing the circular mean for each parameter
    """
    # Convert to unit circle representation
    cos_sum = np.sum(np.cos(2 * np.pi * angles / period), axis=0)
    sin_sum = np.sum(np.sin(2 * np.pi * angles / period), axis=0)
    
    # Compute circular mean angle
    mean_angle = np.arctan2(sin_sum, cos_sum) * period / (2 * np.pi)
    
    # Wrap to [0, period)
    mean_angle = np.mod(mean_angle, period)
    
    return mean_angle

class TorusManifold:
    """
    Torus manifold S^1 x S^1 x ... x S^1 for handling rotation parameters.
    
    Each parameter is wrapped to [0, π] and the manifold structure
    accounts for the periodic nature of rotations.
    """
    
    def __init__(self, dimension: int, period: float = np.pi):
        """
        Initialize torus manifold.
        
        Args:
            dimension: Number of rotation parameters
            period: Period of the parameters (default: π for quantum circuit parameters in [0, π])
        """
        self.dim = dimension
        self.period = period
        self.name = f"Torus S^1 x ... x S^1 ({dimension}D, period={period:.3f})"
        
    def wrap_to_manifold(self, x: np.ndarray) -> np.ndarray:
        """
        Project point to manifold by wrapping angles to [0, period].
        
        Args:
            x: Parameter vector
            
        Returns:
            Wrapped parameter vector on manifold
        """
        return np.mod(x, self.period)
    
    def random_point(self) -> np.ndarray:
        """Generate random point on torus manifold."""
        return np.random.uniform(0, self.period, self.dim)
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Riemannian distance between two points on torus.
        
        For each component, compute minimum arc distance on S^1.
        This accounts for the periodic nature of rotation parameters.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Compute raw difference
        diff = x - y
        
        # For each component, find the shortest path considering periodicity
        wrapped_diff = np.mod(diff + self.period/2, self.period) - self.period/2
        
        return np.linalg.norm(wrapped_diff)
    
    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: move from point x in direction v.
        
        On torus, this is simply addition followed by wrapping.
        """
        return self.wrap_to_manifold(x + v)
    
    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: find tangent vector from x to y.
        
        Returns the shortest tangent vector pointing from x to y.
        """
        return self.wrap_to_manifold(y - x)
    
    def retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Retraction operation (alternative to exp_map, often cheaper).
        
        For torus, we use the same as exp_map.
        """
        return self.exp_map(x, v)
    
    def vector_transport(self, x: np.ndarray, v: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Vector transport: move tangent vector v from x to x+d.
        
        On torus, tangent vectors are just vectors in R^n, so transport is identity.
        """
        return v
    
    def riemannian_gradient(self, x: np.ndarray, euclidean_grad: np.ndarray) -> np.ndarray:
        """
        Convert Euclidean gradient to Riemannian gradient.
        
        On torus, the Riemannian gradient equals the Euclidean gradient
        since the torus is embedded in Euclidean space with the induced metric.
        """
        return euclidean_grad


class RiemannianOptimizer:
    """
    Riemannian optimization algorithms for quantum circuit parameters with balanced optimization strategy.
    """
    
    def __init__(self, manifold: TorusManifold, learning_rate: float = 0.015, 
                 method: str = 'gradient_descent', beta: float = 0.9, 
                 gradient_clip_norm: float = 1.0, max_step_size: float = 0.08):
        """
        Initialize Riemannian optimizer.
        
        Args:
            manifold: Torus manifold for parameter space
            learning_rate: Step size for optimization
            method: Optimization method ('gradient_descent', 'conjugate_gradient', 'momentum')
            beta: Momentum coefficient for momentum and conjugate gradient methods
            gradient_clip_norm: Gradient clipping norm
            max_step_size: Maximum step size
        """
        self.manifold = manifold
        self.lr = learning_rate
        self.method = method
        self.beta = beta
        self.gradient_clip_norm = gradient_clip_norm
        self.max_step_size = max_step_size
        
        # State for momentum and conjugate gradient
        self.velocity = None
        self.prev_grad = None
        self.iteration = 0
        
    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step.
        
        Args:
            x: Current parameter values
            grad: Euclidean gradient
            
        Returns:
            Updated parameter values
        """
        self.iteration += 1
        
        # Clip gradients to prevent large jumps
        grad_clipped = self._clip_gradient(grad)
        
        # Convert to Riemannian gradient
        riem_grad = self.manifold.riemannian_gradient(x, grad_clipped)
        
        if self.method == 'gradient_descent':
            return self._gradient_descent_step(x, riem_grad)
        elif self.method == 'momentum':
            return self._momentum_step(x, riem_grad)
        elif self.method == 'conjugate_gradient':
            return self._conjugate_gradient_step(x, riem_grad)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Clip gradients to prevent large parameter jumps."""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.gradient_clip_norm:
            return grad * (self.gradient_clip_norm / grad_norm)
        return grad
    
    def _gradient_descent_step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Riemannian gradient descent step with step size control."""
        direction = -self.lr * grad
        
        # Limit step size to prevent large jumps
        step_norm = np.linalg.norm(direction)
        if step_norm > self.max_step_size:
            direction = direction * (self.max_step_size / step_norm)
            
        return self.manifold.retraction(x, direction)
    
    def _momentum_step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Riemannian gradient descent with momentum and step size control."""
        if self.velocity is None:
            self.velocity = np.zeros_like(grad)
        
        # Update velocity with momentum
        self.velocity = self.beta * self.velocity - self.lr * grad
        
        # Limit velocity magnitude to prevent large jumps
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > self.max_step_size:
            self.velocity = self.velocity * (self.max_step_size / velocity_norm)
        
        # Take step
        x_new = self.manifold.retraction(x, self.velocity)
        
        return x_new
    
    def _conjugate_gradient_step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Riemannian conjugate gradient step with step size control."""
        if self.prev_grad is None:
            # First iteration: use gradient descent
            direction = -self.lr * grad
            self.prev_grad = grad
            
            # Limit step size
            step_norm = np.linalg.norm(direction)
            if step_norm > self.max_step_size:
                direction = direction * (self.max_step_size / step_norm)
                
            return self.manifold.retraction(x, direction)
        
        # Compute beta using Polak-Ribière formula on manifold
        grad_diff = grad - self.prev_grad
        beta_pr = np.dot(grad, grad_diff) / (np.dot(self.prev_grad, self.prev_grad) + 1e-10)
        beta_pr = max(0, beta_pr)  # Ensure non-negative
        
        # Compute conjugate direction
        if self.velocity is None:
            self.velocity = -grad
        else:
            # Vector transport previous direction
            transported_velocity = self.manifold.vector_transport(x, self.velocity, np.zeros_like(x))
            self.velocity = -grad + beta_pr * transported_velocity
        
        # Take step with size control
        direction = self.lr * self.velocity
        step_norm = np.linalg.norm(direction)
        if step_norm > self.max_step_size:
            direction = direction * (self.max_step_size / step_norm)
        
        self.prev_grad = grad
        return self.manifold.retraction(x, direction)
        x_new = self.manifold.retraction(x, direction)
        
        self.prev_grad = grad
        return x_new


class RiemannianADMM:
    """
    ADMM algorithm adapted for Riemannian manifolds with balanced optimization strategy.
    """
    
    def __init__(self, manifold: TorusManifold, rho: float = 1.0):
        """
        Initialize Riemannian ADMM.
        
        Args:
            manifold: Torus manifold for parameter space
            rho: ADMM penalty parameter
        """
        self.manifold = manifold
        self.rho = rho
        self.iteration = 0
        
    def update_z(self, theta: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """
        Update global variable z using Karcher mean (circular mean) on torus manifold.
        
        This implements the geometrically correct averaging for rotational parameters
        on the torus S^1 x S^1 x ... x S^1, which respects the circular topology.
        
        Args:
            theta: Agent parameters (n_agents x n_params)
            psi: Dual variables (n_agents x n_params)
            
        Returns:
            Updated z on manifold using Karcher mean
        """
        # Compute xi = theta + psi/rho for each agent
        xi = theta + psi / self.rho
        
        # Use circular mean (Karcher mean) on torus manifold
        z_new = circular_mean(xi, period=self.manifold.period)
        
        return z_new
    
    def update_theta(self, z: np.ndarray, grad: np.ndarray, psi: np.ndarray, 
                     L: float, optimizer: RiemannianOptimizer) -> np.ndarray:
        """
        Update agent parameter theta using Riemannian optimization.
        
        Args:
            z: Global parameter
            grad: Gradient of local loss
            psi: Dual variable
            L: Lipschitz constant
            optimizer: Riemannian optimizer
            
        Returns:
            Updated theta on manifold
        """
        # ADMM update rule: minimize local_loss + psi^T(theta - z) + (rho/2)||theta - z||^2
        # This gives: theta = z - (grad + psi) / (rho + L)
        
        # Compute ADMM direction
        admm_direction = -(grad + psi) / (self.rho + L)
        
        # Start from z and move in ADMM direction
        theta_new = self.manifold.retraction(z, admm_direction)
        
        return theta_new
    
    def update_psi(self, psi: np.ndarray, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Update dual variable psi.
        
        Args:
            psi: Current dual variable
            theta: Agent parameter
            z: Global parameter
            
        Returns:
            Updated psi
        """
        # Use logarithmic map to get tangent vector from z to theta
        theta_minus_z = self.manifold.log_map(z, theta)
        
        # Standard ADMM dual update
        psi_new = psi + self.rho * theta_minus_z
        
        return psi_new
    
    def compute_primal_residual(self, theta: np.ndarray, z: np.ndarray) -> float:
        """
        Compute primal residual using Riemannian distance.
        
        Args:
            theta: Agent parameters (n_agents x n_params)
            z: Global parameter
            
        Returns:
            Primal residual norm
        """
        residuals = []
        for i in range(theta.shape[0]):
            dist = self.manifold.distance(theta[i], z)
            residuals.append(dist)
        
        return np.linalg.norm(residuals)
    
    def compute_dual_residual(self, z_new: np.ndarray, z_old: np.ndarray) -> float:
        """
        Compute dual residual using Riemannian distance.
        
        Args:
            z_new: New global parameter
            z_old: Old global parameter
            
        Returns:
            Dual residual norm
        """
        return self.manifold.distance(z_new, z_old)


def create_riemannian_framework(num_parameters: int, learning_rate: float = 0.01, 
                               rho: float = 1.0, method: str = 'gradient_descent',
                               gradient_clip_norm: float = 1.0, 
                               max_step_size: float = 0.1) -> Tuple[TorusManifold, RiemannianOptimizer, RiemannianADMM]:
    """
    Create complete Riemannian optimization framework.
    
    Args:
        num_parameters: Number of rotation parameters
        learning_rate: Learning rate for optimization
        rho: ADMM penalty parameter
        method: Optimization method
        gradient_clip_norm: Gradient clipping norm
        max_step_size: Maximum step size
        
    Returns:
        Tuple of (manifold, optimizer, admm)
    """
    manifold = TorusManifold(num_parameters)
    
    optimizer = RiemannianOptimizer(manifold, learning_rate, method, 
                                  gradient_clip_norm=gradient_clip_norm,
                                  max_step_size=max_step_size)
    
    admm = RiemannianADMM(manifold, rho)
    
    return manifold, optimizer, admm
