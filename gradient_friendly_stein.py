#!/usr/bin/env python3
"""
Gradient-Friendly Stein Equation Solvers for RARL2
====================================================

The challenge: Standard Stein solvers (Schur, Bartels-Stewart) use eigendecompositions
and complex matrix factorizations that are difficult to differentiate through.

Solutions explored here:
1. Fixed-point iteration (differentiable but may not converge)
2. Implicit differentiation (solve forward, compute gradient separately)
3. Unrolled optimization (limited iterations, AD-friendly)
"""

import torch
import numpy as np
from typing import Tuple, Optional
import torch.autograd as autograd


class SteinSolverFixedPoint(torch.autograd.Function):
    """
    Fixed-point iteration for Stein equation with custom backward pass.
    
    Solves: A^H * X * A - X = -Q
    
    This is differentiable but convergence depends on spectral radius of A.
    """
    
    @staticmethod
    def forward(ctx, A, Q, max_iter=100, tol=1e-10):
        """
        Forward pass: Solve Stein equation via fixed-point iteration.
        """
        n = A.shape[0]
        X = torch.zeros_like(Q)
        
        # Fixed-point iteration: X_{k+1} = A^H * X_k * A + Q
        for _ in range(max_iter):
            X_new = A.conj().T @ X @ A + Q
            if torch.norm(X_new - X) < tol:
                break
            X = X_new
        
        # Save for backward
        ctx.save_for_backward(A, X)
        
        return X
    
    @staticmethod
    def backward(ctx, grad_X):
        """
        Backward pass: Compute gradients via implicit differentiation.
        
        Key insight: At solution, F(A, X) = A^H*X*A - X + Q = 0
        Using implicit function theorem:
        dX/dA = -(∂F/∂X)^{-1} * (∂F/∂A)
        """
        A, X = ctx.saved_tensors
        
        # For efficiency, use fixed-point iteration for the adjoint equation
        # Solve: A * λ * A^H - λ = -grad_X
        lambda_adjoint = torch.zeros_like(grad_X)
        for _ in range(100):
            lambda_new = A @ lambda_adjoint @ A.conj().T + grad_X
            if torch.norm(lambda_new - lambda_adjoint) < 1e-10:
                break
            lambda_adjoint = lambda_new
        
        # Gradient w.r.t A
        grad_A = 2 * torch.real(A.conj() @ X @ lambda_adjoint.T)
        
        # Gradient w.r.t Q
        grad_Q = lambda_adjoint
        
        return grad_A, grad_Q, None, None


class SteinSolverImplicit(torch.autograd.Function):
    """
    Stein solver using implicit differentiation.
    
    Forward: Use any efficient method (Schur, Bartels-Stewart)
    Backward: Compute gradient via implicit differentiation
    
    This is the most practical approach!
    """
    
    @staticmethod
    def forward(ctx, A, Q):
        """
        Forward: Solve using scipy (not differentiable but efficient).
        """
        from scipy import linalg
        
        # Convert to numpy for scipy
        A_np = A.detach().cpu().numpy()
        Q_np = Q.detach().cpu().numpy()
        
        # Solve using scipy's efficient method
        X_np = linalg.solve_discrete_lyapunov(A_np.conj().T, Q_np)
        
        # Convert back to torch
        X = torch.tensor(X_np, dtype=A.dtype, device=A.device)
        
        # Save for backward
        ctx.save_for_backward(A, X)
        
        return X
    
    @staticmethod
    def backward(ctx, grad_X):
        """
        Backward: Solve adjoint Stein equation.
        
        The adjoint equation is: A * λ * A^H - λ = -grad_X
        """
        A, X = ctx.saved_tensors
        
        from scipy import linalg
        
        # Solve adjoint equation using scipy
        grad_X_np = grad_X.detach().cpu().numpy()
        A_np = A.detach().cpu().numpy()
        
        lambda_np = linalg.solve_discrete_lyapunov(A_np, grad_X_np)
        lambda_adjoint = torch.tensor(lambda_np, dtype=A.dtype, device=A.device)
        
        # Compute gradients
        # ∂L/∂A = 2 * Re(A^* @ X @ λ^T)
        grad_A = 2 * torch.real(A.conj() @ X @ lambda_adjoint.T)
        
        # ∂L/∂Q = λ
        grad_Q = lambda_adjoint
        
        return grad_A, grad_Q


class UnrolledSteinSolver(torch.nn.Module):
    """
    Unrolled fixed-point iteration for Stein equation.
    
    Limited iterations but fully differentiable via PyTorch autograd.
    Good for when you need exact gradients through the solver.
    """
    
    def __init__(self, max_iter: int = 20):
        super().__init__()
        self.max_iter = max_iter
    
    def forward(self, A: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Solve A^H * X * A - X = -Q via unrolled iteration.
        
        CRITICAL: Limited iterations means approximate solution,
        but exact gradients through the approximation!
        """
        X = torch.zeros_like(Q)
        
        # Unroll the iteration (PyTorch tracks all operations)
        for _ in range(self.max_iter):
            X = A.conj().T @ X @ A + Q
        
        return X


class ChartSteinSolver(torch.nn.Module):
    """
    Specialized Stein solver for RARL2 chart parametrization.
    
    Solves: Λ - A·Λ·W = C·Y (non-standard form!)
    
    This is the critical equation for the chart transformation.
    """
    
    def __init__(self, method='implicit'):
        super().__init__()
        self.method = method
    
    def forward(self, A: torch.Tensor, W: torch.Tensor, 
                C: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Solve the chart Stein equation: Λ - A·Λ·W = C·Y
        
        Rearranging: Λ = A·Λ·W + C·Y
        This is a fixed-point form!
        """
        n = A.shape[0]
        Lambda = torch.zeros((n, n), dtype=torch.complex128)
        RHS = C @ Y
        
        if self.method == 'fixed_point':
            # Simple iteration (may not converge if ||A·W|| >= 1)
            for _ in range(50):
                Lambda = A @ Lambda @ W + RHS
        
        elif self.method == 'implicit':
            # Use implicit differentiation approach
            # First solve, then handle gradients separately
            Lambda = ChartSteinImplicit.apply(A, W, RHS)
        
        else:  # unrolled
            # Limited iterations but exact gradients
            for _ in range(20):
                Lambda = A @ Lambda @ W + RHS
        
        return Lambda


class ChartSteinImplicit(torch.autograd.Function):
    """
    Implicit differentiation for chart Stein equation.
    """
    
    @staticmethod
    def forward(ctx, A, W, RHS):
        """Solve Λ - A·Λ·W = RHS."""
        n = A.shape[0]
        Lambda = torch.zeros((n, n), dtype=torch.complex128)
        
        # Fixed-point iteration (forward pass only)
        with torch.no_grad():
            for _ in range(100):
                Lambda_new = A @ Lambda @ W + RHS
                if torch.norm(Lambda_new - Lambda) < 1e-12:
                    break
                Lambda = Lambda_new
        
        Lambda.requires_grad_(True)
        ctx.save_for_backward(A, W, Lambda)
        
        return Lambda
    
    @staticmethod  
    def backward(ctx, grad_Lambda):
        """
        Compute gradients via implicit differentiation.
        
        At solution: F(A,W,Λ) = Λ - A·Λ·W - RHS = 0
        """
        A, W, Lambda = ctx.saved_tensors
        
        # Solve adjoint equation: μ - W^T·μ·A^T = grad_Λ
        mu = torch.zeros_like(grad_Lambda)
        with torch.no_grad():
            for _ in range(100):
                mu_new = W.T @ mu @ A.T + grad_Lambda
                if torch.norm(mu_new - mu) < 1e-12:
                    break
                mu = mu_new
        
        # Gradients
        grad_A = -mu @ Lambda.T @ W.T
        grad_W = -A.T @ mu @ Lambda.T
        grad_RHS = mu
        
        return grad_A, grad_W, grad_RHS


def test_gradient_friendly_stein():
    """Test different Stein solver approaches."""
    
    print("Testing Gradient-Friendly Stein Solvers")
    print("="*60)
    
    # Create test problem
    n = 3
    torch.manual_seed(42)
    
    # Stable A (spectral radius < 1)
    A = torch.randn(n, n, dtype=torch.complex128) * 0.5
    Q = torch.eye(n, dtype=torch.complex128)
    
    print("1. Fixed-Point with Custom Backward:")
    X1 = SteinSolverFixedPoint.apply(A, Q)
    print(f"   Solution norm: {torch.norm(X1):.6f}")
    
    # Check gradient
    A.requires_grad_(True)
    X1 = SteinSolverFixedPoint.apply(A, Q)
    loss = torch.sum(torch.abs(X1)**2)
    loss.backward()
    print(f"   Gradient norm: {torch.norm(A.grad):.6f}")
    
    print("\n2. Implicit Differentiation:")
    A.grad = None
    X2 = SteinSolverImplicit.apply(A, Q)
    print(f"   Solution norm: {torch.norm(X2):.6f}")
    loss = torch.sum(torch.abs(X2)**2)
    loss.backward()
    print(f"   Gradient norm: {torch.norm(A.grad):.6f}")
    
    print("\n3. Unrolled Iteration:")
    solver = UnrolledSteinSolver(max_iter=20)
    A.grad = None
    X3 = solver(A, Q)
    print(f"   Solution norm: {torch.norm(X3):.6f}")
    loss = torch.sum(torch.abs(X3)**2)
    loss.backward()
    print(f"   Gradient norm: {torch.norm(A.grad):.6f}")
    
    print("\n" + "="*60)
    print("Recommendation:")
    print("- Use implicit differentiation for accuracy + efficiency")
    print("- Use unrolled for debugging (exact gradients)")
    print("- Fixed-point only when spectral radius < 1")


if __name__ == "__main__":
    test_gradient_friendly_stein()
    
    print("\n" + "="*60)
    print("Key Insights for RARL2:")
    print("1. The chart Stein equation Λ - A·Λ·W = C·Y is non-standard")
    print("2. Implicit differentiation is most practical:")
    print("   - Forward: Use efficient solver (Schur/scipy)")
    print("   - Backward: Solve adjoint equation")
    print("3. This preserves efficiency while enabling gradients!")
    print("4. Critical: Test convergence for your specific (A,W) pairs")