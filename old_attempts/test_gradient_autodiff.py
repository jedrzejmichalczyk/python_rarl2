#!/usr/bin/env python3
"""
Test RARL2 Gradient with PyTorch Automatic Differentiation
============================================================
Use PyTorch to compute exact gradients and compare with our implementation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


def numpy_to_torch_complex(A):
    """Convert numpy complex array to torch complex tensor."""
    return torch.complex(
        torch.tensor(A.real, dtype=torch.float64),
        torch.tensor(A.imag, dtype=torch.float64)
    )


def torch_to_numpy_complex(T):
    """Convert torch complex tensor to numpy complex array."""
    return T.detach().numpy()


class RARL2Objective(torch.nn.Module):
    """
    RARL2 objective function in PyTorch for automatic differentiation.
    """
    
    def __init__(self, A_F, B_F, C_F, D_F):
        super().__init__()
        # Store target system as torch tensors
        self.A_F = numpy_to_torch_complex(A_F)
        self.B_F = numpy_to_torch_complex(B_F)
        self.C_F = numpy_to_torch_complex(C_F)
        self.D_F = numpy_to_torch_complex(D_F)
        
        self.n_F = A_F.shape[0]
        self.m = B_F.shape[1]
        self.p = C_F.shape[0]
    
    def forward(self, C_real, C_imag, A_real, A_imag):
        """
        Compute the RARL2 objective ||F - H||²₂.
        
        Args:
            C_real, C_imag: Real and imaginary parts of C matrix
            A_real, A_imag: Real and imaginary parts of A matrix
            
        Returns:
            Objective value (scalar)
        """
        C = torch.complex(C_real, C_imag)
        A = torch.complex(A_real, A_imag)
        
        n = A.shape[0]
        
        # Compute lossless embedding (simplified for testing)
        # In reality, we'd use the full lossless embedding formula
        # For now, just create a simple B matrix
        B = torch.eye(n, self.m, dtype=torch.complex128)
        D = torch.zeros(self.p, self.m, dtype=torch.complex128)
        
        # Build error system: E = F - H
        # We'll compute a simplified H2 norm
        
        # For simplicity, compute Frobenius norm of frequency samples
        # This is a proxy for the H2 norm
        omega_samples = torch.linspace(0, 2*np.pi, 100)
        error_sum = 0.0
        
        for omega in omega_samples:
            z = torch.exp(1j * omega)
            
            # Evaluate F(z)
            zI_F = z * torch.eye(self.n_F, dtype=torch.complex128)
            try:
                F_z = self.D_F + self.C_F @ torch.linalg.inv(zI_F - self.A_F) @ self.B_F
            except:
                continue
            
            # Evaluate H(z)
            zI = z * torch.eye(n, dtype=torch.complex128)
            try:
                H_z = D + C @ torch.linalg.inv(zI - A) @ B
            except:
                continue
            
            # Accumulate error
            error = F_z - H_z
            error_sum += torch.sum(torch.abs(error)**2).real
        
        return error_sum / len(omega_samples)


def test_gradient_with_autodiff():
    """Test gradient computation using PyTorch automatic differentiation."""
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Dimensions
    n = 2      # Approximation order
    n_F = 3    # Target order
    p = 2      # Output dimension
    m = 2      # Input dimension
    
    # Create output normal pair
    from rarl2_integrated import create_output_normal_pair
    C_np, A_np = create_output_normal_pair(n, p)
    
    # Create target system
    A_F = np.random.randn(n_F, n_F) + 1j * np.random.randn(n_F, n_F)
    A_F = A_F * 0.8 / np.max(np.abs(np.linalg.eigvals(A_F)))
    B_F = np.random.randn(n_F, m) + 1j * np.random.randn(n_F, m)
    C_F = np.random.randn(p, n_F) + 1j * np.random.randn(p, n_F)
    D_F = np.zeros((p, m), dtype=np.complex128)
    
    # Create PyTorch objective
    objective = RARL2Objective(A_F, B_F, C_F, D_F)
    
    # Convert (C,A) to torch tensors with gradients
    C_real = torch.tensor(C_np.real, requires_grad=True, dtype=torch.float64)
    C_imag = torch.tensor(C_np.imag, requires_grad=True, dtype=torch.float64)
    A_real = torch.tensor(A_np.real, requires_grad=True, dtype=torch.float64)
    A_imag = torch.tensor(A_np.imag, requires_grad=True, dtype=torch.float64)
    
    # Compute objective and gradients
    loss = objective(C_real, C_imag, A_real, A_imag)
    loss.backward()
    
    # Extract gradients
    grad_C_torch = C_real.grad.numpy() + 1j * C_imag.grad.numpy()
    grad_A_torch = A_real.grad.numpy() + 1j * A_imag.grad.numpy()
    
    print("PyTorch Automatic Differentiation:")
    print(f"  Objective: {loss.item():.6f}")
    print(f"  grad_C norm: {np.linalg.norm(grad_C_torch):.4f}")
    print(f"  grad_A norm: {np.linalg.norm(grad_A_torch):.4f}")
    
    # Compare with our analytical gradient
    from rarl2_gradient_corrected import compute_rarl2_gradient_corrected
    
    grad_C_analytical, grad_A_analytical = compute_rarl2_gradient_corrected(
        C_np, A_np, A_F, B_F, C_F, D_F
    )
    
    print("\nAnalytical Gradient:")
    print(f"  grad_C norm: {np.linalg.norm(grad_C_analytical):.4f}")
    print(f"  grad_A norm: {np.linalg.norm(grad_A_analytical):.4f}")
    
    # Compare element-wise (first element)
    print("\nElement-wise comparison (first element of C):")
    print(f"  PyTorch:    {grad_C_torch.flat[0]:.6f}")
    print(f"  Analytical: {grad_C_analytical.flat[0]:.6f}")
    
    # Also test with finite differences
    from rarl2_integrated import compute_objective_lossless
    
    eps = 1e-7
    obj_base = compute_objective_lossless(C_np, A_np, A_F, B_F, C_F, D_F)
    
    C_pert = C_np.copy()
    C_pert.flat[0] += eps
    obj_pert = compute_objective_lossless(C_pert, A_np, A_F, B_F, C_F, D_F)
    
    grad_fd = (obj_pert - obj_base) / eps
    
    print(f"  Finite diff: {np.real(grad_fd):.6f}")
    
    # Test if gradients preserve output normal constraint
    dON = (A_np.conj().T @ grad_A_analytical + grad_A_analytical.conj().T @ A_np + 
           C_np.conj().T @ grad_C_analytical + grad_C_analytical.conj().T @ C_np)
    print(f"\nOutput normal constraint preservation: {np.linalg.norm(dON):.2e}")


if __name__ == "__main__":
    test_gradient_with_autodiff()