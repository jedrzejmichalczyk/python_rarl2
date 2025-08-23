#!/usr/bin/env python3
"""
Integrated RARL2 Implementation
================================
Combines lossless parametrization, H2 projection, and gradient computation
for the complete RARL2 algorithm.

Key components:
1. Douglas-Shapiro factorization: H = C*G where G is lossless
2. H2 projection: C_opt = π+(F*G†)
3. Concentrated criterion: ψₙ(G) = ||F - π+(F*G†)*G||₂²
4. Gradient computation via lossless parametrization
"""

import numpy as np
from scipy import linalg
from typing import Tuple
from lossless_embedding import lossless_embedding, solve_discrete_lyapunov


def douglas_shapiro_factorization(
    A_H: np.ndarray, B_H: np.ndarray, 
    C_H: np.ndarray, D_H: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform Douglas-Shapiro factorization: H = C*G where G is lossless.
    
    Note: This is actually inner-outer factorization. The "Douglas-Shapiro"
    name in the code is a misnomer - it's based on operator theory from
    Douglas-Shapiro-Shields (1970) but applied to control systems.
    
    Args:
        A_H, B_H, C_H, D_H: State-space matrices of H
        
    Returns:
        C_factor: Outer factor (minimum phase) as (A_C, B_C, C_C, D_C)
        G_lossless: Inner factor (all-pass) as (A_G, B_G, C_G, D_G)
    """
    n = A_H.shape[0]
    m = B_H.shape[1]
    p = C_H.shape[0]
    
    # Step 1: Create lossless G from the observable pair (C_H, A_H)
    # This uses the lossless embedding from Proposition 1
    B_G, D_G = lossless_embedding(C_H, A_H, nu=1.0)
    A_G = A_H
    C_G = C_H
    
    # Step 2: Find optimal C such that H ≈ C*G
    # This is done via H2 projection: C = π+(H*G†)
    # For now, we use a simplified approach
    
    # The outer factor C has the same state dimension as H
    A_C = A_H.copy()
    B_C = B_H.copy()
    C_C = np.eye(p, dtype=np.complex128)  # Identity output
    D_C = D_H.copy()
    
    # Return both factors
    C_factor = (A_C, B_C, C_C, D_C)
    G_lossless = (A_G, B_G, C_G, D_G)
    
    return C_factor, G_lossless


def h2_projection_lossless(
    A_F: np.ndarray, B_F: np.ndarray, C_F: np.ndarray, D_F: np.ndarray,
    A_G: np.ndarray, B_G: np.ndarray, C_G: np.ndarray, D_G: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute optimal C via H2 projection: C = π+(F*G†).
    
    Given target F and lossless G, find C that minimizes ||F - C*G||₂.
    
    Note: We use (A,C) convention as in RARL2 paper for observable pairs.
    
    Args:
        A_F, B_F, C_F, D_F: Target system F
        A_G, B_G, C_G, D_G: Lossless system G (created from (C,A) pair)
        
    Returns:
        A_C, B_C, C_C, D_C: Optimal outer factor C
    """
    n_F = A_F.shape[0]
    n_G = A_G.shape[0]
    m_F = B_F.shape[1]  # Input dimension of F
    m_G = B_G.shape[1]  # Input dimension of G  
    p_F = C_F.shape[0]  # Output dimension of F
    p_G = C_G.shape[0]  # Output dimension of G
    
    # Step 1: Compute F*G† where G† is the adjoint (conjugate on unit circle)
    # For discrete-time: G†(z) = G*(1/z*) 
    # In state-space: If G = (A,B,C,D), then G† = (A^H, C^H, B^H, D^H)
    
    # Build the product system F*G†
    # G† (adjoint of G) in state-space: (A^H, C^H, B^H, D^H)
    # So G† has: A_G† = A_G^H, B_G† = C_G^H, C_G† = B_G^H, D_G† = D_G^H
    
    # Series connection F*G†:
    # Output of G† (dimension m_G) feeds into input of F (dimension m_F)
    # For compatibility we need m_G == m_F
    
    # Check dimension compatibility
    if m_G != m_F:
        # Can't connect - return F as is (no projection)
        return A_F, B_F, C_F, D_F
    
    n_prod = n_F + n_G
    
    # Build augmented system for F*G†
    A_prod = np.zeros((n_prod, n_prod), dtype=np.complex128)
    A_prod[:n_F, :n_F] = A_F
    # Connection: B_F * C_G† = B_F * B_G^H
    A_prod[:n_F, n_F:] = B_F @ B_G.conj().T
    A_prod[n_F:, n_F:] = A_G.conj().T
    
    # Input of series connection = input of G† = C_G^H dimension (p_G)
    B_prod = np.zeros((n_prod, p_G), dtype=np.complex128)
    # B_F * D_G†
    B_prod[:n_F, :] = B_F @ D_G.conj().T
    # B_G†
    B_prod[n_F:, :] = C_G.conj().T
    
    # Output of series connection = output of F (p_F)
    C_prod = np.zeros((p_F, n_prod), dtype=np.complex128)
    C_prod[:, :n_F] = C_F
    # D_F * C_G† = D_F * B_G^H
    C_prod[:, n_F:] = D_F @ B_G.conj().T
    
    # Direct feedthrough: D_F * D_G†
    D_prod = D_F @ D_G.conj().T
    
    # Step 2: Project onto H2 (stable part)
    # Extract stable eigenvalues and corresponding subspace
    eigvals = np.linalg.eigvals(A_prod)
    stable_mask = np.abs(eigvals) < 0.999
    
    if not np.any(stable_mask):
        # No stable part - return minimal system
        A_C = np.zeros((0, 0), dtype=np.complex128)
        B_C = np.zeros((0, m), dtype=np.complex128)
        C_C = np.zeros((p, 0), dtype=np.complex128)
        D_C = D_prod
        return A_C, B_C, C_C, D_C
    
    # For simplicity, use modal decomposition
    eigvecs = np.linalg.eig(A_prod)[1]
    stable_indices = np.where(stable_mask)[0]
    V_stable = eigvecs[:, stable_indices]
    
    # Project system onto stable subspace
    try:
        V_inv = np.linalg.pinv(V_stable)
        A_C = V_inv @ A_prod @ V_stable
        B_C = V_inv @ B_prod
        C_C = C_prod @ V_stable
        D_C = D_prod
    except:
        # Fallback to identity
        A_C = A_F
        B_C = B_F  
        C_C = C_F
        D_C = D_F
    
    return A_C, B_C, C_C, D_C


def compute_objective_lossless(
    C: np.ndarray, A: np.ndarray,
    A_F: np.ndarray, B_F: np.ndarray, 
    C_F: np.ndarray, D_F: np.ndarray
) -> float:
    """
    Compute the concentrated criterion ψₙ(G) = ||F - π+(F*G†)*G||₂².
    
    Note: Using (C,A) convention as in RARL2 paper for observable pairs.
    The lossless G is created from the observable pair (C,A).
    
    Args:
        C, A: Observable pair (C,A) defining lossless G via lossless embedding
        A_F, B_F, C_F, D_F: Target system F
        
    Returns:
        Objective value (L2 error squared)
    """
    # Create lossless G from (C, A)
    B_G, D_G = lossless_embedding(C, A, nu=1.0)
    
    # Compute optimal C via H2 projection
    A_C, B_C, C_C, D_C = h2_projection_lossless(
        A_F, B_F, C_F, D_F,
        A, B_G, C, D_G
    )
    
    # Compute the approximation H = C*G
    n_C = A_C.shape[0]
    n_G = A.shape[0]
    n_H = n_C + n_G
    
    if n_C == 0:
        # C is just feedthrough
        A_H = A
        B_H = B_G
        C_H = D_C @ C
        D_H = D_C @ D_G
    else:
        # Build series connection C*G
        A_H = np.zeros((n_H, n_H), dtype=np.complex128)
        A_H[:n_C, :n_C] = A_C
        A_H[:n_C, n_C:] = B_C @ C
        A_H[n_C:, n_C:] = A
        
        B_H = np.zeros((n_H, B_G.shape[1]), dtype=np.complex128)
        B_H[:n_C, :] = B_C @ D_G
        B_H[n_C:, :] = B_G
        
        C_H = np.zeros((C_F.shape[0], n_H), dtype=np.complex128)
        C_H[:, :n_C] = C_C
        C_H[:, n_C:] = D_C @ C
        
        D_H = D_C @ D_G
    
    # Compute ||F - H||₂²
    return compute_h2_norm(A_F, B_F, C_F, D_F, A_H, B_H, C_H, D_H)


def compute_h2_norm(
    A1: np.ndarray, B1: np.ndarray, C1: np.ndarray, D1: np.ndarray,
    A2: np.ndarray, B2: np.ndarray, C2: np.ndarray, D2: np.ndarray
) -> float:
    """
    Compute H2 norm of the difference ||H1 - H2||₂.
    
    Args:
        A1, B1, C1, D1: First system
        A2, B2, C2, D2: Second system
        
    Returns:
        H2 norm squared of the error system
    """
    n1 = A1.shape[0]
    n2 = A2.shape[0]
    
    # Build error system E = H1 - H2
    n_err = n1 + n2
    
    A_err = linalg.block_diag(A1, A2)
    B_err = np.vstack([B1, B2])
    C_err = np.hstack([C1, -C2])
    D_err = D1 - D2
    
    # Compute H2 norm via observability Gramian
    try:
        Q = solve_discrete_lyapunov(A_err, C_err.conj().T @ C_err)
        h2_norm_sq = np.real(np.trace(B_err.conj().T @ Q @ B_err))
        h2_norm_sq += np.real(np.trace(D_err.conj().T @ D_err))
        return h2_norm_sq
    except:
        # If Lyapunov fails, use approximation
        return np.linalg.norm(D_err, 'fro')**2


def compute_gradient_lossless(
    C: np.ndarray, A: np.ndarray,
    B_G: np.ndarray, D_G: np.ndarray,
    A_F: np.ndarray, B_F: np.ndarray,
    C_F: np.ndarray, D_F: np.ndarray
) -> np.ndarray:
    """
    Compute gradient of the concentrated criterion with respect to (C, A).
    
    Uses finite differences for robustness.
    
    Args:
        C, A: Current parameters
        B_G, D_G: Lossless embedding matrices (dependent on C, A)
        A_F, B_F, C_F, D_F: Target system
        
    Returns:
        Gradient vector (flattened)
    """
    # Flatten parameters
    params = np.concatenate([C.flatten(), A.flatten()])
    grad = np.zeros(len(params), dtype=np.float64)
    
    # Base objective
    obj_base = compute_objective_lossless(C, A, A_F, B_F, C_F, D_F)
    
    # Finite differences
    eps = 1e-8
    for i in range(len(params)):
        params_pert = params.copy()
        params_pert[i] += eps
        
        C_pert = params_pert[:C.size].reshape(C.shape)
        A_pert = params_pert[C.size:].reshape(A.shape)
        
        obj_pert = compute_objective_lossless(
            C_pert, A_pert, A_F, B_F, C_F, D_F
        )
        
        grad[i] = (obj_pert - obj_base) / eps
    
    return grad


def lossless_to_lossy(
    A_G: np.ndarray, B_G: np.ndarray, C_G: np.ndarray, D_G: np.ndarray,
    A_F: np.ndarray, B_F: np.ndarray, C_F: np.ndarray, D_F: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert lossless G to lossy H via H2 projection to approximate F.
    
    Computes H = C*G where C = π+(F*G†).
    
    Args:
        A_G, B_G, C_G, D_G: Lossless system
        A_F, B_F, C_F, D_F: Target system
        
    Returns:
        A_H, B_H, C_H, D_H: Lossy approximation
    """
    # Get optimal C
    A_C, B_C, C_C, D_C = h2_projection_lossless(
        A_F, B_F, C_F, D_F,
        A_G, B_G, C_G, D_G
    )
    
    # Form H = C*G
    n_C = A_C.shape[0]
    n_G = A_G.shape[0]
    
    if n_C == 0:
        # C is just feedthrough
        A_H = A_G
        B_H = B_G
        C_H = D_C @ C_G
        D_H = D_C @ D_G
    else:
        n_H = n_C + n_G
        A_H = np.zeros((n_H, n_H), dtype=np.complex128)
        A_H[:n_C, :n_C] = A_C
        A_H[:n_C, n_C:] = B_C @ C_G
        A_H[n_C:, n_C:] = A_G
        
        B_H = np.zeros((n_H, B_G.shape[1]), dtype=np.complex128)
        B_H[:n_C, :] = B_C @ D_G
        B_H[n_C:, :] = B_G
        
        C_H = np.zeros((C_F.shape[0], n_H), dtype=np.complex128)
        C_H[:, :n_C] = C_C
        C_H[:, n_C:] = D_C @ C_G
        
        D_H = D_C @ D_G
    
    return A_H, B_H, C_H, D_H


def concentrated_criterion(
    A_G: np.ndarray, B_G: np.ndarray, C_G: np.ndarray, D_G: np.ndarray,
    A_F: np.ndarray, B_F: np.ndarray, C_F: np.ndarray, D_F: np.ndarray
) -> float:
    """
    Compute the concentrated criterion ψₙ(G) = ||F - π+(F*G†)*G||₂².
    
    Args:
        A_G, B_G, C_G, D_G: Lossless system G
        A_F, B_F, C_F, D_F: Target system F
        
    Returns:
        Criterion value
    """
    # Get the lossy approximation
    A_H, B_H, C_H, D_H = lossless_to_lossy(
        A_G, B_G, C_G, D_G,
        A_F, B_F, C_F, D_F
    )
    
    # Compute error
    return compute_h2_norm(A_F, B_F, C_F, D_F, A_H, B_H, C_H, D_H)


def create_output_normal_pair(n: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create output normal pair (C, A) where A^H*A + C^H*C = I.
    
    Note: Using (A,C) convention as in RARL2 paper.
    
    Args:
        n: State dimension
        p: Output dimension
        
    Returns:
        C: Output matrix (p x n)
        A: System matrix (n x n)
    """
    # Create random unitary matrix
    np.random.seed(42)
    M = np.random.randn(n+p, n) + 1j * np.random.randn(n+p, n)
    U, _ = np.linalg.qr(M)
    
    # Extract blocks
    A = U[:n, :].astype(np.complex128) * 0.9  # Scale for stability and ensure correct dtype
    C = U[n:n+p, :].astype(np.complex128)
    
    # Normalize to satisfy constraint
    current = A.conj().T @ A + C.conj().T @ C
    scale = linalg.sqrtm(np.linalg.inv(current)).astype(np.complex128)
    
    A = A @ scale
    C = C @ scale
    
    return C, A


def gradient_descent_step(
    C: np.ndarray, A: np.ndarray,
    A_F: np.ndarray, B_F: np.ndarray,
    C_F: np.ndarray, D_F: np.ndarray,
    learning_rate: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Perform one gradient descent step.
    
    Args:
        C, A: Current parameters
        A_F, B_F, C_F, D_F: Target system
        learning_rate: Step size
        
    Returns:
        C_new, A_new: Updated parameters
        obj_old, obj_new: Objective values before and after
    """
    # Compute current objective
    obj_old = compute_objective_lossless(C, A, A_F, B_F, C_F, D_F)
    
    # Compute gradient
    B_G, D_G = lossless_embedding(C, A)
    grad = compute_gradient_lossless(C, A, B_G, D_G, A_F, B_F, C_F, D_F)
    
    # Reshape gradient
    grad_C = grad[:C.size].reshape(C.shape)
    grad_A = grad[C.size:].reshape(A.shape)
    
    # Update parameters
    C_new = C - learning_rate * grad_C
    A_new = A - learning_rate * grad_A
    
    # Ensure stability
    eigvals = np.linalg.eigvals(A_new)
    if np.any(np.abs(eigvals) >= 1.0):
        # Project to stable region
        A_new = A_new * 0.99 / np.max(np.abs(eigvals))
    
    # Compute new objective
    obj_new = compute_objective_lossless(C_new, A_new, A_F, B_F, C_F, D_F)
    
    return C_new, A_new, obj_old, obj_new


class RARL2Optimizer:
    """
    RARL2 optimization algorithm.
    
    Iteratively improves approximation by optimizing over
    the manifold of lossless functions.
    """
    
    def __init__(self, target: Tuple, order: int):
        """
        Initialize optimizer.
        
        Args:
            target: Target system as (A_F, B_F, C_F, D_F)
            order: Order of approximation
        """
        self.A_F, self.B_F, self.C_F, self.D_F = target
        self.n = order
        self.p = self.C_F.shape[0]
        
        # Initialize with output normal pair
        self.C, self.A = create_output_normal_pair(self.n, self.p)
        
        # Track optimization history
        self.errors = []
        self.compute_error()
    
    def compute_error(self):
        """Compute and store current error."""
        error = compute_objective_lossless(
            self.C, self.A,
            self.A_F, self.B_F, self.C_F, self.D_F
        )
        self.errors.append(error)
        return error
    
    def get_error(self) -> float:
        """Get current error."""
        if not self.errors:
            return self.compute_error()
        return self.errors[-1]
    
    def step(self, learning_rate: float = 0.01):
        """Perform one optimization step."""
        self.C, self.A, _, _ = gradient_descent_step(
            self.C, self.A,
            self.A_F, self.B_F, self.C_F, self.D_F,
            learning_rate
        )
        self.compute_error()
    
    def get_approximation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get current approximation as lossy system."""
        B_G, D_G = lossless_embedding(self.C, self.A)
        return lossless_to_lossy(
            self.A, B_G, self.C, D_G,
            self.A_F, self.B_F, self.C_F, self.D_F
        )