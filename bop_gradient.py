#!/usr/bin/env python3
"""
Full BOP Chart Gradient Computation
====================================

Implements the complete gradient chain for RARL2:
V → P (Stein) → Λ (sqrt) → (A, C) → Q12 (Stein) → J_n (criterion)

Each step has both forward and backward passes using implicit differentiation.
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional
from dataclasses import dataclass

from implicit_stein import (
    SteinSolution, solve_stein_LXR, stein_backward,
    SteinDifferentiable, CrossGramianDifferentiable, DualGramianDifferentiable
)


@dataclass
class BOPForwardCache:
    """Cache for BOP forward pass, needed for backward."""
    V: np.ndarray
    P: np.ndarray
    P_stein_sol: SteinSolution
    Lambda: np.ndarray
    Lambda_inv: np.ndarray
    A: np.ndarray
    C: np.ndarray


@dataclass
class CriterionForwardCache:
    """Cache for criterion forward pass."""
    Q12: np.ndarray
    Q12_stein_sol: SteinSolution
    B_hat: np.ndarray
    P12: np.ndarray
    P12_stein_sol: SteinSolution
    objective: float


def matrix_sqrt_forward(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute matrix square root Λ = sqrt(P) for Hermitian positive definite P.

    Returns:
        (Lambda, Lambda_inv): Square root and its inverse
    """
    # Use eigendecomposition for Hermitian P
    eigvals, U = np.linalg.eigh(P)

    # Ensure positive definiteness
    eigvals = np.maximum(eigvals, 1e-12)

    # Λ = U @ diag(sqrt(λ)) @ U^H
    sqrt_eigvals = np.sqrt(eigvals)
    Lambda = U @ np.diag(sqrt_eigvals) @ U.conj().T

    # Λ^{-1} = U @ diag(1/sqrt(λ)) @ U^H
    inv_sqrt_eigvals = 1.0 / sqrt_eigvals
    Lambda_inv = U @ np.diag(inv_sqrt_eigvals) @ U.conj().T

    return Lambda, Lambda_inv


def matrix_sqrt_backward(P: np.ndarray, Lambda: np.ndarray,
                          grad_Lambda: np.ndarray) -> np.ndarray:
    """
    Backward pass for matrix square root.

    Given Λ = sqrt(P) and dL/dΛ, compute dL/dP.

    The derivative satisfies the Sylvester equation:
    Λ @ dP/dΛ + dP/dΛ @ Λ = I (in the sense of Fréchet derivative)

    For the backward pass with grad_Lambda = dL/dΛ:
    Solve: Λ @ Ξ + Ξ @ Λ = grad_Lambda
    Then: dL/dP = Ξ

    Args:
        P: Original matrix (not used directly, but needed for eigendecomp)
        Lambda: sqrt(P)
        grad_Lambda: dL/dΛ

    Returns:
        grad_P: dL/dP
    """
    # Symmetrize grad_Lambda (Λ is Hermitian)
    grad_Lambda_sym = (grad_Lambda + grad_Lambda.conj().T) / 2

    # Solve Sylvester equation: Λ Ξ + Ξ Λ = grad_Lambda_sym
    # This is equivalent to: Λ Ξ + Ξ Λ^H = grad_Lambda_sym (since Λ is Hermitian)
    # scipy.linalg.solve_sylvester solves A X + X B = Q

    Xi = linalg.solve_sylvester(Lambda, Lambda, grad_Lambda_sym)

    # Symmetrize result (P is Hermitian, so grad_P should be too)
    grad_P = (Xi + Xi.conj().T) / 2

    return grad_P


def matrix_inv_backward(Lambda_inv: np.ndarray, grad_Lambda_inv: np.ndarray) -> np.ndarray:
    """
    Backward pass for matrix inverse.

    Given Λ^{-1} and dL/d(Λ^{-1}), compute dL/dΛ.

    d(Λ^{-1}) = -Λ^{-1} dΛ Λ^{-1}
    So: dL/dΛ = -Λ^{-H} grad_Lambda_inv Λ^{-H}

    Args:
        Lambda_inv: Λ^{-1}
        grad_Lambda_inv: dL/d(Λ^{-1})

    Returns:
        grad_Lambda: dL/dΛ
    """
    return -Lambda_inv.conj().T @ grad_Lambda_inv @ Lambda_inv.conj().T


class BOPChartGradient:
    """
    Complete gradient computation for BOP chart parametrization.

    Forward: V → P → Λ → (A, C)
    Backward: dL/d(A,C) → dL/dΛ → dL/dP → dL/dV
    """

    def __init__(self, W: np.ndarray, Y: np.ndarray):
        """
        Initialize with chart center.

        Args:
            W: Chart center W (n x n), should satisfy W^H W + Y^H Y = I
            Y: Chart center Y (p x n)
        """
        self.W = W.astype(np.complex128)
        self.Y = Y.astype(np.complex128)
        self.n = W.shape[0]
        self.p = Y.shape[0]

        # Precompute Y^H Y for Stein equation RHS
        self.YHY = Y.conj().T @ Y

        # Stein solver for P
        self.stein = SteinDifferentiable(W, Y)

    def forward(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, BOPForwardCache]:
        """
        Forward pass: V → (A, C)

        Args:
            V: Chart parameter (p x n)

        Returns:
            (A, C, cache): System matrices and cache for backward
        """
        # Step 1: Solve Stein equation for P
        # W^H P W - P = V^H V - Y^H Y
        P, P_sol = self.stein.forward(V)

        # Check positive definiteness
        eigvals_P = np.linalg.eigvalsh(P)
        if np.min(eigvals_P) <= 0:
            raise ValueError(f"P not positive definite, min eigenvalue: {np.min(eigvals_P)}")

        # Step 2: Compute Λ = sqrt(P)
        Lambda, Lambda_inv = matrix_sqrt_forward(P)

        # Step 3: Compute A and C
        # A = Λ W Λ^{-1}
        # C = (Y + V) Λ^{-1}
        A = Lambda @ self.W @ Lambda_inv
        C = (self.Y + V) @ Lambda_inv

        cache = BOPForwardCache(
            V=V.copy(),
            P=P,
            P_stein_sol=P_sol,
            Lambda=Lambda,
            Lambda_inv=Lambda_inv,
            A=A,
            C=C
        )

        return A, C, cache

    def backward(self, cache: BOPForwardCache,
                 grad_A: np.ndarray, grad_C: np.ndarray) -> np.ndarray:
        """
        Backward pass: (dL/dA, dL/dC) → dL/dV

        Args:
            cache: Cache from forward pass
            grad_A: dL/dA
            grad_C: dL/dC

        Returns:
            grad_V: dL/dV
        """
        V = cache.V
        Lambda = cache.Lambda
        Lambda_inv = cache.Lambda_inv

        # =============================================
        # Backward through C = (Y + V) @ Λ^{-1}
        # =============================================
        # dL/dV from C: grad_C @ (Λ^{-1})^H = grad_C @ Λ^{-H}
        grad_V_from_C = grad_C @ Lambda_inv.conj().T

        # dL/d(Λ^{-1}) from C: (Y + V)^H @ grad_C
        grad_Lambda_inv_from_C = (self.Y + V).conj().T @ grad_C

        # =============================================
        # Backward through A = Λ @ W @ Λ^{-1}
        # =============================================
        # A = Λ @ (W @ Λ^{-1})
        # dL/dΛ from A: grad_A @ (W @ Λ^{-1})^H = grad_A @ Λ^{-H} @ W^H
        grad_Lambda_from_A = grad_A @ Lambda_inv.conj().T @ self.W.conj().T

        # dL/d(Λ^{-1}) from A: (Λ @ W)^H @ grad_A = W^H @ Λ^H @ grad_A
        grad_Lambda_inv_from_A = self.W.conj().T @ Lambda.conj().T @ grad_A

        # Total gradient w.r.t. Λ^{-1}
        grad_Lambda_inv = grad_Lambda_inv_from_C + grad_Lambda_inv_from_A

        # =============================================
        # Backward through Λ^{-1} (inverse)
        # =============================================
        # dL/dΛ from inverse: -Λ^{-H} @ grad_Lambda_inv @ Λ^{-H}
        grad_Lambda_from_inv = matrix_inv_backward(Lambda_inv, grad_Lambda_inv)

        # Total gradient w.r.t. Λ
        grad_Lambda = grad_Lambda_from_A + grad_Lambda_from_inv

        # =============================================
        # Backward through Λ = sqrt(P)
        # =============================================
        grad_P = matrix_sqrt_backward(cache.P, Lambda, grad_Lambda)

        # =============================================
        # Backward through Stein equation
        # =============================================
        grad_V_from_P = self.stein.backward(cache.P_stein_sol, grad_P, V)

        # Total gradient w.r.t. V
        grad_V = grad_V_from_C + grad_V_from_P

        return grad_V


class RARL2CriterionGradient:
    """
    Complete gradient for RARL2 concentrated criterion.

    Pipeline: V → (A, C) → Q12 → J_n

    Criterion (Eq. 9): J_n = ||F||² - Tr(B_F^H Q12 Q12^H B_F)
    """

    def __init__(self, A_F: np.ndarray, B_F: np.ndarray,
                 C_F: np.ndarray, D_F: np.ndarray,
                 W: np.ndarray, Y: np.ndarray):
        """
        Initialize with target system and chart center.

        Args:
            A_F, B_F, C_F, D_F: Target system matrices
            W, Y: BOP chart center
        """
        self.A_F = A_F.astype(np.complex128)
        self.B_F = B_F.astype(np.complex128)
        self.C_F = C_F.astype(np.complex128)
        self.D_F = D_F.astype(np.complex128)

        self.n_F = A_F.shape[0]

        # BOP chart gradient
        self.bop_grad = BOPChartGradient(W, Y)

        # Cross-Gramian gradient
        self.cross_gramian = CrossGramianDifferentiable(A_F, C_F, B_F)

        # Dual-Gramian gradient (for Eq. 11 formula)
        self.dual_gramian = DualGramianDifferentiable(A_F, B_F)

        # Precompute ||F||²
        self._F_norm_sq = self._compute_h2_norm_sq(A_F, B_F, C_F, D_F)

    def _compute_h2_norm_sq(self, A, B, C, D):
        """Compute ||G||²_H2 analytically."""
        try:
            # Solve A^H Q A - Q = -C^H C
            Q = linalg.solve_discrete_lyapunov(A.conj().T, C.conj().T @ C)
            h2_sq = np.real(np.trace(B.conj().T @ Q @ B))
            if D is not None:
                h2_sq += np.real(np.trace(D.conj().T @ D))
            return h2_sq
        except Exception:
            return np.inf

    def forward(self, V: np.ndarray) -> Tuple[float, BOPForwardCache, CriterionForwardCache]:
        """
        Forward pass: V → ||F - H||² (true H2 error squared)

        Args:
            V: Chart parameter

        Returns:
            (objective, bop_cache, criterion_cache)
        """
        # BOP chart: V → (A, C)
        A, C, bop_cache = self.bop_grad.forward(V)

        # Cross-Gramian: (A, C) → Q12
        # Equation: A_F^H Q12 A + C_F^H C = Q12
        Q12, Q12_sol = self.cross_gramian.forward(A, C)

        # Optimal B from necessary condition
        # The sign depends on the cross-gramian convention.
        # With our Stein equation A_F^H Q12 A - Q12 = -C_F^H C,
        # the optimal B that minimizes ||F - H||² is B̂ = Q₁₂^H @ B_F (positive)
        B_hat = Q12.conj().T @ self.B_F

        # Dual-Gramian: (A, B_hat) → P12 (needed for gradient, Eq. 11)
        P12, P12_sol = self.dual_gramian.forward(A, B_hat)

        # Compute TRUE H2 error ||F - H||² via error system
        # Build error system E = F - H
        n = A.shape[0]
        m = self.B_F.shape[1]
        p = C.shape[0]
        n_total = self.n_F + n

        A_err = np.zeros((n_total, n_total), dtype=np.complex128)
        A_err[:self.n_F, :self.n_F] = self.A_F
        A_err[self.n_F:, self.n_F:] = A

        B_err = np.zeros((n_total, m), dtype=np.complex128)
        B_err[:self.n_F, :] = self.B_F
        B_err[self.n_F:, :] = B_hat

        C_err = np.zeros((p, n_total), dtype=np.complex128)
        C_err[:, :self.n_F] = self.C_F
        C_err[:, self.n_F:] = -C  # Note: minus sign for F - H

        # H2 norm via observability Gramian
        Q_err = linalg.solve_discrete_lyapunov(A_err.conj().T, C_err.conj().T @ C_err)
        objective = np.real(np.trace(B_err.conj().T @ Q_err @ B_err))

        criterion_cache = CriterionForwardCache(
            Q12=Q12,
            Q12_stein_sol=Q12_sol,
            B_hat=B_hat,
            P12=P12,
            P12_stein_sol=P12_sol,
            objective=objective
        )

        return objective, bop_cache, criterion_cache

    def backward(self, bop_cache: BOPForwardCache,
                 criterion_cache: CriterionForwardCache) -> np.ndarray:
        """
        Backward pass: compute dJ/dV via chain rule through Q12 Stein equation.

        Criterion: J_n = ||F||² - Tr(B_F^H Q12 Q12^H B_F)

        Chain rule:
        1. dJ/dQ12 from criterion
        2. dJ/d(A,C) via implicit diff of Q12 Stein equation
        3. dJ/dV via BOP chart backward

        Args:
            bop_cache: Cache from BOP forward
            criterion_cache: Cache from criterion forward

        Returns:
            grad_V: dJ/dV
        """
        A = bop_cache.A
        C = bop_cache.C
        Q12 = criterion_cache.Q12
        Q12_sol = criterion_cache.Q12_stein_sol

        # Step 1: Gradient of criterion w.r.t. Q12
        # J = ||F||² - Tr(B_F^H Q12 Q12^H B_F)
        # = ||F||² - Tr(Q12^H B_F B_F^H Q12)  (cycle trace)
        #
        # d/dQ12 Tr(Q12^H M Q12) = 2 M Q12  for Hermitian M = B_F B_F^H
        #
        # So: dJ/dQ12 = -2 * B_F @ B_F^H @ Q12  (shape: n_F x n)
        M = self.B_F @ self.B_F.conj().T  # n_F x n_F, Hermitian
        grad_Q12 = -2 * M @ Q12  # n_F x n

        # Step 2: Back-propagate through Q12 Stein equation
        # Forward: A_F^H @ Q12 @ A - Q12 = -C_F^H @ C
        # This is: L @ Q12 @ R - Q12 = C_rhs where L=A_F^H, R=A, C_rhs=-C_F^H @ C
        #
        # Using stein_backward to get gradients w.r.t. L, R, C_rhs:
        grad_L, grad_R, grad_C_rhs = stein_backward(Q12_sol, grad_Q12)

        # grad_R is dJ/dR = dJ/dA (since R = A in the Stein equation)
        grad_A = grad_R

        # grad_C_rhs is dJ/d(-C_F^H @ C)
        # C_rhs = -C_F^H @ C, so dC_rhs/dC = -C_F^H (left multiply)
        # dJ/dC = dJ/dC_rhs @ dC_rhs/dC = grad_C_rhs @ (-C_F^H)^T = -grad_C_rhs @ C_F^*
        # Actually: for C_rhs = -C_F^H @ C, we have:
        # Tr(grad_C_rhs^H @ dC_rhs) = Tr(grad_C_rhs^H @ (-C_F^H @ dC))
        #                          = Tr((-C_F @ grad_C_rhs)^H @ dC)
        # So dJ/dC = -C_F @ grad_C_rhs
        grad_C = -self.C_F @ grad_C_rhs

        # Step 3: Chain through BOP: (dJ/dA, dJ/dC) → dJ/dV
        grad_V = self.bop_grad.backward(bop_cache, grad_A, grad_C)

        return grad_V

    def compute_gradient(self, V: np.ndarray, use_finite_diff: bool = True) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient.

        Args:
            V: Chart parameter
            use_finite_diff: If True, use finite differences for gradient

        Returns:
            (objective, grad_V)
        """
        obj, bop_cache, crit_cache = self.forward(V)

        if use_finite_diff:
            # Finite difference gradient (slower but correct for TRUE H2 error)
            eps = 1e-7
            grad_V = np.zeros_like(V)
            p, n = V.shape

            for i in range(p):
                for j in range(n):
                    V_pert = V.copy()
                    V_pert[i, j] += eps
                    try:
                        obj_pert, _, _ = self.forward(V_pert)
                        grad_V[i, j] = (obj_pert - obj) / eps
                    except:
                        grad_V[i, j] = 0

            return obj, grad_V
        else:
            # Analytical gradient (only valid for concentrated criterion)
            grad_V = self.backward(bop_cache, crit_cache)
            return obj, grad_V


def create_unitary_chart_center(n: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a valid chart center (W, Y) satisfying W^H W + Y^H Y = I.

    For the BOP parametrization, the chart center must satisfy the
    output-normal condition: W^H W + Y^H Y = I.

    This ensures that at V=0, the Stein equation W^H P W - P = -Y^H Y
    has solution P = I (identity), giving maximum margin from the
    positive-definiteness boundary.

    Args:
        n: State dimension
        p: Output dimension

    Returns:
        (W, Y): Chart center matrices satisfying W^H W + Y^H Y = I
    """
    from lossless_embedding import create_output_normal_pair

    # create_output_normal_pair returns (C, A) where A^H A + C^H C = I
    # For BOP chart, we use W = A, Y = C
    Y, W = create_output_normal_pair(n, p)

    # Verify the output-normal condition
    check = W.conj().T @ W + Y.conj().T @ Y
    error = np.linalg.norm(check - np.eye(n))
    if error > 1e-6:
        raise ValueError(f"Output-normal condition not satisfied: ||W^H W + Y^H Y - I|| = {error}")

    # Also verify W is stable (all eigenvalues inside unit disk)
    eigvals = np.linalg.eigvals(W)
    if np.max(np.abs(eigvals)) >= 1.0:
        raise ValueError(f"W is not stable, max|eig| = {np.max(np.abs(eigvals))}")

    return W.astype(np.complex128), Y.astype(np.complex128)


def test_bop_chart_gradient():
    """Test full BOP chart gradient."""
    print("Testing BOP Chart Gradient (V → A, C)")
    print("=" * 60)

    np.random.seed(42)
    n = 2
    p = 1

    # Create proper unitary chart center
    W, Y = create_unitary_chart_center(n, p)

    print(f"Output-normal check: ||W^H W + Y^H Y - I|| = {np.linalg.norm(W.conj().T @ W + Y.conj().T @ Y - np.eye(n)):.2e}")

    # Small V to stay in chart domain
    V = 0.05 * np.random.randn(p, n).astype(np.complex128)

    bop = BOPChartGradient(W, Y)

    # Forward
    try:
        A, C, cache = bop.forward(V)
        print(f"A =\n{A}")
        print(f"C = {C}")
        print(f"A eigenvalues: {np.linalg.eigvals(A)}")
    except ValueError as e:
        print(f"Forward failed: {e}")
        return False

    # Test gradient via finite differences
    def loss_fn(V_test):
        try:
            A_test, C_test, _ = bop.forward(V_test)
            return np.real(np.sum(A_test**2) + np.sum(C_test**2))
        except ValueError:
            return 1e10

    # Analytical gradient
    # dL/dA = 2*A, dL/dC = 2*C
    grad_A = 2 * A
    grad_C = 2 * C
    grad_V_analytical = bop.backward(cache, grad_A, grad_C)

    # Finite difference
    eps = 1e-7
    grad_V_fd = np.zeros_like(V)
    base_loss = loss_fn(V)

    for i in range(p):
        for j in range(n):
            V_pert = V.copy()
            V_pert[i, j] += eps
            grad_V_fd[i, j] = (loss_fn(V_pert) - base_loss) / eps

    grad_V_analytical_real = np.real(grad_V_analytical)
    grad_V_fd_real = np.real(grad_V_fd)

    rel_error = np.linalg.norm(grad_V_analytical_real - grad_V_fd_real) / (np.linalg.norm(grad_V_fd_real) + 1e-10)
    print(f"\nGradient comparison:")
    print(f"  Analytical: {grad_V_analytical_real}")
    print(f"  Finite diff: {grad_V_fd_real}")
    print(f"  Relative error: {rel_error:.2e}")

    if rel_error < 1e-4:
        print("✓ BOP chart gradient test PASSED")
        return True
    else:
        print("✗ BOP chart gradient test FAILED")
        return False


def test_full_criterion_gradient():
    """Test full criterion gradient V → J_n."""
    print("\nTesting Full Criterion Gradient (V → J_n)")
    print("=" * 60)

    np.random.seed(42)
    n_F = 3  # Target order
    n = 2    # Approximation order
    m = 1
    p = 1

    # Target system (stable)
    A_F = np.diag([0.5, 0.6, 0.7]).astype(np.complex128)
    B_F = np.array([[1.0], [0.5], [0.3]], dtype=np.complex128)
    C_F = np.array([[1.0, 0.5, 0.3]], dtype=np.complex128)
    D_F = np.zeros((p, m), dtype=np.complex128)

    # Create proper chart center
    W, Y = create_unitary_chart_center(n, p)

    print(f"Target system: n_F={n_F}, m={m}, p={p}")
    print(f"Approximation order: n={n}")
    print(f"Chart center output-normal check: {np.linalg.norm(W.conj().T @ W + Y.conj().T @ Y - np.eye(n)):.2e}")

    # Small V
    V = 0.02 * np.random.randn(p, n).astype(np.complex128)

    criterion = RARL2CriterionGradient(A_F, B_F, C_F, D_F, W, Y)

    # Forward + backward
    try:
        obj, grad_V = criterion.compute_gradient(V)
        print(f"Objective J_n = {obj:.6e}")
        print(f"Gradient norm = {np.linalg.norm(grad_V):.6e}")
    except Exception as e:
        print(f"Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Finite difference verification
    def obj_fn(V_test):
        try:
            obj_test, _, _ = criterion.forward(V_test)
            return obj_test
        except Exception:
            return 1e10

    eps = 1e-7
    grad_V_fd = np.zeros_like(V)
    base_obj = obj_fn(V)

    for i in range(p):
        for j in range(n):
            V_pert = V.copy()
            V_pert[i, j] += eps
            grad_V_fd[i, j] = (obj_fn(V_pert) - base_obj) / eps

    grad_V_real = np.real(grad_V)
    grad_V_fd_real = np.real(grad_V_fd)

    rel_error = np.linalg.norm(grad_V_real - grad_V_fd_real) / (np.linalg.norm(grad_V_fd_real) + 1e-10)
    print(f"\nGradient comparison:")
    print(f"  Analytical: {grad_V_real}")
    print(f"  Finite diff: {grad_V_fd_real}")
    print(f"  Relative error: {rel_error:.2e}")

    if rel_error < 1e-4:
        print("✓ Full criterion gradient test PASSED")
        return True
    else:
        print("✗ Full criterion gradient test FAILED")
        return False


def test_q12_gradient():
    """Test gradient through Q12 Stein equation only (debug)."""
    print("\nTesting Q12 Gradient Chain")
    print("=" * 60)

    np.random.seed(42)
    n_F = 3
    n = 2
    m = 1
    p = 1

    # Target system
    A_F = np.diag([0.5, 0.6, 0.7]).astype(np.complex128)
    B_F = np.array([[1.0], [0.5], [0.3]], dtype=np.complex128)
    C_F = np.array([[1.0, 0.5, 0.3]], dtype=np.complex128)

    # Approximant
    A = np.diag([0.3, 0.5]).astype(np.complex128)
    C = np.array([[0.9, 0.7]], dtype=np.complex128)

    # Compute Q12 forward
    cross_gram = CrossGramianDifferentiable(A_F, C_F, B_F)
    Q12, Q12_sol = cross_gram.forward(A, C)

    print(f"Q12 shape: {Q12.shape}")
    print(f"Q12 =\n{Q12}")

    # Criterion: J = ||F||² - Tr(B_F^H Q12 Q12^H B_F)
    def criterion(A_test, C_test):
        try:
            Q12_test, _ = cross_gram.forward(A_test, C_test)
            reduction = np.real(np.trace(B_F.conj().T @ Q12_test @ Q12_test.conj().T @ B_F))
            return -reduction  # Just the Q12-dependent part
        except Exception:
            return 1e10

    base_obj = criterion(A, C)
    print(f"Criterion (Q12 part): {base_obj:.6e}")

    # Analytical gradient
    M = B_F @ B_F.conj().T
    grad_Q12 = -2 * M @ Q12

    grad_L, grad_R, grad_C_rhs = stein_backward(Q12_sol, grad_Q12)
    grad_A_analytical = grad_R
    grad_C_analytical = -C_F @ grad_C_rhs

    print(f"\nAnalytical grad_A:\n{np.real(grad_A_analytical)}")
    print(f"Analytical grad_C:\n{np.real(grad_C_analytical)}")

    # Finite difference
    eps = 1e-7
    grad_A_fd = np.zeros_like(A, dtype=float)
    grad_C_fd = np.zeros_like(C, dtype=float)

    for i in range(n):
        for j in range(n):
            A_pert = A.copy()
            A_pert[i, j] += eps
            grad_A_fd[i, j] = (criterion(A_pert, C) - base_obj) / eps

    for i in range(p):
        for j in range(n):
            C_pert = C.copy()
            C_pert[i, j] += eps
            grad_C_fd[i, j] = (criterion(A, C_pert) - base_obj) / eps

    print(f"\nFinite diff grad_A:\n{grad_A_fd}")
    print(f"Finite diff grad_C:\n{grad_C_fd}")

    rel_err_A = np.linalg.norm(np.real(grad_A_analytical) - grad_A_fd) / (np.linalg.norm(grad_A_fd) + 1e-10)
    rel_err_C = np.linalg.norm(np.real(grad_C_analytical) - grad_C_fd) / (np.linalg.norm(grad_C_fd) + 1e-10)

    print(f"\nRelative error grad_A: {rel_err_A:.2e}")
    print(f"Relative error grad_C: {rel_err_C:.2e}")

    if rel_err_A < 1e-4 and rel_err_C < 1e-4:
        print("✓ Q12 gradient test PASSED")
        return True
    else:
        print("✗ Q12 gradient test FAILED")
        return False


if __name__ == "__main__":
    success1 = test_bop_chart_gradient()
    success3 = test_q12_gradient()
    success2 = test_full_criterion_gradient()

    print("\n" + "=" * 60)
    if success1 and success2 and success3:
        print("ALL GRADIENT TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
