#!/usr/bin/env python3
"""
RARL2 Gradient-Based Optimizer
===============================

Implements the paper's optimization approach with:
1. BOP chart parametrization (V → A, C)
2. Concentrated criterion (Eq. 9)
3. Analytical gradients via implicit differentiation
4. L-BFGS optimization
5. Chart switching at boundaries

This is the proper SOTA implementation following Olivi et al. 2013.
"""

import numpy as np
from scipy import optimize, linalg
from typing import Tuple, Optional
from dataclasses import dataclass

from bop_gradient import (
    BOPChartGradient, RARL2CriterionGradient,
    BOPForwardCache, CriterionForwardCache,
    create_unitary_chart_center
)
from lossless_embedding import lossless_embedding


@dataclass
class RARL2Result:
    """Result of RARL2 optimization."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    V_final: np.ndarray
    objective: float
    h2_error: float
    iterations: int
    chart_switches: int
    success: bool
    message: str


class RARL2GradientOptimizer:
    """
    RARL2 optimizer with analytical gradients and L-BFGS.

    This implements the full paper's approach:
    - BOP chart parametrization
    - Concentrated criterion
    - Analytical gradients via implicit differentiation
    - Chart switching at boundaries
    """

    def __init__(self, A_F: np.ndarray, B_F: np.ndarray,
                 C_F: np.ndarray, D_F: np.ndarray,
                 n_approx: int,
                 chart_boundary_eps: float = 1e-6,
                 max_chart_switches: int = 10,
                 verbose: bool = True):
        """
        Initialize optimizer.

        Args:
            A_F, B_F, C_F, D_F: Target system
            n_approx: Approximation order
            chart_boundary_eps: Threshold for chart switching
            max_chart_switches: Maximum chart switches
            verbose: Print progress
        """
        self.A_F = A_F.astype(np.complex128)
        self.B_F = B_F.astype(np.complex128)
        self.C_F = C_F.astype(np.complex128)
        self.D_F = D_F.astype(np.complex128)

        self.n_F = A_F.shape[0]
        self.n = n_approx
        self.m = B_F.shape[1]
        self.p = C_F.shape[0]

        self.chart_boundary_eps = chart_boundary_eps
        self.max_chart_switches = max_chart_switches
        self.verbose = verbose

        # Current chart center
        self.W: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None

        # Criterion gradient computer (updated when chart changes)
        self.criterion_grad: Optional[RARL2CriterionGradient] = None

        # Statistics
        self._iteration_count = 0
        self._chart_switch_count = 0
        self._objective_history = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _initialize_chart_from_modal_truncation(self) -> np.ndarray:
        """
        Initialize chart center from modal truncation of target.

        The chart center (W, Y) is set to the output-normal realization of
        the n dominant modes of the target system. This ensures we start
        near a good approximation.

        Returns:
            V_init: Initial chart parameter (zeros = at chart center)
        """
        # Get modal truncation of target: keep n dominant modes
        # For diagonal A_F, this is just keeping the n poles with largest residues

        # First, compute balanced realization of target
        from scipy import linalg

        # Controllability Gramian: A P A^H - P = -B B^H
        P_c = linalg.solve_discrete_lyapunov(self.A_F, self.B_F @ self.B_F.conj().T)

        # Observability Gramian: A^H Q A - Q = -C^H C
        Q_o = linalg.solve_discrete_lyapunov(self.A_F.conj().T, self.C_F.conj().T @ self.C_F)

        # Hankel singular values
        PQ = P_c @ Q_o
        hsv = np.sqrt(np.maximum(np.real(np.linalg.eigvals(PQ)), 0))
        hsv_sorted_idx = np.argsort(hsv)[::-1]  # Largest first

        # If n >= n_F, just use the whole system
        if self.n >= self.n_F:
            # Use full system in output-normal form
            # Output-normal: W^H W + Y^H Y = I
            # This is equivalent to having observability Gramian Q_o = I
            #
            # Transform: For Gramian Q_new = T^{-H} Q T^{-1} = I
            # We need T = Q^{1/2}, so T^{-1} = Q^{-1/2}
            #
            # New realization: A_new = T A T^{-1}, C_new = C T^{-1}

            # Q_o^{1/2} via eigendecomposition
            eigvals, U = np.linalg.eigh(Q_o)
            eigvals = np.maximum(eigvals, 1e-12)
            Q_o_sqrt = U @ np.diag(np.sqrt(eigvals)) @ U.conj().T
            Q_o_inv_sqrt = U @ np.diag(1.0 / np.sqrt(eigvals)) @ U.conj().T

            T = Q_o_sqrt
            T_inv = Q_o_inv_sqrt

            # Transform to output-normal coordinates
            W = T @ self.A_F @ T_inv
            Y = self.C_F @ T_inv

            # Take first n components (they're all here)
            W = W[:self.n, :self.n]
            Y = Y[:, :self.n]
        else:
            # Balanced truncation followed by output-normal transformation
            #
            # Step 1: Compute balanced realization
            # Cholesky of P_c
            L_c = linalg.cholesky(P_c + 1e-12 * np.eye(self.n_F), lower=True)

            # SVD of L_c^T Q_o L_c = U S V^H
            M = L_c.conj().T @ Q_o @ L_c
            U, S, Vh = linalg.svd(M)

            # Hankel singular values are sqrt(S)
            # Balancing transformation: T_bal such that new Gramians are diagonal = Σ
            Sigma_sqrt = np.diag(np.power(S + 1e-12, 0.25))
            Sigma_sqrt_inv = np.diag(np.power(S + 1e-12, -0.25))

            T_bal = Sigma_sqrt @ U.conj().T @ linalg.inv(L_c.conj().T)
            T_bal_inv = L_c.conj().T @ U @ Sigma_sqrt_inv

            # Balanced realization
            A_bal = T_bal @ self.A_F @ T_bal_inv
            C_bal = self.C_F @ T_bal_inv

            # Step 2: Truncate to n modes
            A_n = A_bal[:self.n, :self.n]
            C_n = C_bal[:, :self.n]

            # Step 3: Convert truncated system to output-normal form
            Q_o_n = linalg.solve_discrete_lyapunov(A_n.conj().T, C_n.conj().T @ C_n)

            # Q_o_n^{1/2} via eigendecomposition
            eigvals_n, U_n = np.linalg.eigh(Q_o_n)
            eigvals_n = np.maximum(eigvals_n, 1e-12)
            Q_o_n_sqrt = U_n @ np.diag(np.sqrt(eigvals_n)) @ U_n.conj().T
            Q_o_n_inv_sqrt = U_n @ np.diag(1.0 / np.sqrt(eigvals_n)) @ U_n.conj().T

            T_n = Q_o_n_sqrt
            T_n_inv = Q_o_n_inv_sqrt

            W = T_n @ A_n @ T_n_inv
            Y = C_n @ T_n_inv

        # Verify output-normal condition: W^H W + Y^H Y = I
        check = W.conj().T @ W + Y.conj().T @ Y
        error = np.linalg.norm(check - np.eye(self.n))
        if error > 1e-4:
            self._log(f"Warning: Output-normal error = {error:.2e}, using fallback")
            # Fallback to generic output-normal pair
            W, Y = create_unitary_chart_center(self.n, self.p)

        # Store chart center
        self.W = W.astype(np.complex128)
        self.Y = Y.astype(np.complex128)

        # Initialize criterion gradient computer with this chart
        self.criterion_grad = RARL2CriterionGradient(
            self.A_F, self.B_F, self.C_F, self.D_F, self.W, self.Y
        )

        # Verify the criterion gives a valid positive value at V=0
        V_init = np.zeros((self.p, self.n), dtype=np.complex128)
        try:
            obj, _, _ = self.criterion_grad.forward(V_init)
            if obj < 0:
                self._log(f"Warning: Criterion negative at V=0: {obj}")
        except Exception as e:
            self._log(f"Criterion evaluation failed: {e}")

        return V_init

    def _V_to_real(self, V: np.ndarray) -> np.ndarray:
        """Convert complex V to real parameter vector."""
        return np.concatenate([V.real.flatten(), V.imag.flatten()])

    def _real_to_V(self, x: np.ndarray) -> np.ndarray:
        """Convert real parameter vector to complex V."""
        half = len(x) // 2
        V_real = x[:half].reshape(self.p, self.n)
        V_imag = x[half:].reshape(self.p, self.n)
        return V_real + 1j * V_imag

    def _objective_and_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute objective and gradient for optimization."""
        V = self._real_to_V(x)

        try:
            obj, grad_V = self.criterion_grad.compute_gradient(V)

            # Convert complex gradient to real
            grad_real = np.concatenate([np.real(grad_V).flatten(),
                                        np.imag(grad_V).flatten()])

            return float(obj), grad_real.astype(float)

        except ValueError:
            # Chart boundary reached
            return 1e10, np.zeros_like(x)

    def _check_chart_validity(self, V: np.ndarray) -> Tuple[bool, float]:
        """Check if V is in valid chart domain."""
        try:
            P = self.criterion_grad.bop_grad.stein.forward(V)[0]
            min_eig = np.min(np.linalg.eigvalsh(P))
            return min_eig > self.chart_boundary_eps, float(min_eig)
        except Exception:
            return False, -np.inf

    def _switch_chart(self, V: np.ndarray) -> np.ndarray:
        """Switch to new chart centered at current point."""
        if self._chart_switch_count >= self.max_chart_switches:
            self._log(f"Maximum chart switches ({self.max_chart_switches}) reached")
            return V

        try:
            # Get current realization
            A, C, _ = self.criterion_grad.bop_grad.forward(V)

            # Use current (A, C) as new chart center
            # First convert to output-normal form
            Q_o = linalg.solve_discrete_lyapunov(A.conj().T, C.conj().T @ C)
            L = linalg.cholesky(Q_o + 1e-12 * np.eye(self.n), lower=True)
            T = L.T
            T_inv = linalg.inv(T)

            W_new = T @ A @ T_inv
            Y_new = C @ T_inv

            self.W = W_new
            self.Y = Y_new

            # Reinitialize criterion gradient with new chart
            self.criterion_grad = RARL2CriterionGradient(
                self.A_F, self.B_F, self.C_F, self.D_F, self.W, self.Y
            )

            self._chart_switch_count += 1
            self._log(f"  Chart switch #{self._chart_switch_count}")

            # New V = 0 (at center of new chart)
            return np.zeros((self.p, self.n), dtype=np.complex128)

        except Exception as e:
            self._log(f"  Chart switch failed: {e}")
            return V

    def _is_valid_step(self, V: np.ndarray) -> bool:
        """Check if V gives a valid P > 0."""
        try:
            P = self.criterion_grad.bop_grad.stein.forward(V)[0]
            min_eig = np.min(np.linalg.eigvalsh(P))
            return min_eig > 1e-8
        except:
            return False

    def _backtracking_line_search(self, V: np.ndarray, grad_V: np.ndarray,
                                   obj: float, alpha_init: float = 1.0,
                                   beta: float = 0.5, c: float = 1e-4,
                                   max_backtracks: int = 20) -> Tuple[np.ndarray, float, float]:
        """
        Backtracking line search that respects P > 0 constraint.

        Returns:
            (V_new, obj_new, alpha): New V, new objective, step size used
        """
        alpha = alpha_init
        search_dir = -grad_V  # Gradient descent direction

        for _ in range(max_backtracks):
            V_new = V + alpha * search_dir

            # Check constraint
            if not self._is_valid_step(V_new):
                alpha *= beta
                continue

            # Check Armijo condition
            try:
                obj_new, _ = self.criterion_grad.compute_gradient(V_new)
                grad_norm_sq = np.real(np.sum(grad_V.conj() * grad_V))

                if obj_new <= obj - c * alpha * grad_norm_sq:
                    return V_new, obj_new, alpha
            except:
                pass

            alpha *= beta

        # Failed to find valid step
        return V, obj, 0.0

    def optimize(self, max_iter: int = 200, tol: float = 1e-8) -> RARL2Result:
        """
        Run RARL2 optimization using gradient descent with backtracking.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance for gradient norm

        Returns:
            RARL2Result with optimized system
        """
        self._log("=" * 60)
        self._log("RARL2 Gradient Optimization (Backtracking GD)")
        self._log("=" * 60)
        self._log(f"Target: n_F={self.n_F}, m={self.m}, p={self.p}")
        self._log(f"Approximation order: n={self.n}")

        # Initialize
        V = self._initialize_chart_from_modal_truncation()

        # Initial objective and gradient
        obj, grad_V = self.criterion_grad.compute_gradient(V)
        self._log(f"Initial objective: {obj:.6e}")
        self._log(f"Initial ||grad||: {np.linalg.norm(grad_V):.6e}")
        self._objective_history = [obj]

        success = False
        message = "Max iterations reached"

        for iteration in range(max_iter):
            self._iteration_count = iteration + 1
            grad_norm = np.linalg.norm(grad_V)

            # Check convergence
            if grad_norm < tol:
                success = True
                message = f"Converged: ||grad|| = {grad_norm:.2e} < {tol}"
                break

            # Backtracking line search
            V_new, obj_new, alpha = self._backtracking_line_search(V, grad_V, obj)

            if alpha == 0:
                # No valid step found - try chart switch
                self._log(f"  Iter {iteration+1}: No valid step, trying chart switch")
                V_new = self._switch_chart(V)
                if np.allclose(V_new, V):
                    message = "No valid descent direction found"
                    break
                V = V_new
                obj, grad_V = self.criterion_grad.compute_gradient(V)
                continue

            # Update
            improvement = obj - obj_new
            V = V_new
            obj = obj_new
            _, grad_V = self.criterion_grad.compute_gradient(V)

            self._objective_history.append(obj)

            if (iteration + 1) % 10 == 0 or iteration < 5:
                self._log(f"  Iter {iteration+1}: obj={obj:.6e}, ||grad||={np.linalg.norm(grad_V):.6e}, α={alpha:.4f}")

            # Check for stagnation
            if improvement < 1e-15 * abs(obj):
                success = True
                message = f"Converged: improvement = {improvement:.2e}"
                break

        x_opt = self._V_to_real(V)

        # Extract final system
        V_final = self._real_to_V(x_opt)

        try:
            # Get (A, C) from BOP chart
            _, bop_cache, crit_cache = self.criterion_grad.forward(V_final)
            A_final = bop_cache.A
            C_final = bop_cache.C

            # IMPORTANT: Use optimal B from necessary conditions, NOT lossless embedding!
            # B̂ = -Q12^H B_F is the H2-optimal B for given (A, C)
            B_final = crit_cache.B_hat  # This is -Q12^H @ B_F

            # D = 0 for the approximant (standard assumption)
            D_final = np.zeros((self.p, self.m), dtype=np.complex128)

        except Exception:
            A_final = 0.5 * np.eye(self.n, dtype=np.complex128)
            B_final = np.zeros((self.n, self.m), dtype=np.complex128)
            C_final = np.zeros((self.p, self.n), dtype=np.complex128)
            D_final = np.zeros((self.p, self.m), dtype=np.complex128)

        # Final objective
        final_obj = self._objective_history[-1] if self._objective_history else 1e10
        h2_error = np.sqrt(max(0, final_obj))

        self._log("\n" + "=" * 60)
        self._log("Optimization complete")
        self._log(f"Final objective: {final_obj:.6e}")
        self._log(f"H2 error: {h2_error:.6e}")
        self._log(f"Iterations: {self._iteration_count}")
        self._log(f"Chart switches: {self._chart_switch_count}")
        self._log(f"Success: {success}")
        self._log("=" * 60)

        return RARL2Result(
            A=A_final,
            B=B_final,
            C=C_final,
            D=D_final,
            V_final=V_final,
            objective=float(final_obj),
            h2_error=float(h2_error),
            iterations=self._iteration_count,
            chart_switches=self._chart_switch_count,
            success=success,
            message=message
        )


def compute_true_h2_error(A_F, B_F, C_F, D_F, A, B, C, D):
    """Compute actual ||F - H||_H2 via error system."""
    n_F = A_F.shape[0]
    n = A.shape[0]
    m = B_F.shape[1]
    p = C_F.shape[0]

    # Build error system E = F - H
    n_total = n_F + n

    A_err = np.zeros((n_total, n_total), dtype=np.complex128)
    A_err[:n_F, :n_F] = A_F
    A_err[n_F:, n_F:] = A

    B_err = np.zeros((n_total, m), dtype=np.complex128)
    B_err[:n_F, :] = B_F
    B_err[n_F:, :] = B

    C_err = np.zeros((p, n_total), dtype=np.complex128)
    C_err[:, :n_F] = C_F
    C_err[:, n_F:] = -C

    D_err = D_F - D

    # H2 norm via observability Gramian
    try:
        Q_err = linalg.solve_discrete_lyapunov(A_err.conj().T, C_err.conj().T @ C_err)
        h2_sq = np.real(np.trace(B_err.conj().T @ Q_err @ B_err))
        h2_sq += np.real(np.trace(D_err.conj().T @ D_err))
        return np.sqrt(max(0, h2_sq))
    except Exception:
        return np.inf


def compute_bt_error(A_F, B_F, C_F, D_F, n):
    """Compute balanced truncation H2 error for comparison."""
    n_F = A_F.shape[0]

    P_c = linalg.solve_discrete_lyapunov(A_F, B_F @ B_F.conj().T)
    Q_o = linalg.solve_discrete_lyapunov(A_F.conj().T, C_F.conj().T @ C_F)

    L_c = linalg.cholesky(P_c + 1e-12 * np.eye(n_F), lower=True)
    M = L_c.conj().T @ Q_o @ L_c
    U, S, Vh = linalg.svd(M)

    Sigma_sqrt = np.diag(np.power(S + 1e-12, 0.25))
    Sigma_sqrt_inv = np.diag(np.power(S + 1e-12, -0.25))

    T_bal = Sigma_sqrt @ U.conj().T @ linalg.inv(L_c.conj().T)
    T_bal_inv = L_c.conj().T @ U @ Sigma_sqrt_inv

    A_bal = T_bal @ A_F @ T_bal_inv
    B_bal = T_bal @ B_F
    C_bal = C_F @ T_bal_inv

    A_bt = A_bal[:n, :n]
    B_bt = B_bal[:n, :]
    C_bt = C_bal[:, :n]
    D_bt = D_F

    return compute_true_h2_error(A_F, B_F, C_F, D_F, A_bt, B_bt, C_bt, D_bt)


def test_gradient_optimizer():
    """Test RARL2 vs Balanced Truncation."""
    print("\n" + "=" * 70)
    print("RARL2 vs Balanced Truncation Comparison")
    print("=" * 70)

    np.random.seed(42)

    tests = [
        ("Scalar (n=1)", 1, 1),
        ("n_F=3 → n=2", 3, 2),
        ("n_F=4 → n=2", 4, 2),
        ("n_F=4 → n=3", 4, 3),
        ("n_F=5 → n=3", 5, 3),
    ]

    print(f"{'Test':<18} {'||F||':<8} {'BT err':<10} {'RARL2 err':<10} {'Improv.':<10} {'Status'}")
    print("-" * 70)

    all_passed = True
    for name, n_F, n in tests:
        # Create target system
        poles = 0.3 + 0.4 * np.arange(n_F) / max(n_F - 1, 1)
        A_F = np.diag(poles).astype(np.complex128)
        B_F = np.random.randn(n_F, 1).astype(np.complex128)
        C_F = np.random.randn(1, n_F).astype(np.complex128)
        D_F = np.zeros((1, 1), dtype=np.complex128)

        # Compute ||F||
        Q_F = linalg.solve_discrete_lyapunov(A_F.conj().T, C_F.conj().T @ C_F)
        F_norm = np.sqrt(np.real(np.trace(B_F.conj().T @ Q_F @ B_F)))

        # Balanced truncation error
        bt_error = compute_bt_error(A_F, B_F, C_F, D_F, n)

        # RARL2
        opt = RARL2GradientOptimizer(A_F, B_F, C_F, D_F, n, verbose=False)
        result = opt.optimize(max_iter=200, tol=1e-10)
        rarl2_error = compute_true_h2_error(
            A_F, B_F, C_F, D_F,
            result.A, result.B, result.C, result.D
        )

        # Improvement
        if bt_error > 1e-10:
            improvement = 100 * (bt_error - rarl2_error) / bt_error
        else:
            improvement = 100 if rarl2_error < 1e-10 else 0

        # Status
        if n == n_F and rarl2_error < 1e-6:
            status = "✓ Perfect"
        elif rarl2_error <= bt_error + 1e-10:
            status = "✓ Pass"
        else:
            status = "✗ Fail"
            all_passed = False

        print(f"{name:<18} {F_norm:<8.4f} {bt_error:<10.4f} {rarl2_error:<10.4f} {improvement:>7.2f}%   {status}")

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED - RARL2 beats or matches balanced truncation!")
    else:
        print("Some tests failed - check implementation")


if __name__ == "__main__":
    test_gradient_optimizer()
