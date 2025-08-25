#!/usr/bin/env python3
"""
Chart-based parametrization (no chart switching)
===============================================
Implements a thin wrapper around the BOP (Balanced Output Pairs)
parametrization to expose the interfaces outlined in IMPLEMENTATION_PLAN.md.

Assumptions:
- Chart switching is not required (single valid chart throughout).
- Numpy/Scipy implementation; can be ported to Torch later for AD.

Key methods:
- set_chart_center(omega): sets Ω = (W, X, Y, Z)
- parametrize((A,B,C,D)) -> (V, D0)
- deparametrize(V, D0) -> (A,B,C,D)
- check_chart_boundary() -> bool (checks P = Λ^H Λ > 0)

Internally uses bop.BOP for the forward/backward maps.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy import linalg

from bop import BOP


class ChartParametrization:
    def __init__(self, n: int, p: int, m: int):
        self.n = n
        self.p = p
        self.m = m
        self._bop: BOP | None = None
        self._last_P: np.ndarray | None = None

    def set_chart_center(self, omega: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        W, X, Y, Z = omega
        if W.shape != (self.n, self.n):
            raise ValueError("W has wrong shape")
        if X.shape != (self.n, self.m) or Y.shape != (self.p, self.n) or Z.shape != (self.p, self.m):
            # For square case p == m; test_bop uses m outputs. Keep p==m convention here.
            # To stay consistent with bop.BOP tests, set p = m locally if needed.
            if self.p != self.m and Y.shape[0] == self.m and Z.shape == (self.m, self.m):
                self.p = self.m
            else:
                raise ValueError("Chart center block shapes don't match n,p,m")
        self._bop = BOP(W, X, Y, Z)

    def _ensure_bop(self):
        if self._bop is None:
            raise RuntimeError("Chart center not set. Call set_chart_center first.")

    def solve_stein_for_chart(self, A: np.ndarray, W: np.ndarray, C: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Solve Λ - A Λ W = C Y via fixed-point or vectorization.
        Returns Λ.
        """
        # vec(Λ - A Λ W) = vec(CY) => (I - W^T ⊗ A) vec(Λ) = vec(CY)
        n = A.shape[0]
        I = np.eye(n * n, dtype=complex)
        K = I - np.kron(W.T, A)
        b = (C @ Y).reshape(-1, order="F")
        vecL, *_ = np.linalg.lstsq(K, b, rcond=None)
        return vecL.reshape((n, n), order="F")

    def deparametrize(self, V: np.ndarray, D0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_bop()
        A, B, C, D = self._bop.V_to_realization(V, D0)
        # Cache P for boundary check: solve W^H P W - P = V^H V - Y^H Y
        RHS = V.conj().T @ V - self._bop.Y.conj().T @ self._bop.Y
        P = linalg.solve_discrete_lyapunov(self._bop.W.conj().T, -RHS)
        P = (P + P.conj().T) / 2
        self._last_P = P
        return A, B, C, D

    def parametrize(self, realization: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_bop()
        A, B, C, D = realization
        V, D0 = self._bop.realization_to_V(A, B, C, D)
        # Update cached P corresponding to V
        RHS = V.conj().T @ V - self._bop.Y.conj().T @ self._bop.Y
        P = linalg.solve_discrete_lyapunov(self._bop.W.conj().T, -RHS)
        P = (P + P.conj().T) / 2
        self._last_P = P
        return V, D0

    def check_chart_boundary(self, eps: float = 1e-6) -> bool:
        if self._last_P is None:
            return True
        lam_min = np.min(np.linalg.eigvalsh(self._last_P))
        return bool(lam_min > eps)

