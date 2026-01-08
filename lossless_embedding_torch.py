#!/usr/bin/env python3
"""
Lossless Embedding in PyTorch (differentiable)
===============================================
Implements Proposition 1 from AUTO_MOSfinal.pdf for completing
an output-normal pair (C,A) to a lossless system (A,B,C,D).
"""

import torch
from typing import Tuple


def lossless_embedding_torch(C: torch.Tensor, A: torch.Tensor, nu: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complete output-normal (C,A) to lossless (A,B,C,D).

    Given observable pair (C,A) with observability Gramian Q = I (output normal),
    compute (B,D) such that G = [A B; C D] is lossless (allpass/inner).

    Formula (Proposition 1, simplified for Q = I):
        B = -(A - νI)·(I - ν·A^H)^(-1)·C^H
        D = I - C·(I - ν·A^H)^(-1)·C^H

    where ν is a point on the unit circle (typically ν = 1).

    Args:
        C: Output matrix (p × n)
        A: System matrix (n × n)
        nu: Point on unit circle for embedding (default: 1.0)

    Returns:
        B: Input matrix (n × m) where m = p
        D: Feedthrough matrix (p × m)
    """
    n = A.shape[0]
    p = C.shape[0]

    # Create identity matrices
    I_n = torch.eye(n, dtype=A.dtype, device=A.device)
    I_p = torch.eye(p, dtype=A.dtype, device=A.device)

    # Convert nu to tensor
    nu_tensor = torch.tensor(nu, dtype=A.dtype, device=A.device)

    # Compute (I - ν·A^H)^(-1)
    inv_term = torch.linalg.inv(I_n - nu_tensor * A.conj().T)

    # Compute B
    B = -(A - nu_tensor * I_n) @ inv_term @ C.conj().T

    # Compute D
    D = I_p - C @ inv_term @ C.conj().T

    return B, D


def verify_lossless_torch(A: torch.Tensor, B: torch.Tensor,
                         C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Verify losslessness: G^H·G = I

    For discrete-time lossless system:
        [A B]^H  [A B]   [I 0]
        [C D]    [C D] = [0 I]

    Returns:
        Maximum entry-wise error
    """
    # Build realization matrix
    n = A.shape[0]
    p = C.shape[0]
    m = B.shape[1]

    G = torch.cat([
        torch.cat([A, B], dim=1),
        torch.cat([C, D], dim=1)
    ], dim=0)

    # Should be unitary
    G_H_G = G.conj().T @ G
    I_total = torch.eye(n + m, dtype=G.dtype, device=G.device)

    error = torch.max(torch.abs(G_H_G - I_total))
    return error
