# CLAUDE.md - AI Assistant Guidelines for RARL2

This file provides guidance to Claude (claude.ai) when working with the RARL2 codebase.

## Project Overview

RARL2 is a Python implementation of the Rational Approximation in the Real Lossless bounded-real Lemma algorithm for optimal H2 model reduction. It finds the globally optimal approximation of a given system among ALL stable systems of a prescribed McMillan degree.

## Critical Concepts

### 1. Chart-Based Parametrization (MOST IMPORTANT)
- The manifold of lossless functions requires chart-based parametrization
- **NEVER use naive QR parametrization** - this was the fundamental flaw in early attempts
- Charts must be switched when approaching boundaries (when min(eig(Λ*Λ)) < 1e-6)
- The Stein equation Λ - A·Λ·W = C·Y is NON-STANDARD (not the usual form)

### 2. Gradient Computation
- Use **implicit differentiation** for Stein equations
- Forward pass: Use efficient solvers (scipy, C++)
- Backward pass: Solve adjoint equation for gradients
- This balances efficiency and correctness

### 3. Two Different B Matrices
- **B from lossless embedding**: Via Proposition 1, creates lossless G
- **B̂ from necessary conditions**: Optimal approximation, B̂ = -Q₁₂*·B_F
- These are DIFFERENT - don't confuse them!

### 4. Outer Factor P
- **NEVER compute P explicitly** - it would increase model order
- P is implicitly represented through optimal (B̂, D̂)
- The concentrated criterion eliminates P from optimization

## Implementation Architecture

### Core Modules
1. **chart_parametrization.py**: Manifold chart transformations
2. **gradient_friendly_stein.py**: Stein solvers with automatic differentiation
3. **cross_gramians.py**: Compute Q₁₂ for necessary conditions
4. **rarl2_optimizer.py**: Main optimization loop with chart switching

### Technology Stack
- **PyTorch**: For automatic differentiation
- **NumPy/SciPy**: For numerical computations
- **Implicit Differentiation**: For Stein equation gradients

## Common Pitfalls to Avoid

1. ❌ **Don't use simple QR([V; I])** for parametrization
2. ❌ **Don't compute outer factor P explicitly**
3. ❌ **Don't forget chart switching at boundaries**
4. ❌ **Don't confuse lossless B with optimal B̂**
5. ❌ **Don't use .T for complex transpose** (use .conj().T)

## Testing Guidelines

### Critical Tests
1. **Scalar case**: 1st order target/approximation should match exactly (zero error)
2. **Gradient check**: Compare with finite differences (relative error < 1e-6)
3. **Chart switching**: Must handle boundary transitions smoothly
4. **Improvement**: Should beat balanced truncation by 10-30%

### Debugging Tips
- If optimization diverges: Check chart validity (Λ*Λ > 0)
- If gradients are wrong: Verify Stein solver convergence
- If no improvement: Check initialization from balanced truncation
- If scalar test fails: Chart parametrization is likely wrong

## Mathematical References

Key equations from AUTO_MOSfinal.pdf:
- **Concentrated criterion** (Eq. 6): ψₙ(G) = ||F - G·P_H2(G♯·F)||²
- **Chart Stein equation** (Eq. 12): Λ - A·Λ·W = C·Y
- **Necessary condition** (Eq. 8): B̂ = -Q₁₂*·B_F
- **Gradient formula** (Eq. 11): dJ/dλ = 2·Re·Tr(P₁₂*·[A*·Q₁₂·∂A/∂λ + C*·∂C/∂λ])

## Code Style

- Use type hints for all functions
- Document mathematical formulas in docstrings
- Include equation references from paper
- Test complex arithmetic carefully
- Validate numerical stability

## Performance Considerations

- Start with correctness, optimize later
- Implicit differentiation is key for efficiency
- Chart switching is expensive but necessary
- Cache Stein solutions when possible
- Use L-BFGS optimizer (better than Adam for this problem)

## When Making Changes

1. **Always test the scalar case first** - it should work perfectly
2. **Check gradients** against finite differences
3. **Verify chart validity** throughout optimization
4. **Compare with balanced truncation** as baseline
5. **Document any deviations** from the paper's formulation

## Getting Help

If stuck, refer to:
1. `RARL2_ALGORITHM_REFERENCE.md` - Complete mathematical details
2. `IMPLEMENTATION_PLAN.md` - Development roadmap
3. AUTO_MOSfinal.pdf - Original paper
4. Test files - Working examples and validation

Remember: The chart parametrization is the heart of RARL2. Get that right and everything else follows.