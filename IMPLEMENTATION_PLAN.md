# RARL2 Implementation Plan

## Overview
Complete rewrite of RARL2 with correct chart-based parametrization following AUTO_MOSfinal.pdf.

## Architecture Decision: Automatic Differentiation Strategy
**Use PyTorch with Implicit Differentiation for Stein equations**
- Chart transformations: PyTorch autograd
- Stein solvers: Implicit differentiation (efficient forward, exact backward)
- This balances efficiency and correctness

## Module Structure

### 1. `chart_parametrization.py` (HIGHEST PRIORITY)
**Purpose**: Implement the manifold chart-based parametrization
**Key Functions**:
- `solve_stein_for_chart()`: Solve Λ - A·Λ·W = C·Y using implicit diff
- `compute_unitary_completions()`: Build U and V matrices (eq. 20)
- `parametrize()`: Map (A,B,C,D) → (V, D₀)
- `deparametrize()`: Map (V, D₀) → (A,B,C,D)
- `check_chart_boundary()`: Detect when to switch charts

**Critical Implementation Notes**:
- This replaces the flawed QR parametrization
- Must handle complex matrices properly
- Chart switching is essential for convergence
- Use implicit differentiation for Stein equation
- Test with scalar case first

**Delegation**: Can be implemented independently once interfaces are clear

### 2. `gradient_friendly_stein.py` ✅ (COMPLETED)
**Purpose**: Gradient-compatible Stein solvers
**Status**: DONE - Three approaches implemented:
- Implicit differentiation (recommended)
- Unrolled iteration
- Custom backward

### 3. `lossless_embedding.py` (EXISTS - NEEDS REVIEW)
**Purpose**: Create lossless systems from output-normal pairs
**Status**: Already implemented but needs verification
**Actions**:
- Review existing implementation
- Ensure it handles Q = I (output-normal) case
- Convert to PyTorch for AD compatibility
- Add validation tests

**Delegation**: Quick review and PyTorch conversion

### 4. `cross_gramians.py` (NEW)
**Purpose**: Compute cross-Gramians for necessary conditions
**Key Functions**:
- `compute_Q12()`: Solve A_F^H·Q₁₂·A + C_F^H·C = Q₁₂
- `compute_optimal_B()`: B̂ = -Q₁₂^H·B_F
- `compute_P12()`: Solve A·P₁₂·A_F^H + B̂·B_F^H = P₁₂

**Implementation Notes**:
- Use implicit differentiation from gradient_friendly_stein.py
- These determine the optimal approximation

**Delegation**: Straightforward given Stein solver

### 5. `balanced_truncation.py`
**Purpose**: Initialize RARL2 with balanced truncation
**Key Functions**:
- `compute_gramians()`: Controllability and observability
- `balance_realization()`: Balance the Gramians
- `truncate()`: Reduce to order n
- `to_output_normal()`: Convert to output-normal form

**Implementation Notes**:
- Can use scipy for standard computations
- Must produce output-normal form for RARL2
- No gradient needed (just initialization)

**Delegation**: Standard algorithm, independent module

### 6. `rarl2_optimizer.py`
**Purpose**: Main optimization loop with PyTorch
**Key Components**:
```python
class RARL2Optimizer(nn.Module):
    def __init__(self, n, p, m, target):
        # Initialize chart parametrization
        # Set up parameters V (real-valued)
        
    def forward(self):
        # V → (A,B,C,D) via chart
        # Compute B̂ via cross-Gramian
        # Return loss ||F - H||²
        
    def switch_chart(self):
        # Detect boundary
        # Reparametrize
```

**Implementation Notes**:
- Orchestrates all components
- Uses PyTorch optimizers (L-BFGS recommended)
- Handles chart switching logic

**Delegation**: Integrate all modules

### 7. `h2_norm.py`
**Purpose**: H2 norm computations in PyTorch
**Key Functions**:
- `compute_h2_norm()`: Single system norm
- `compute_h2_error()`: ||H1 - H2||² via frequency sampling
- `frequency_response()`: Evaluate H(e^{iω})

**Implementation Notes**:
- Must be differentiable (PyTorch tensors)
- Use frequency sampling for discrete-time

**Delegation**: Independent module

### 8. `test_suite.py`
**Purpose**: Comprehensive testing
**Test Cases**:
1. **Scalar test**: 1st order → 1st order (exact match expected)
2. **Gradient check**: Compare with finite differences
3. **Chart switching**: Force boundary crossing
4. **Stein solver convergence**: Various (A,W) pairs
5. **Balanced truncation baseline**: Verify improvement

**Delegation**: Can be developed in parallel with modules

## Implementation Order & Delegation

### Phase 1: Core Components (Week 1)
| Module | Owner | Status | Dependencies |
|--------|-------|--------|--------------|
| gradient_friendly_stein.py | - | ✅ DONE | None |
| chart_parametrization.py | Developer 1 | 🔧 TODO | Stein solver |
| lossless_embedding.py | Developer 2 | 🔧 REVIEW | None |
| cross_gramians.py | Developer 3 | 🔧 TODO | Stein solver |

### Phase 2: Integration (Week 2)
| Module | Owner | Status | Dependencies |
|--------|-------|--------|--------------|
| balanced_truncation.py | Developer 4 | 🔧 TODO | None |
| h2_norm.py | Developer 5 | 🔧 TODO | None |
| rarl2_optimizer.py | Lead | 🔧 TODO | All above |

### Phase 3: Testing (Week 3)
| Module | Owner | Status | Dependencies |
|--------|-------|--------|--------------|
| test_suite.py | All | 🔧 TODO | Complete system |
| Performance optimization | Lead | 🔧 TODO | Working system |

## Module Interfaces (Contract)

### Chart Parametrization
```python
class ChartParametrization(nn.Module):
    def set_chart_center(self, omega: Tuple[Tensor, ...])
    def deparametrize(self, V: Tensor) -> Tuple[Tensor, ...]
    def check_boundary(self) -> bool
```

### Cross-Gramians
```python
def compute_cross_gramian(A, A_F, C, C_F) -> Q12
def compute_optimal_B(Q12, B_F) -> B_hat
```

### H2 Norm
```python
def compute_h2_error(sys1, sys2) -> Tensor (scalar)
```

## Success Criteria

1. **Scalar Test**: Zero error for equal-order approximation
2. **Gradient Test**: Relative error < 1e-6 vs finite differences
3. **Improvement**: 10-30% better than balanced truncation
4. **Convergence**: < 100 iterations for typical problems
5. **Stability**: No NaN/Inf in 1000 random tests

## Critical Implementation Rules

### DO:
✅ Use implicit differentiation for Stein equations
✅ Test gradients against finite differences
✅ Set D = 0 for simplicity
✅ Use output-normal form (Q = I)
✅ Handle complex arithmetic correctly (.conj().T not .T)

### DON'T:
❌ Use naive QR parametrization
❌ Compute outer factor P explicitly
❌ Confuse lossless B with optimal B̂
❌ Forget chart switching
❌ Use scipy directly in forward pass (breaks gradients)

## Review Checklist

Each module must pass:
- [ ] Unit tests with known solutions
- [ ] Gradient check (if applicable)
- [ ] Complex number handling
- [ ] Numerical stability tests
- [ ] Interface compliance
- [ ] Documentation with examples

## References

- `RARL2_ALGORITHM_REFERENCE.md`: Mathematical details
- `gradient_friendly_stein.py`: Stein solver implementation
- `rarl2_architecture_with_ad.py`: Interface specifications
- AUTO_MOSfinal.pdf: Original paper