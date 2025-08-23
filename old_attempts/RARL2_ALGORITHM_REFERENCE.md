# RARL2 Algorithm Reference

## 1. Overview and Objective

RARL2 (Rational Approximation in the Real Lossless bounded-real Lemma) finds the **optimal H2 approximation** of a given transfer function F among ALL stable systems of prescribed McMillan degree n. Unlike balanced truncation which provides a suboptimal approximation by simply truncating states, RARL2 searches over the entire manifold of stable systems of degree n to find the globally optimal H2 approximation.

**Concentrated Criterion Formula** (Equation 6):
```
ψₙ(G) = ||F - G·P_H2(G♯·F)||²
```
where G is a p×p lossless matrix of degree n, G♯ = G⁻¹, and P_H2 is the projection onto Hardy space H2.

**Why it's better than balanced truncation**: Balanced truncation provides a good but suboptimal H2 approximation. RARL2 uses balanced truncation as initialization, then iteratively improves it to find the true H2-optimal approximation of the given degree.

## 2. Mathematical Foundation

### 2.1 Key Definitions

**Lossless/Inner Functions**: A rational matrix G(z) is lossless if G(z)·G*(z) = I for all |z| = 1. In state-space, the realization matrix [A B; C D] is unitary.

**Output-Normal Pairs**: An observable pair (C,A) is output-normal if A*·A + C*·C = I. This implies the observability Gramian Q = I.

**Hardy Space H2**: The space of functions analytic in |z| > 1 (discrete-time). The projection π₊ = P_H2 extracts the stable/causal part of a function.

### 2.2 The Core Optimization Problem

**Direct Formulation**: 
```
min ||F - H||²₂ over all stable H with McMillan degree n
```

**Concentrated Formulation**: 
```
min ψₙ(G) = ||F - G·P||²₂ where P = P_H2(G♯·F)
```

**Relationship**: Every stable H can be factored as H = G·P where G is lossless (inner) and P is outer. The concentrated formulation eliminates P from the optimization by computing it optimally for each G.

## 3. The Parametrization (Most Critical Part)

### 3.1 Manifold Structure

The set L^p_n of p×p lossless functions of degree n forms a manifold of dimension **2np**. Direct parametrization is difficult, so we use an atlas of charts.

### 3.2 Chart-Based Parametrization

**The map φ_Ω: (A,B,C,D) → (V,D₀)** where Ω = (W,X,Y,Z) is the chart center:

1. **Solve Stein equation** (Equation 12):
   ```
   Λ - A·Λ·W = C·Y
   ```

2. **Compute V** (Equation 13):
   ```
   V = D·Y + B·Λ·W
   ```

3. **Check chart domain**: P = Λ*·Λ must be positive definite

4. **Normalize** (Equation 17):
   ```
   Ỹ = Y·Λ⁻¹
   W̃ = Λ·W·Λ⁻¹  
   Ṽ = V·Λ⁻¹
   ```

5. **Form K matrix** (Equation 18):
   ```
   K = W̃*·W̃ + Ỹ*·Ỹ = Ṽ*·Ṽ + I
   ```

6. **Unitary completions** (Equation 20):
   ```
   U = [Ỹ    -W̃*  ]
       [W̃     Ỹ*   ]
   
   V = [(I + Ṽ*·Ṽ)⁻¹/²    -Ṽ*              ]
       [Ṽ·(I + Ṽ*·Ṽ)⁻¹/²  (I + Ṽ·Ṽ*)⁻¹/² ]
   ```

7. **Extract parameters**: V ∈ ℝ^(2np), D₀ ∈ U_p

**Inverse map φ_Ω⁻¹(V,D₀)**:
```
[A B] = U·diag(I_n, D₀)·V*
[C D]
```

### 3.3 Chart Domain and Switching

- **Chart domain D_Ω**: All realizations where P = Λ*·Λ > 0
- **Boundary**: When smallest eigenvalue of P approaches 0
- **Switching**: Choose new chart center Ω' = current realization, recompute parametrization

## 4. Lossless Embedding

### 4.1 Proposition 1 Formula

Given observable pair (C,A) with observability Gramian Q (solving A*·Q·A - Q = -C*·C):

```
B = -(A - νI)·Q⁻¹·(I - ν·A*)⁻¹·C*
D = I - C·Q⁻¹·(I - ν·A*)⁻¹·C*
```
where ν is a point on unit circle (typically ν = 1).

### 4.2 Special Case: Output-Normal

When (C,A) is output-normal, Q = I, simplifying to:
```
B = -(A - νI)·(I - ν·A*)⁻¹·C*
D = I - C·(I - ν·A*)⁻¹·C*
```

## 5. Gradient Computation

### 5.1 Explicit Gradient Formula

From equation (11) in paper:
```
dJₙ/dλ = 2·Re·Tr(P₁₂*·[A*·Q₁₂·(∂A/∂λ) + C*·(∂C/∂λ)])
```

Where:
- Q₁₂ solves: A_F*·Q₁₂·A + C_F*·C = Q₁₂ (Stein equation)
- P₁₂ solves: A·P₁₂·A_F* + B̂·B_F* = P₁₂ (Sylvester equation)

### 5.2 Necessary Conditions

**First condition** (Equation 8):
```
B̂ = -Q₁₂*·B_F
```
where Q₁₂ is the cross-Gramian between approximation and target.

**Second condition** (Equation 9): P₁₂ Sylvester equation as above.

## 6. Algorithm Implementation

### 6.1 Initialization

1. Start with high-order model from completion/AAK
2. Apply balanced truncation to get order n approximation
3. Extract output-normal pair (C,A) from balanced realization
4. Set initial chart center Ω = balanced realization
5. Compute initial parameters via φ_Ω

### 6.2 Main Optimization Loop

```python
while not converged:
    1. Parameters (V,D₀) → realization via φ_Ω⁻¹
    2. Extract (C,A) from realization
    3. Compute B̂ via necessary condition (Q₁₂ Stein equation)
    4. Set D̂ = 0 (for simplicity)
    5. Evaluate objective ||F - H||² where H = (A,B̂,C,D̂)
    6. Compute gradient via equation (11)
    7. Update parameters: V_new = V - α·∇V
    8. Check if P = Λ*·Λ still positive definite
    9. If near boundary (min eigenvalue of P < ε):
        - Switch chart: new center Ω' = current realization
        - Reparametrize in new chart
```

### 6.3 Practical Considerations

- **Numerical stability**: Use QR/SVD for unitary completions
- **Convergence criteria**: ||gradient|| < tol or relative improvement < tol
- **Step size**: Adaptive with backtracking line search
- **Chart switching threshold**: min(eig(P)) < 1e-6
- **Typical values**: 50-200 iterations, tolerance 1e-8

## 7. Key Implementation Pitfalls to Avoid

1. **Don't compute P explicitly** - it would increase the model order. P is implicitly represented through optimal (B̂,D̂).

2. **Don't use naive parametrization** - Simple QR of [V;I] is NOT sufficient. Must use full chart-based parametrization.

3. **Don't forget chart switching** - Algorithm will fail at chart boundaries without switching.

4. **Set D = 0 for simplicity** - Both target and approximation can have D = 0 without loss of generality.

5. **Don't confuse the two B matrices**:
   - B from lossless embedding (Proposition 1)
   - B̂ from optimal approximation (necessary condition)

## 8. Test Cases

### 8.1 Scalar Case Verification

Target: 1st order system with pole at 0.6
```python
A_F = [[0.6]], B_F = [[0.8]], C_F = [[0.8]], D_F = [[0]]
```

Expected: 1st order approximation should match exactly (zero error) since degrees are equal.

### 8.2 Comparison with Balanced Truncation

1. Compute balanced truncation of order n
2. Evaluate its H2 error: e_BT
3. Run RARL2 initialized from balanced truncation
4. Final error e_RARL2 should satisfy: e_RARL2 ≤ e_BT
5. Typical improvement: 10-30% reduction in H2 error

### 8.3 Validation Checks

- Output-normal constraint: ||A*·A + C*·C - I|| < 1e-12
- Stability: max|eig(A)| < 1
- Chart validity: min(eig(Λ*·Λ)) > 0
- Gradient descent: objective decreasing (with line search)

## 9. Summary of Key Equations

- **Concentrated criterion**: ψₙ(G) = ||F - G·P_H2(G♯·F)||² (Eq. 6)
- **Stein equation for chart**: Λ - A·Λ·W = C·Y (Eq. 12)
- **Gradient formula**: dJₙ/dλ = 2·Re·Tr(P₁₂*·[A*·Q₁₂·∂A/∂λ + C*·∂C/∂λ]) (Eq. 11)
- **Necessary condition**: B̂ = -Q₁₂*·B_F (Eq. 8)
- **Lossless embedding**: B = -(A-νI)·Q⁻¹·(I-νA*)⁻¹·C* (Prop. 1)