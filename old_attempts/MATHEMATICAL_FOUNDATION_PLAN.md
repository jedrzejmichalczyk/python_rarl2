# RARL2 Mathematical Foundation Plan

## Core Problem Statement

Find the **optimal H2 approximation** of a given system F by a lower-order system H.

### Two Formulations:
1. **Direct**: minimize ||F - H||²₂ over all stable H of order n
2. **Concentrated**: minimize ψₙ(G) = ||F - π₊(FG†)G||²₂ over lossless G

The key insight: Every stable H can be written as H = G*P where:
- G is lossless (inner factor)
- P is outer (minimum phase)

## Critical Questions We Need to Answer

### 1. Are we computing the right objective?
Currently we compute: ||F - H||² where H has optimal (B,D) for fixed (C,A)

But the paper says: 
- First create lossless G from (C,A)
- Then find P = π₊(G†*F) via H2 projection
- Final H = G*P

**Key issue**: We might be conflating two different problems!

### 2. What exactly is the H2 projection π₊?
- Projects onto stable, causal systems
- In frequency domain: removes unstable poles
- In time domain: makes impulse response causal

**Mathematical formula needed**: How exactly do we compute π₊(G†*F)?

### 3. Inner-outer factorization confusion
The code mentions "Douglas-Shapiro" but this is actually inner-outer factorization:
- Any H = G*P where G is inner, P is outer
- NOT the same as lossless embedding!

## Two Types of Targets Needed

### Type 1: State-Space Realization
```
Target: F(z) = D_F + C_F(zI - A_F)⁻¹B_F
Given: (A_F, B_F, C_F, D_F)
```

### Type 2: Frequency Domain (Fourier Coefficients)
```
Target: F(e^{iω}) = Σ f_k e^{ikω}
Given: Fourier coefficients {f_k}
```

## Mathematical Transformations to Explore

### 1. Balanced Truncation Connection
The paper mentions using balanced truncation as initialization. Why?
- Balanced truncation gives suboptimal H2 approximation
- RARL2 should improve upon it
- The balanced realization has special properties

### 2. Cayley Transform
Maps between discrete and continuous time:
```
z = (1+s)/(1-s)
```
Maybe the problem is simpler in continuous time?

### 3. Coprime Factorization
Alternative to inner-outer:
```
H = N*M⁻¹ where N,M are coprime
```

### 4. Hankel Operator Theory
The Hankel singular values determine approximation quality:
- AAK optimizes Hankel norm (different from H2!)
- Connection to balanced truncation
- Might provide bounds or initialization

## Proposed Implementation Plan

### Phase 1: Verify Mathematical Formulation
1. **Test with known solutions**:
   - Simple 1st order system → scalar approximation
   - Verify we get expected H2 error

2. **Compare three approaches**:
   - Direct optimization over (A,B,C,D)
   - Concentrated criterion over lossless G
   - Balanced truncation baseline

### Phase 2: Proper Inner-Outer Implementation
1. **Implement true inner-outer factorization**:
   ```python
   def inner_outer_factorization(H):
       # H = G*P where G is inner, P is outer
       return G, P
   ```

2. **Implement H2 projection correctly**:
   ```python
   def h2_projection(H):
       # Remove unstable parts
       # NOT the same as our current implementation!
       return H_stable
   ```

3. **Verify the concentrated criterion**:
   ```python
   def concentrated_criterion(G, F):
       # ψₙ(G) = ||F - π₊(F*G†)*G||²
       P = h2_projection(adjoint(G) @ F)
       H = G @ P
       return h2_norm(F - H)
   ```

### Phase 3: Handle Different Target Types
1. **Frequency domain targets**:
   - Given Fourier coefficients
   - Evaluate at frequency points
   - Use FFT for efficiency

2. **Time domain targets**:
   - Given impulse response
   - Markov parameters

### Phase 4: Analytical Test Cases
Create test cases with known optimal solutions:

1. **Scalar case** (n=1, n_F=2):
   - Can solve analytically
   - Verify our implementation matches

2. **Diagonal systems**:
   - Decouples into scalar problems
   - Easy to verify

3. **FIR to IIR**:
   - Finite impulse response target
   - Known optimal IIR approximations

## Key Mathematical Questions to Resolve

1. **What exactly is B̂ in the necessary conditions?**
   - Is it the B from lossless embedding?
   - Or optimal B for lossy approximation?
   - The paper seems to suggest it's neither!

2. **How does P relate to B and D?**
   - In H = G*P, how do we extract (B,D) from P?
   - Is P itself a realization or transfer function?

3. **Gradient computation**:
   - Should we differentiate through the H2 projection?
   - Or use implicit differentiation on optimality conditions?
   - Or is there a closed-form gradient?

## Simplifying Transformations to Try

### 1. Orthogonal Transformation
If we transform to a basis where the problem is diagonal:
```
F_diag = U*F*V
H_diag = U*H*V
```
Then the problem decouples?

### 2. Frequency Domain Formulation
Instead of state-space, work entirely in frequency:
```
minimize ∫|F(e^{iω}) - H(e^{iω})|² dω
```
Parametrize H directly by poles/zeros?

### 3. Gramian-Based Formulation
The H2 norm relates to controllability/observability Gramians:
```
||H||²₂ = trace(C*P*C^T) = trace(B^T*Q*B)
```
Maybe optimize over Gramians directly?

## Next Steps

1. **Implement clean test framework** with known solutions
2. **Verify our objective function** matches the paper
3. **Implement proper inner-outer factorization**
4. **Test both frequency and state-space targets**
5. **Compare with balanced truncation baseline**

The key is to ensure mathematical correctness before optimizing!