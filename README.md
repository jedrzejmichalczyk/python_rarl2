# RARL2 - Rational Approximation in the Real Lossless Bounded-Real Lemma

A Python implementation of the RARL2 algorithm for optimal H2 model reduction, based on the paper AUTO_MOSfinal.pdf.

## Overview

RARL2 finds the **optimal H2 approximation** of a given system among ALL stable systems of prescribed McMillan degree. Unlike balanced truncation which provides suboptimal approximations, RARL2 searches over the entire manifold of stable systems to find the globally optimal solution.

## Key Features

- **Chart-based manifold parametrization** for lossless functions
- **Gradient-friendly Stein solvers** with implicit differentiation
- **Automatic differentiation** via PyTorch for complex gradient chains
- **Chart switching** for numerical stability
- **Better than balanced truncation** (10-30% improvement typical)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rarl2.git
cd rarl2

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
import rarl2

# Create target system (high-order)
target = (A_F, B_F, C_F, D_F)  # State-space matrices

# Initialize RARL2 optimizer
optimizer = rarl2.RARL2Optimizer(
    order=10,  # Desired approximation order
    target=target
)

# Run optimization
result = optimizer.optimize(max_iter=100)

# Extract optimal approximation
A_opt, B_opt, C_opt, D_opt = result.system

# Check improvement over balanced truncation
print(f"H2 error: {result.error}")
print(f"Improvement: {result.improvement_percent}%")
```

## Project Structure

```
rarl2/
├── src/
│   ├── chart_parametrization.py    # Manifold parametrization
│   ├── gradient_friendly_stein.py  # Stein solvers with AD
│   ├── cross_gramians.py          # Necessary conditions
│   ├── lossless_embedding.py      # Lossless system creation
│   ├── balanced_truncation.py     # Initialization
│   ├── h2_norm.py                 # H2 norm computations
│   └── rarl2_optimizer.py         # Main optimization loop
├── tests/
│   ├── test_scalar_case.py        # Scalar validation
│   ├── test_gradients.py          # Gradient checks
│   └── test_chart_switching.py    # Chart boundary tests
├── docs/
│   ├── RARL2_ALGORITHM_REFERENCE.md
│   └── IMPLEMENTATION_PLAN.md
├── examples/
│   └── filter_approximation.py
└── benchmarks/
    └── compare_with_balanced.py
```

## Mathematical Background

RARL2 solves the concentrated optimization problem:
```
min ψₙ(G) = ||F - G·P_H2(G♯·F)||²
```
where G is a lossless matrix of degree n, optimizing over the manifold of all stable systems.

See [RARL2_ALGORITHM_REFERENCE.md](docs/RARL2_ALGORITHM_REFERENCE.md) for detailed mathematics.

## Development

See [CLAUDE.md](CLAUDE.md) for AI assistant guidelines and [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for development roadmap.

## Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_scalar_case.py

# Check gradients
pytest tests/test_gradients.py -v
```

## Citation

If you use this software, please cite:
```bibtex
@article{rarl2,
  title={RARL2: Rational Approximation in the Real Lossless Bounded-Real Lemma},
  author={...},
  journal={...},
  year={...}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Acknowledgments

Based on the AUTO_MOSfinal.pdf paper and inspired by the MAFIN project's filter synthesis work.