# Latent Design Labs

A Python package for experimental design and optimization using various criteria and models.

## Features

- **Basis Functions**: Polynomial, B-spline, and Fourier bases
- **Optimality Criteria**: A, D, G, and I optimality for experimental design
- **Statistical Models**: FOF, FOS, SOF, and SOS models
- **Optimization**: NBDO (Nested Bayesian Design Optimization) algorithm
- **Diagnostics**: Tools for analyzing and visualizing experimental designs
- **Inner Products**: J-matrix and penalty-based inner product computations

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/latent_design_labs.git
cd latent_design_labs

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e .
```

### Using pip

```bash
# Install in development mode from local source
pip install -e .

# Or install from PyPI (when published)
pip install latent-design
```

## Quick Start

```python
from latent_design.bases import PolynomialBasis
from latent_design.criteria import DOptimality
from latent_design.models import FOF

# Create a polynomial basis
basis = PolynomialBasis(degree=3)

# Create a D-optimality criterion
criterion = DOptimality()

# Create a FOF model
model = FOF(basis=basis, criterion=criterion)

# Your experimental design code here...
```

## CLI Quick Start

```bash
# Single run from a packaged config (resolved by name)
nbdo run -c fof_paper.json -o tag=demo

# Batch: execute multiple commands from a packaged file
nbdo batch -f cmds_paper.txt
nbdo --help
```

## Project Structure

```
src/latent_design/
├── bases/          # Basis functions (polynomial, B-spline, Fourier)
├── criteria/       # Optimality criteria (A, D, G, I optimality)
├── diagnostics/    # Diagnostic tools and visualization
├── inner_products/ # Inner product computations
├── models/         # Statistical models (FOF, FOS, SOF, SOS)
├── optimizers/     # Optimization algorithms (NBDO)
└── utils/          # Utility functions and backend management
```

## Dependencies

- **Core**: numpy, scipy, scikit-learn
- **Optimization**: scikit-optimize, tensorflow
- **Visualization**: matplotlib, rich
- **CLI**: typer, tqdm

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
