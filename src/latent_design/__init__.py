"""
Latent Design Labs - A Python package for experimental design and optimization.

This package provides tools for creating and optimizing experimental designs
using various criteria and models.
"""

__version__ = "0.1.0"
__author__ = "tedoiler"
__email__ = "ladas.thodoris@gmail.com"

# Import main modules to make them available at package level
# We'll import them individually to avoid circular import issues
try:
    from . import bases
    from . import criteria
    from . import diagnostics
    from . import inner_products
    from . import models
    from . import optimizers
    from . import utils
except ImportError as e:
    # If there are import issues, we'll still allow the package to be imported
    # but some modules might not be available
    pass

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Main modules
    "bases",
    "criteria", 
    "diagnostics",
    "inner_products",
    "models",
    "optimizers",
    "utils",
]
