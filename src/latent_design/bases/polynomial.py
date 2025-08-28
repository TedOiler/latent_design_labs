# bases/polynomial.py
import numpy as np
from latent_design.bases.base import Basis

class PolynomialBasis(Basis):
    """
    Monomial polynomial basis on [0, 1].

    Parameters
    ----------
    degree : int
        Highest power included. The basis is {1, t, t^2, ..., t^degree}.
    """
    def __init__(self, degree: int):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = int(degree)
        self._num_basis = self.degree + 1

    # ---- New interface ----
    def num_basis(self) -> int:
        return self._num_basis

    def evaluate(self, i: int, t: float) -> float:
        if i < 0 or i >= self._num_basis:
            raise IndexError(f"basis index {i} out of range [0, {self._num_basis-1}]")
        tt = float(t)
        # domain is [0,1]; allow slight rounding outside due to FP
        tt = 0.0 if tt < 0.0 else (1.0 if tt > 1.0 else tt)
        return float(tt ** i)

    def support(self, i: int):
        # All polynomials are supported on the full interval
        return 0.0, 1.0

    # ---- Back-compat shims (if any old helpers call these) ----
    def evaluate_basis_function(self, i: int, t: float) -> float:
        return self.evaluate(i, t)

    def get_basis_support(self, i: int):
        return self.support(i)