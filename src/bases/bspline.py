import numpy as np
from .base import Basis

class BSplineBasis(Basis):
    """
    Open-uniform B-spline basis on [0, 1].

    Parameters
    ----------
    degree : int
        Polynomial degree (k). Order = degree + 1.
    total_knots_num : int
        Number of grid points to form internal knots in (0, 1).
    """
    def __init__(self, degree: int, total_knots_num: int):
        self.degree = int(degree)
        self.order = self.degree + 1

        # Internal knots strictly inside (0, 1)
        internal = np.linspace(0.0, 1.0, int(total_knots_num))
        self._internal_knots = internal[1:-1]

        # Open-uniform knot vector (repeat endpoints 'order' times)
        self._lower = np.zeros(self.order, dtype=float)
        self._upper = np.ones(self.order, dtype=float)
        self.augmented_knots = np.concatenate([self._lower, self._internal_knots, self._upper])

        # Number of basis functions
        self._num_basis = len(self.augmented_knots) - self.order

    # ---- New interface ----
    def num_basis(self) -> int:
        return self._num_basis

    def evaluate(self, i: int, t: float) -> float:
        return self._eval_bspline(i, self.degree, float(t))

    def support(self, i: int):
        # i-th basis has support on [t_i, t_{i+order}]
        a = float(self.augmented_knots[i])
        b = float(self.augmented_knots[i + self.order])
        return a, b

    # ---- Back-compat shims (if any old helpers call these) ----
    def evaluate_basis_function(self, i: int, t: float) -> float:
        return self.evaluate(i, t)

    def get_basis_support(self, i: int):
        return self.support(i)

    # ---- Internal: Cox–de Boor recursion ----
    def _eval_bspline(self, i: int, k: int, t: float) -> float:
        knots = self.augmented_knots

        if k == 0:
            if knots[i] <= t < knots[i + 1]:
                return 1.0
            # include the final knot on the rightmost interval
            if t == knots[-1] and t == knots[i + 1]:
                return 1.0
            return 0.0

        # Left term
        denom1 = knots[i + k] - knots[i]
        term1 = 0.0
        if denom1 != 0.0:
            term1 = ((t - knots[i]) / denom1) * self._eval_bspline(i, k - 1, t)

        # Right term
        denom2 = knots[i + k + 1] - knots[i + 1]
        term2 = 0.0
        if denom2 != 0.0:
            term2 = ((knots[i + k + 1] - t) / denom2) * self._eval_bspline(i + 1, k - 1, t)

        return term1 + term2
