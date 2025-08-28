# bases/fourier.py
import math
from latent_design.bases.base import Basis

class FourierBasis(Basis):
    """
    Real Fourier basis on [0, 1].

    Ordering (include_constant=True):
      i = 0: 1
      i = 1: cos(2π*1*t),  i = 2: sin(2π*1*t)
      i = 3: cos(2π*2*t),  i = 4: sin(2π*2*t)
      ...
      up to n_harmonics.

    Parameters
    ----------
    n_harmonics : int
        Number of positive-frequency harmonics (k = 1..n_harmonics).
    include_constant : bool
        Whether to include the constant term 1.
    """
    def __init__(self, n_harmonics: int, include_constant: bool = True):
        if n_harmonics < 0:
            raise ValueError("n_harmonics must be >= 0")
        self.n_harmonics = int(n_harmonics)
        self.include_constant = bool(include_constant)
        self._offset = 1 if self.include_constant else 0
        self._num_basis = self._offset + 2 * self.n_harmonics

    # ---- New interface ----
    def num_basis(self) -> int:
        return self._num_basis

    def evaluate(self, i: int, t: float) -> float:
        if i < 0 or i >= self._num_basis:
            raise IndexError(f"basis index {i} out of range [0, {self._num_basis-1}]")
        tt = float(t)
        tt = 0.0 if tt < 0.0 else (1.0 if tt > 1.0 else tt)

        if self.include_constant and i == 0:
            return 1.0

        # map index -> (k, kind)
        j = i - self._offset          # 0-based among harmonics
        k = (j // 2) + 1              # harmonic number in {1..n_harmonics}
        is_cos = (j % 2 == 0)         # even: cos, odd: sin

        arg = 2.0 * math.pi * k * tt  # period = 1 on [0,1]
        return math.cos(arg) if is_cos else math.sin(arg)

    def support(self, i: int):
        # Full interval support
        return 0.0, 1.0

    # ---- Back-compat shims ----
    def evaluate_basis_function(self, i: int, t: float) -> float:
        return self.evaluate(i, t)

    def get_basis_support(self, i: int):
        return self.support(i)