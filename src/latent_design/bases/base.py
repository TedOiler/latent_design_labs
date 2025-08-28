from typing import Tuple, Protocol

class Basis(Protocol):
    def num_basis(self) -> int:
        """Number of basis functions."""
        ...

    def evaluate(self, i: int, t: float) -> float:
        """Value of the i-th basis function at scalar t (Python/NumPy float)."""
        ...

    def support(self, i: int) -> Tuple[float, float]:
        """Interval [a, b] where the i-th basis function can be nonzero."""
        ...
