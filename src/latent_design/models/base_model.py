from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from latent_design.utils.backend import ArrayLike  # Union[np.ndarray, tf.Tensor]

class BaseModel(ABC):
    """
    Pure interface (no implementations).
    Subclasses (e.g., ScalarOnScalarModel) must implement these methods.
    """

    # Unified TF-first API on shaped inputs
    @abstractmethod
    def model_matrix(self, X_batch: ArrayLike) -> tf.Tensor:
        """
        X_batch: (B, m, Kx) or (m, Kx) [np.ndarray or tf.Tensor]
        Return: Z of shape (B, m, p)
        """
        ...

    @abstractmethod
    def information_matrix(self, Z_batch: ArrayLike) -> tf.Tensor:
        """
        Z_batch: (B, m, p) or (m, p)
        Return: M of shape (B, p, p)
        """
        ...

    @abstractmethod
    def objective(self, X_batch: ArrayLike) -> tf.Tensor:
        """
        X_batch: (B, m, Kx) or (m, Kx)
        Return: loss (B,) tensor (or scalar for unbatched)
        """
        ...

    @abstractmethod
    def objective_from_flat(self, X_flat: ArrayLike, m: int, n: int) -> tf.Tensor:
        """
        X_flat: (B, m*n) or (m*n,)
        Return: loss (B,) tensor (or scalar for unbatched)
        """
        ...

    @abstractmethod
    def report(self, X_batch: ArrayLike) -> tf.Tensor:
        """
        X_batch: (B, m, Kx) or (m, Kx)
        Return: positive criterion (B,) tensor (or scalar for unbatched)
        """
        ...

    # Numeric conveniences (used by BO / logging)
    @abstractmethod
    def objective_num(self, X: NDArray[np.floating]) -> float:
        """
        X: (m, Kx) ndarray -> Python float
        """
        ...

    @abstractmethod
    def report_num(self, X: NDArray[np.floating]) -> float:
        """
        X: (m, Kx) ndarray -> Python float (positive criterion)
        """
        ...