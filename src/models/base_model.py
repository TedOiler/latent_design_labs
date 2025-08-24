from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class BaseModel(ABC):
    # NumPy path on (runs, Kx)
    @abstractmethod
    def objective_np(self, X: np.ndarray) -> float:
        ...

    # TF path on flattened input (batch, runs*Kx) – used by Keras loss
    @abstractmethod
    def objective_tf_from_flat(self, X_flat: tf.Tensor, m: int, n: int) -> tf.Tensor:
        ...

    # Convenience (shared default): reshape then call objective_np
    def objective_np_from_flat(self, X_flat: np.ndarray, m: int, n: int) -> float:
        X = np.asarray(X_flat, dtype=float).reshape(m, n)
        return float(self.objective_np(X))