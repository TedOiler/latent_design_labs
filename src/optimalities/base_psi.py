# optimalities/base_psi.py
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class BasePsi(ABC):
    # Loss used by optimizers (NumPy/TF)
    @abstractmethod
    def loss_from_M_np(self, M: np.ndarray) -> float: ...
    @abstractmethod
    def loss_from_M_tf(self, M: tf.Tensor) -> tf.Tensor: ...

    # Positive criterion to display / compare (defaults to loss if already positive)
    def report_from_M_np(self, M: np.ndarray) -> float:
        return float(self.loss_from_M_np(M))

    def report_from_M_tf(self, M: tf.Tensor) -> tf.Tensor:
        return self.loss_from_M_tf(M)
