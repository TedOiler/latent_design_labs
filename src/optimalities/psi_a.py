# optimalities/psi_a.py
import numpy as np
import tensorflow as tf
from .base_psi import BasePsi

class PsiA(BasePsi):
    """
    A-optimality (minimize trace(M^{-1})).
    NumPy and TF definitions are identical algebraically.
    """
    # ---- NumPy ----
    def loss_from_M_np(self, M: np.ndarray) -> float:
        Minv = np.linalg.inv(M)
        return float(np.trace(Minv))

    # ---- TensorFlow ----
    def loss_from_M_tf(self, M: tf.Tensor) -> tf.Tensor:
        Minv = tf.linalg.inv(M)
        return tf.linalg.trace(Minv)
