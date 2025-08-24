# optimalities/psi_d.py
import numpy as np
import tensorflow as tf
from .base_psi import BasePsi

class PsiD(BasePsi):
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    # ---- NumPy ----
    def loss_from_M_np(self, M: np.ndarray) -> float:
        p = M.shape[-1]
        M_spd = M + self.eps * np.eye(p, dtype=M.dtype)
        sign, logdet = np.linalg.slogdet(M_spd)
        return float(-logdet)

    def report_from_M_np(self, M: np.ndarray) -> float:
        # Positive criterion to display
        return float(np.exp(-self.loss_from_M_np(M)))

    # ---- TF ----
    def loss_from_M_tf(self, M: tf.Tensor) -> tf.Tensor:
        p = tf.shape(M)[-1]
        M_spd = M + tf.cast(self.eps, M.dtype) * tf.eye(p, dtype=M.dtype)
        return -tf.linalg.logdet(M_spd)

    def report_from_M_tf(self, M: tf.Tensor) -> tf.Tensor:
        return tf.exp(-self.loss_from_M_tf(M))
