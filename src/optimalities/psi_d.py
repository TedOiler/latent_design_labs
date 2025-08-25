from __future__ import annotations
import tensorflow as tf
from .base_psi import BasePsi
from utils.backend import to_tensor, ArrayLike

class PsiD(BasePsi):
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps: float = float(eps)

    def loss_from_M(self, M: ArrayLike) -> tf.Tensor:
        M = to_tensor(M)
        p = tf.shape(M)[-1]
        I = tf.eye(p, batch_shape=tf.shape(M)[:-2], dtype=M.dtype)
        M_spd = M + tf.cast(self.eps, M.dtype) * I
        return -tf.linalg.logdet(M_spd)  # (B,) or scalar

    def report_from_M(self, M: ArrayLike) -> tf.Tensor:
        # Invert the log transform to report determinant on the original scale
        return tf.exp(-self.loss_from_M(M))