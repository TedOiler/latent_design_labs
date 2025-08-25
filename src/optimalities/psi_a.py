from __future__ import annotations
import tensorflow as tf
from .base_psi import BasePsi
from utils.backend import to_tensor, ArrayLike

class PsiA(BasePsi):
    def __init__(self, eps: float = 0.0) -> None:
        # Optional jitter for SPD safety (not required).
        self.eps: float = float(eps)

    def loss_from_M(self, M: ArrayLike) -> tf.Tensor:
        M = to_tensor(M)
        p = tf.shape(M)[-1]
        I = tf.eye(p, batch_shape=tf.shape(M)[:-2], dtype=M.dtype)
        if self.eps:
            M = M + tf.cast(self.eps, M.dtype) * I
        Minv = tf.linalg.solve(M, I)   # batched solve
        return tf.linalg.trace(Minv)   # (B,) or scalar

    def report_from_M(self, M: ArrayLike) -> tf.Tensor:
        # Report the original A-opt criterion (trace(M^{-1}))
        return self.loss_from_M(M)