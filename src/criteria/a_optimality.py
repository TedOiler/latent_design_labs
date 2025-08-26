from __future__ import annotations
import tensorflow as tf
from .base_psi import BasePsi
from utils.backend import to_tensor, ArrayLike

class AOptimality(BasePsi):
    """
    Numerically-stable A-optimality:
      loss(M)   = trace(M^{-1})
      report(M) = 1 / trace(M^{-1})
    with SPD safety via ridge jitter.

    Notes:
      - Works with batched M of shape (..., p, p).
      - Expects M to already be in the model's preferred dtype
        (float64 for SoF), but will honor whatever dtype is passed in.
    """

    def __init__(self, eps: float = 1e-8, max_tries: int = 5, jitter_mul: float = 10.0) -> None:
        # eps is the base ridge; we escalate it if Cholesky fails.
        self.eps: float = float(eps)
        self.max_tries: int = int(max_tries)
        self.jitter_mul: float = float(jitter_mul)

    def _chol_with_jitter(self, M: tf.Tensor) -> tf.Tensor:
        """
        Try a Cholesky factorization, escalating diagonal jitter if needed.
        Returns: lower-triangular L such that (M + jitter*I) = L @ L^T
        """
        M = tf.convert_to_tensor(M)
        p = tf.shape(M)[-1]
        batch_shape = tf.shape(M)[:-2]
        I = tf.eye(p, batch_shape=batch_shape, dtype=M.dtype)

        jitter = tf.cast(self.eps, M.dtype)
        last_err = None
        for _ in range(self.max_tries):
            try:
                L = tf.linalg.cholesky(M + jitter * I)
                return L
            except tf.errors.InvalidArgumentError as e:
                last_err = e
                jitter = tf.cast(tf.maximum(jitter * self.jitter_mul, jitter + tf.cast(self.eps, M.dtype)), M.dtype)
        # Final attempt with a larger jitter before giving up
        L = tf.linalg.cholesky(M + jitter * I)  # let this raise if truly non-SPD
        return L

    def loss_from_M(self, M: ArrayLike) -> tf.Tensor:
        """
        A-opt loss = trace(M^{-1}), computed via Cholesky:
          If M = L L^T, then M^{-1} = L^{-T} L^{-1}.
          trace(M^{-1}) = sum_{i,j} (L^{-1})_{ij}^2 = ||L^{-1}||_F^2
        """
        M = to_tensor(M)
        # Cholesky with jitter escalation for SPD safety
        L = self._chol_with_jitter(M)

        # Solve L * X = I  ->  X = L^{-1}
        p = tf.shape(M)[-1]
        I = tf.eye(p, batch_shape=tf.shape(M)[:-2], dtype=M.dtype)
        L_inv = tf.linalg.triangular_solve(L, I, lower=True)

        # Frobenius norm squared of L^{-1} equals trace(M^{-1})
        loss = tf.reduce_sum(tf.square(L_inv), axis=[-2, -1])
        return loss

    def report_from_M(self, M: ArrayLike) -> tf.Tensor:
        """
        Positive report for A-opt: 1 / trace(M^{-1}).
        This mirrors D-opt using det(M) as the positive report.
        """
        loss = self.loss_from_M(M)
        # Safe reciprocal
        tiny = tf.cast(1e-16, loss.dtype)
        return tf.maximum(loss, tiny)
