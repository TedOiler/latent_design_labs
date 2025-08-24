import numpy as np
import tensorflow as tf
from .base_psi import BasePsi

class PsiG(BasePsi):
    """
    G-optimality with a fixed evaluation grid G:
      loss/report: max_j phi(x_j)^T M^{-1} phi(x_j)
    """
    name = "G"

    def __init__(self, model, X_eval: np.ndarray):
        Phi = model.model_matrix(np.asarray(X_eval, dtype=float))  # (m_eval, p)
        self.Phi_np = Phi.astype(float, copy=False)
        self.Phi_tf = tf.constant(self.Phi_np, dtype=tf.float32)

    # ---- NumPy ----
    def loss_from_M_np(self, M: np.ndarray) -> float:
        Minv = np.linalg.inv(M)
        pv = np.einsum('ij,jk,ik->i', self.Phi_np, Minv, self.Phi_np, optimize=True)
        return float(np.max(pv))

    def report_from_M_np(self, M: np.ndarray) -> float:
        return self.loss_from_M_np(M)

    # ---- TensorFlow (batched) ----
    def loss_from_M_tf(self, M: tf.Tensor) -> tf.Tensor:
        M = tf.convert_to_tensor(M)
        Phi = tf.cast(self.Phi_tf, M.dtype)                     # (m_eval, p)
        Phi_b = tf.expand_dims(Phi, axis=0)
        Minv = tf.linalg.inv(M)
        if tf.rank(Minv) == 2:
            Minv = tf.expand_dims(Minv, axis=0)
        y = tf.matmul(Phi_b, Minv)                              # (batch, m_eval, p)
        pv = tf.reduce_sum(y * Phi_b, axis=-1)                  # (batch, m_eval)
        return tf.reduce_max(pv, axis=-1)

    def report_from_M_tf(self, M: tf.Tensor) -> tf.Tensor:
        return self.loss_from_M_tf(M)
