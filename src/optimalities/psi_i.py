import numpy as np
import tensorflow as tf
from .base_psi import BasePsi

class PsiI(BasePsi):
    """
    I-optimality with a fixed evaluation grid G:
      loss/report: mean_j phi(x_j)^T M^{-1} phi(x_j)
    """
    name = "I"

    def __init__(self, model, X_eval: np.ndarray):
        """
        model: your ScalarOnScalarModel (for basis construction)
        X_eval: (m_eval, Kx) grid in coded space
        """
        # Precompute Phi_G in NumPy (consistent with model.order/const)
        Phi = model.model_matrix(np.asarray(X_eval, dtype=float))  # (m_eval, p)
        self.Phi_np = Phi.astype(float, copy=False)
        # Store TF constant for training path (batched)
        self.Phi_tf = tf.constant(self.Phi_np, dtype=tf.float32)   # dtype adapted at call

    # ---- NumPy ----
    def loss_from_M_np(self, M: np.ndarray) -> float:
        Minv = np.linalg.inv(M)                                 # (p,p)
        # pv_j = phi_j^T Minv phi_j   for all j
        pv = np.einsum('ij,jk,ik->i', self.Phi_np, Minv, self.Phi_np, optimize=True)  # (m_eval,)
        return float(np.mean(pv))

    def report_from_M_np(self, M: np.ndarray) -> float:
        # I-opt is already a positive quantity -> report == loss
        return self.loss_from_M_np(M)

    # ---- TensorFlow (batched) ----
    def loss_from_M_tf(self, M: tf.Tensor) -> tf.Tensor:
        """
        M: (batch, p, p) or (p, p)
        returns: (batch,) or scalar — mean prediction variance over X_eval
        """
        M = tf.convert_to_tensor(M)
        # Make Phi_tf match dtype of M
        Phi = tf.cast(self.Phi_tf, M.dtype)                     # (m_eval, p)
        # Promote to batch: (1, m_eval, p)
        Phi_b = tf.expand_dims(Phi, axis=0)
        # Minv: handle both batched and unbatched
        Minv = tf.linalg.inv(M)                                 # (batch,p,p) or (p,p)
        if tf.rank(Minv) == 2:
            Minv = tf.expand_dims(Minv, axis=0)                # (1,p,p)
        # y = Phi * Minv  -> (batch, m_eval, p)
        y = tf.matmul(Phi_b, Minv)
        # pv = sum_j (Phi ⊙ y) across columns -> (batch, m_eval)
        pv = tf.reduce_sum(y * Phi_b, axis=-1)
        # mean over evaluation points -> (batch,)
        return tf.reduce_mean(pv, axis=-1)

    def report_from_M_tf(self, M: tf.Tensor) -> tf.Tensor:
        return self.loss_from_M_tf(M)
