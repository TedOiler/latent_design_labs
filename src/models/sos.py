from .base_model import BaseModel
import numpy as np
import tensorflow as tf
from itertools import combinations_with_replacement
from optimalities import PsiA, PsiD, PsiG, PsiI

class ScalarOnScalarModel(BaseModel):
    def __init__(self, Kx, order=1, const: bool = True, criterion: str = "A"):
        self.Kx = int(Kx)
        self.order = int(order)
        self.const = bool(const)

        c = str(criterion).upper()
        if c == "A":
            self.psi = PsiA()
        elif c == "D":
            self.psi = PsiD()
        elif c == "G":
            self.psi = PsiG()
        elif c == "I":
            self.psi = PsiI()
        else:
            (_ for _ in ()).throw(ValueError(f"Unknown {c}"))

        # --- Polynomial basis meta (attribute as requested) ---
        self.monomial_combos = [
            combo
            for o in range(1, self.order + 1)
            for combo in combinations_with_replacement(range(self.Kx), o)
        ]
        self.num_params = (1 if self.const else 0) + len(self.monomial_combos)

    # 1) Raw design
    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.Kx:
            raise ValueError(f"Expected (runs, {self.Kx}) design, got {X.shape}")
        return X

    # 2) Model matrix (NumPy)
    def model_matrix(self, X: np.ndarray) -> np.ndarray:
        D = self.design_matrix(X)  # (runs, Kx)
        cols = []
        if self.const:
            cols.append(np.ones((D.shape[0], 1), dtype=D.dtype))
        for combo in self.monomial_combos:
            block = D[:, combo] if len(combo) > 1 else D[:, [combo[0]]]
            term = np.prod(block, axis=1, dtype=D.dtype) if block.ndim == 2 else block
            cols.append(term.reshape(-1, 1))
        return np.hstack(cols) if cols else np.empty((D.shape[0], 0), dtype=D.dtype)

    # 2b) Model matrix (TF batched)
    def model_matrix_tf(self, X_batch: tf.Tensor) -> tf.Tensor:
        B = tf.shape(X_batch)[0]
        m = tf.shape(X_batch)[1]
        cols = []
        if self.const:
            cols.append(tf.ones((B, m, 1), dtype=X_batch.dtype))
        for combo in self.monomial_combos:
            g = tf.gather(X_batch, indices=list(combo), axis=2)
            term = g if len(combo) == 1 else tf.reduce_prod(g, axis=2, keepdims=True)
            cols.append(term)
        return tf.concat(cols, axis=2) if cols else X_batch[:, :, :0]

    # 3) Information matrices
    def information_matrix(self, Z: np.ndarray) -> np.ndarray:
        return Z.T @ Z

    def information_matrix_tf(self, Z_batch: tf.Tensor) -> tf.Tensor:
        Zt = tf.linalg.matrix_transpose(Z_batch)
        M = tf.matmul(Zt, Z_batch)
        return 0.5 * (M + tf.linalg.matrix_transpose(M))  # harmless symmetrization

    # ---- Objectives (loss) used by training/BO ----
    def objective_np(self, X: np.ndarray) -> float:
        Z = self.model_matrix(X)
        M = self.information_matrix(Z)
        return self.psi.loss_from_M_np(M)

    def objective_tf(self, X_batch: tf.Tensor) -> tf.Tensor:
        Z = self.model_matrix_tf(X_batch)
        M = self.information_matrix_tf(Z)
        return self.psi.loss_from_M_tf(M)

    def objective_tf_from_flat(self, X_flat: tf.Tensor, m: int, n: int) -> tf.Tensor:
        return self.objective_tf(tf.reshape(X_flat, (-1, m, n)))

    # ---- Positive criterion for display/comparison ----
    def report_np(self, X: np.ndarray) -> float:
        Z = self.model_matrix(X)
        M = self.information_matrix(Z)
        return self.psi.report_from_M_np(M)