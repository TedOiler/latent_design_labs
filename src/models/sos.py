from __future__ import annotations
from typing import Any, List, Tuple
from .base_model import BaseModel
from optimalities.base_psi import BasePsi
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from itertools import combinations_with_replacement
from optimalities import PsiA, PsiD, PsiG, PsiI
from utils.backend import to_tensor, to_float, ArrayLike

class ScalarOnScalarModel(BaseModel):
    def __init__(self, Kx: int, order: int = 1, const: bool = True, criterion: str = "A") -> None:
        self.Kx: int = int(Kx)
        self.order: int = int(order)
        self.const: bool = bool(const)

        c = str(criterion).upper()
        if c == "A":
            self.psi: BasePsi = PsiA()
        elif c == "D":
            self.psi = PsiD()
        elif c == "G":
            self.psi = PsiG()
        elif c == "I":
            self.psi = PsiI()
        else:
            (_ for _ in ()).throw(ValueError(f"Unknown {c}"))

        # --- Polynomial basis meta (attribute as requested) ---
        self.monomial_combos: List[Tuple[int, ...]] = [
            combo
            for o in range(1, self.order + 1)
            for combo in combinations_with_replacement(range(self.Kx), o)
        ]
        self.num_params: int = (1 if self.const else 0) + len(self.monomial_combos)

    # 1) Raw design (NumPy checker — optional convenience)
    def design_matrix(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.Kx:
            raise ValueError(f"Expected (runs, {self.Kx}) design, got {X.shape}")
        return X

    # 2) Model matrix (TensorFlow; accepts np or tf)
    def model_matrix(self, X_batch: ArrayLike) -> tf.Tensor:
        """
        X_batch: (B, m, Kx) or (m, Kx)  (np.ndarray or tf.Tensor)
        Returns Z: (B, m, p), where p = num_params
        """
        X = to_tensor(X_batch)
        if X.shape.rank == 2:
            X = tf.expand_dims(X, axis=0)  # (1, m, Kx)

        cols: List[tf.Tensor] = []
        if self.const:
            B = tf.shape(X)[0]
            m = tf.shape(X)[1]
            cols.append(tf.ones((B, m, 1), dtype=X.dtype))

        for combo in self.monomial_combos:
            g = tf.gather(X, indices=list(combo), axis=2)  # (B, m, len(combo))
            if len(combo) == 1:
                term = g[..., :1]                           # (B, m, 1)
            else:
                term = tf.reduce_prod(g, axis=2, keepdims=True)  # (B, m, 1)
            cols.append(term)

        Z = tf.concat(cols, axis=2) if cols else X[:, :, :0]
        return Z

    # 3) Information matrix (TensorFlow; batched)
    def information_matrix(self, Z_batch: ArrayLike) -> tf.Tensor:
        """
        Z_batch: (B, m, p) or (m, p)
        Returns M: (B, p, p)
        """
        Z = to_tensor(Z_batch)
        if Z.shape.rank == 2:
            Z = tf.expand_dims(Z, axis=0)  # (1, m, p)
        Zt = tf.linalg.matrix_transpose(Z)  # (B, p, m)
        M = tf.matmul(Zt, Z)                # (B, p, p)
        return 0.5 * (M + tf.linalg.matrix_transpose(M))

    # 4) Objective (TensorFlow; batched)
    def objective(self, X_batch: ArrayLike) -> tf.Tensor:
        """
        X_batch: (B, m, Kx) or (m, Kx)
        Returns loss: (B,)
        """
        Z = self.model_matrix(X_batch)
        M = self.information_matrix(Z)
        out = self.psi.loss_from_M(M)      # (B,) or scalar
        return tf.reshape(out, (-1,))      # ensure (B,)

    def objective_from_flat(self, X_flat: ArrayLike, m: int, n: int) -> tf.Tensor:
        """
        X_flat: (B, m*n) or (m*n,)
        Returns loss: (B,)
        """
        X = tf.reshape(to_tensor(X_flat), (-1, m, n))
        return self.objective(X)

    # 5) Positive criterion for display/comparison (TensorFlow; batched)
    def report(self, X_batch: ArrayLike) -> tf.Tensor:
        """
        X_batch: (B, m, Kx) or (m, Kx)
        Returns positive metric: (B,)
        """
        Z = self.model_matrix(X_batch)
        M = self.information_matrix(Z)
        out = self.psi.report_from_M(M)    # (B,) or scalar
        return tf.reshape(out, (-1,))

    # 6) Numeric convenience wrappers (for BO / printing)
    def objective_num(self, X: NDArray[np.floating]) -> float:
        """
        X: (m, Kx) NumPy array -> Python float
        """
        val = self.objective(tf.expand_dims(to_tensor(X), axis=0))  # (1,)
        return to_float(tf.reshape(val, ()))    # scalar

    def report_num(self, X: NDArray[np.floating]) -> float:
        val = self.report(tf.expand_dims(to_tensor(X), axis=0))     # (1,)
        return to_float(tf.reshape(val, ()))    # scalar
