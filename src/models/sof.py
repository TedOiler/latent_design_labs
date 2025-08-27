from __future__ import annotations

from typing import List, Sequence, Tuple, Optional
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from .base_model import BaseModel
from bases.base import Basis
from inner_products.jmatrix import JBuilder
from inner_products.penalty import PenaltyBuilder
from criteria import AOptimality, DOptimality


class ScalarOnFunctionModel(BaseModel):
    """
    Scalar-on-Function linear model with potentially multiple functional factors.

    Notation:
      - Γ (Gamma): design coefficients matrix, shape (runs, Kx_total)
      - J: block-diagonal cross-Gram matrix, shape (Kx_total, Kb_total), fixed
      - Z = [1, Γ J] if intercept=True else Γ J
      - M = Zᵀ Z

    The class builds J once (NumPy), freezes it as a TF constant, and implements:
      Γ -> Z -> M, then delegates the scalar criterion to PsiA / PsiD.
    """

    def __init__(
        self,
        basis_pairs: Sequence[Tuple[Basis, Basis]],
        criterion: str = "A",
        intercept: bool = True,
        eps: float = 1e-6,
        dtype: tf.dtypes.DType = tf.float64,
        lambda_penalty: float | None = None,
        quad_points_penalty: int = 256,
        diff_h: float = 1e-4,
    ) -> None:
        # config
        self.basis_pairs: List[Tuple[Basis, Basis]] = list(basis_pairs)
        self.criterion: str = criterion.upper()
        if self.criterion not in ("A", "D"):
            raise ValueError("criterion must be 'A' or 'D'")
        self.intercept: bool = bool(intercept)
        self.eps: float = float(eps)
        self.dtype = dtype
        self.lambda_penalty = lambda_penalty          # NEW
        if self.lambda_penalty is not None and self.lambda_penalty < 0:
            raise ValueError("lambda_penalty must be non-negative or None")
        self._pen_quad_pts = int(quad_points_penalty)
        self._pen_diff_h = float(diff_h)

        # build J (NumPy) once; freeze as TF tensor for runtime
        jb = JBuilder(self.basis_pairs)
        out = jb.build()

        self.J_np: np.ndarray = out["J"]               # (Kx_total, Kb_total)
        self.J: tf.Tensor = tf.constant(self.J_np, dtype=self.dtype)
        self.Kx_list: List[int] = out["Kx_list"]
        self.Kb_list: List[int] = out["Kb_list"]
        self.x_slices = out["x_slices"]                # Γ column slices per factor
        self.b_slices = out["b_slices"]                # Φ=Γ@J column slices per factor

        self._Kx_total: int = int(self.J_np.shape[0])
        self._Kb_total: int = int(self.J_np.shape[1])

        # --- NEW: build padded penalty R once if λ is active ---
        self.R: Optional[tf.Tensor] = None
        if self.lambda_penalty is not None and self.lambda_penalty > 0:
            # Parameter bases only (order must match J's column layout)
            b_bases = [b for (_x, b) in self.basis_pairs]
            R0_np = PenaltyBuilder(
                b_bases=b_bases,
                quad_points=self._pen_quad_pts,
                diff_h=self._pen_diff_h,
                boundary_scheme="one_sided",
            ).build()  # (Kb, Kb)
            if self.intercept:
                R_np = np.zeros((self.Kb + 1, self.Kb + 1), dtype=R0_np.dtype)
                R_np[1:, 1:] = R0_np  # leave intercept unpenalized
            else:
                R_np = R0_np
            self.R = tf.constant(R_np, dtype=self.dtype)

        # choose Psi (numerically-stable A/D)
        if self.criterion == "A":
            self.psi = AOptimality(eps=self.eps)
        else:
            self.psi = DOptimality(eps=self.eps)

    # ---- public properties -------------------------------------------------

    @property
    def Kx(self) -> int:
        """Total number of input-basis coefficients (sum over factors)."""
        return self._Kx_total

    @property
    def Kb(self) -> int:
        """Total number of parameter-basis coefficients (sum over factors)."""
        return self._Kb_total

    @property
    def p(self) -> int:
        """Number of columns in Z: Kb + 1 if intercept else Kb."""
        return self.Kb + (1 if self.intercept else 0)

    # ---- helpers -----------------------------------------------------------

    def _as_batched_gamma(self, Gamma: tf.Tensor):
        """
        Ensure Gamma has shape (B, runs, Kx).
        Accepts (runs, Kx) and adds a leading batch dim.
        Returns (Gamma_batched, added_batch_flag).
        """
        Gamma = tf.cast(Gamma, self.dtype)
        added_batch = False
        static_rank = Gamma.shape.rank  # may be 2, 3, or None

        if static_rank == 2:
            Gamma = tf.expand_dims(Gamma, axis=0)  # -> (1, runs, Kx)
            added_batch = True
        elif static_rank == 3:
            pass  # already (B, runs, Kx)
        elif static_rank is None:
            # In typical eager/keras usage static rank is known; if not, keep as-is.
            pass
        else:
            raise ValueError(f"Gamma must have rank 2 or 3; got {Gamma.shape}")

        # Optional static shape check on the last dim
        if Gamma.shape[-1] is not None and int(Gamma.shape[-1]) != self.Kx:
            raise ValueError(f"Gamma last dimension ({Gamma.shape[-1]}) must equal Kx={self.Kx}")

        return Gamma, added_batch

    def _with_intercept(self, Phi: tf.Tensor) -> tf.Tensor:
        """
        Prepend a column of ones if intercept=True.
        Phi: (B, runs, Kb)  ->  Z: (B, runs, Kb + 1) when intercept, else Phi.
        """
        if not self.intercept:
            return Phi
        B = tf.shape(Phi)[0]
        runs = tf.shape(Phi)[1]
        ones = tf.ones((B, runs, 1), dtype=Phi.dtype)
        return tf.concat([ones, Phi], axis=2)

    # ---- BaseModel required API -------------------------------------------

    def model_matrix(self, X_batch) -> tf.Tensor:
        """
        Build Z from Γ:
          Phi = Γ @ J,  Z = [1, Phi] if intercept else Phi

        Accepts:
          - X_batch as Γ with shape (runs, Kx) or (B, runs, Kx)

        Returns:
          - Z with shape (runs, p) or (B, runs, p)
            where p = Kb + 1 if intercept else Kb
        """
        Gamma, added = self._as_batched_gamma(X_batch)   # (B, runs, Kx)
        # (B, runs, Kx) @ (Kx, Kb) -> (B, runs, Kb)
        Phi = tf.matmul(Gamma, self.J)                    # dtype=self.dtype
        Z = self._with_intercept(Phi)                     # (B, runs, p)

        if added:
            Z = tf.squeeze(Z, axis=0)                     # (runs, p)
        return Z

    def information_matrix(self, Z: tf.Tensor) -> tf.Tensor:
        """
        Compute M = Zᵀ Z.

        Accepts:
          - Z with shape (runs, p) or (B, runs, p)

        Returns:
          - M with shape (p, p) or (B, p, p)
        """
        Z = tf.convert_to_tensor(Z, dtype=self.dtype)
        added = False
        if Z.shape.rank == 2:
            Z = tf.expand_dims(Z, axis=0)   # (1, runs, p)
            added = True

        # (B, runs, p)ᵀ @ (B, runs, p) -> (B, p, p)
        M = tf.matmul(Z, Z, transpose_a=True)

        if added:
            M = tf.squeeze(M, axis=0)       # (p, p)
        return M
    
    # --- NEW: apply M -> M + λR if active ---
    def _regularize_information(self, M: tf.Tensor) -> tf.Tensor:
        if self.R is None or self.lambda_penalty in (None, 0.0):
            return M
        lam = tf.cast(self.lambda_penalty, self.dtype)
        return M + tf.cast(self.lambda_penalty, self.dtype) * self.R

    # ---- Criterion (Psi) wiring -------------------------------------------

    def objective(self, Gamma_batch: tf.Tensor) -> tf.Tensor:
        """
        Return the scalar loss (A- or D-optimality) for Γ or a batch of Γ.
        If a batch is provided, returns a 1-D tensor of losses (length B).
        """
        Z = self.model_matrix(Gamma_batch)
        M = self.information_matrix(Z)
        M = self._regularize_information(M)  # NEW
        loss = self.psi.loss_from_M(M)
        return loss

    def report(self, Gamma_batch: tf.Tensor) -> tf.Tensor:
        """
        Return the positive 'report' score corresponding to the criterion,
        e.g., det(M) for D-opt or 1/trace(M^{-1}) for A-opt (depending on Psi).
        """
        Z = self.model_matrix(Gamma_batch)
        M = self.information_matrix(Z)
        M = self._regularize_information(M)  # NEW
        score = self.psi.report_from_M(M)
        return score

    def objective_from_flat(self, flat_batch: tf.Tensor, runs: int) -> tf.Tensor:
        flat = tf.cast(flat_batch, self.dtype)
        if flat.shape.rank == 1:
            flat = tf.expand_dims(flat, axis=0)
        Gamma = tf.reshape(flat, (-1, runs, self.Kx))
        return self.objective(Gamma)

    def report_from_flat(self, flat_batch: tf.Tensor, runs: int) -> tf.Tensor:
        flat = tf.cast(flat_batch, self.dtype)
        if flat.shape.rank == 1:
            flat = tf.expand_dims(flat, axis=0)
        Gamma = tf.reshape(flat, (-1, runs, self.Kx))
        return self.report(Gamma)


    # ---- numeric conveniences (NumPy in/out, TF under the hood) -----------

    def objective_num(self, Gamma):
        loss = self.objective(tf.cast(Gamma, self.dtype))
        loss = tf.reshape(loss, [-1])
        return float(loss[0].numpy())

    def report_num(self, Gamma):
        score = self.report(tf.cast(Gamma, self.dtype))
        score = tf.reshape(score, [-1])
        return float(score[0].numpy())