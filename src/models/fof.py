# src/models/fof.py
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


class FunctionOnFunctionModel(BaseModel):
    """
    Function-on-Function linear model with potentially multiple functional factors.

    Notation:
      - Γ (Gamma): design coefficients matrix, shape (runs, Kx_total)
      - J: block-diagonal cross-Gram matrix, shape (Kx_total, Kb_total), fixed
      - Z = [1, Γ J] if intercept=True else Γ J
      - M = Zᵀ Z

    Public API mirrors ScalarOnFunctionModel exactly so NBDO can use it
    without special-casing. The only FoF-specific addition is `response_basis`,
    stored for future penalty/reporting work (unused for A/D criteria).
    """

    def __init__(
        self,
        basis_pairs: Sequence[Tuple[Basis, Basis]],
        criterion: str = "A",
        intercept: bool = True,
        eps: float = 1e-6,
        dtype: tf.dtypes.DType = tf.float64,
        response_basis: Optional[Basis] = None,  # FoF-specific; needed if λ_t>0
        lambda_s: Optional[float] = None,
        lambda_t: Optional[float] = None,
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
        self.response_basis = response_basis
        self.n_predictors: int = len(self.basis_pairs)
        # NEW: store λ's and numeric penalty options (no behavior change yet)
        self.lambda_s: Optional[float] = lambda_s
        self.lambda_t: Optional[float] = lambda_t
        if self.lambda_s is not None and self.lambda_s < 0:
            raise ValueError("lambda_s must be non-negative or None")
        if self.lambda_t is not None and self.lambda_t < 0:
            raise ValueError("lambda_t must be non-negative or None")
        if self.lambda_t is not None and self.lambda_t > 0 and self.response_basis is None:
            raise ValueError("response_basis is required when lambda_t > 0")
        # keep naming consistent with SoF
        self._pen_quad_pts = int(quad_points_penalty)
        self._pen_diff_h = float(diff_h)
        # placeholders (will be built in Item 3)
        self.R_s: Optional[tf.Tensor] = None
        self.S_t: Optional[tf.Tensor] = None
        self.J_HH: Optional[tf.Tensor] = None

        # build J (NumPy) once; freeze as TF tensor for runtime
        jb = JBuilder(self.basis_pairs)
        out = jb.build()
        # Match ScalarOnFunctionModel's expected keys
        self.J_np: np.ndarray = out["J"]               # (Kx_total, Kb_total)
        self.J: tf.Tensor = tf.constant(self.J_np, dtype=self.dtype)
        self.Kx_list: List[int] = out["Kx_list"]
        self.Kb_list: List[int] = out["Kb_list"]
        self.x_slices = out["x_slices"]                # Γ column slices per factor
        self.b_slices = out["b_slices"]                # Φ=Γ@J column slices per factor

        self._Kx_total: int = int(self.J_np.shape[0])
        self._Kb_total: int = int(self.J_np.shape[1])

        # --- NEW: precompute FoF penalty/Gram caches (NumPy -> TF) ----------
        # R_s: block-diagonal roughness over s (parameter-direction, per predictor),
        #      padded to (p x p) to leave intercept unpenalized.
        # S_t: roughness over t (response-direction), (L x L), L = response_basis.num_basis()
        # J_HH: diag(1, J_etaeta1, ..., J_etaetaF) with J_etaeta = ∫ η(s)η(s)^T ds, padded to (p x p)
        self.R_s: Optional[tf.Tensor] = None
        self.S_t: Optional[tf.Tensor] = None
        self.J_HH: Optional[tf.Tensor] = None

        # Build R_s if lambda_s > 0
        if self.lambda_s is not None and self.lambda_s > 0:
            eta_bases = [b for (_x, b) in self.basis_pairs]
            R0_np = PenaltyBuilder(
                b_bases=eta_bases,
                quad_points=self._pen_quad_pts,
                diff_h=self._pen_diff_h,
                boundary_scheme="one_sided",
            ).build()  # (Kb, Kb)
            if self.intercept:
                R_pad = np.zeros((self.Kb + 1, self.Kb + 1), dtype=R0_np.dtype)
                R_pad[1:, 1:] = R0_np
            else:
                R_pad = R0_np
            self.R_s = tf.constant(R_pad, dtype=self.dtype)

        # Build S_t and J_HH if lambda_t > 0
        if self.lambda_t is not None and self.lambda_t > 0:
            # S_t from response basis (single block)
            S_np = PenaltyBuilder(
                b_bases=[self.response_basis],  # type: ignore[arg-type]
                quad_points=self._pen_quad_pts,
                diff_h=self._pen_diff_h,
                boundary_scheme="one_sided",
            ).build()  # (L, L)
            self.S_t = tf.constant(S_np, dtype=self.dtype)

            # J_HH = diag(1, J_etaeta1, ..., J_etaetaF), padded to (p x p)
            eta_pairs = [(b, b) for (_x, b) in self.basis_pairs]
            jh = JBuilder(eta_pairs).build()
            J_etaeta_np: np.ndarray = jh["J"]  # (Kb, Kb)
            if self.intercept:
                J_pad = np.zeros((self.Kb + 1, self.Kb + 1), dtype=J_etaeta_np.dtype)
                J_pad[0, 0] = 1.0
                J_pad[1:, 1:] = J_etaeta_np
            else:
                J_pad = J_etaeta_np
            self.J_HH = tf.constant(J_pad, dtype=self.dtype)

        # choose Psi (numerically-stable A/D) — same interface as SoF
        if self.criterion == "A":
            self.psi = AOptimality(eps=self.eps)
        else:
            self.psi = DOptimality(eps=self.eps)

    # ---- public properties (parity with SoF) -------------------------------

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

    @property
    def effective_p(self) -> int:
        """
        When penalties are inactive: effective_p = p.
        When λ_s>0 or λ_t>0: effective_p = L * p, with
        L = response_basis.num_basis() if provided else 1.
        """
        any_pen = not (self.lambda_s in (None, 0.0) and self.lambda_t in (None, 0.0))
        if not any_pen:
            return self.p
        L = self.response_basis.num_basis() if self.response_basis is not None else 1
        return int(L) * int(self.p)    

    # ---- helpers (identical semantics to SoF) ------------------------------

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
            # if unknown at graph-build time, allow it (runtime checks later)
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

    # ---- BaseModel required API (parity with SoF) --------------------------

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
        Gamma, added = self._as_batched_gamma(X_batch)  # (B, runs, Kx)
        Phi = tf.matmul(Gamma, self.J)                  # (B, runs, Kb)
        Z = self._with_intercept(Phi)                   # (B, runs, p)

        if added:
            Z = tf.squeeze(Z, axis=0)                   # (runs, p)
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
    
    # --- Keep naming consistent with SoF: penalty hook ------------------
    def _regularize_information(self, M: tf.Tensor) -> tf.Tensor:
        """
        Return F_λ = (I_L ⊗ M) + λ_s (I_L ⊗ R_s) + λ_t (S_t ⊗ J_HH)
        when λ_s or λ_t are active; otherwise return M unchanged.
        Shapes:
          - Input M: (p,p) or (B,p,p)
          - Output F_λ: (L*p, L*p) or (B,L*p,L*p)
        return M
        """
        # If both penalties inactive, keep original behavior
        if (self.lambda_s in (None, 0.0)) and (self.lambda_t in (None, 0.0)):
            return M

        # Ensure batch dimension
        added_batch = False
        if M.shape.rank == 2:
            M = tf.expand_dims(M, axis=0)   # (1, p, p)
            added_batch = True

        B = tf.shape(M)[0]
        p = tf.shape(M)[-1]
        # Response basis size; default L=1 if not provided
        L = (self.response_basis.num_basis() if self.response_basis is not None else 1)

        # --- kron(I_L, M): block-diagonal of M repeated L times (batched) ---
        # Use LinearOperatorBlockDiag to avoid manual tiling logic.
        opM = tf.linalg.LinearOperatorFullMatrix(M)  # batch (B,)
        op_blocks = [opM] * int(L)
        F = tf.linalg.LinearOperatorBlockDiag(op_blocks).to_dense()  # (B, L*p, L*p)

        # --- + λ_s * kron(I_L, R_s) -----------------------------------------
        if self.lambda_s not in (None, 0.0):
            if self.R_s is None:
                raise RuntimeError("R_s is not built but lambda_s > 0.")
            opRs = tf.linalg.LinearOperatorFullMatrix(self.R_s)  # (p,p)
            Rs_big = tf.linalg.LinearOperatorBlockDiag([opRs] * int(L)).to_dense()  # (L*p, L*p)
            F = F + tf.cast(self.lambda_s, self.dtype) * Rs_big  # broadcast over batch

        # --- + λ_t * kron(S_t, J_HH) ----------------------------------------
        if self.lambda_t not in (None, 0.0):
            if self.S_t is None or self.J_HH is None:
                raise RuntimeError("S_t or J_HH is not built but lambda_t > 0.")
            SJ = self._kron(self.S_t, self.J_HH)   # (L*p, L*p), no batch
            F = F + tf.cast(self.lambda_t, self.dtype) * SJ     # broadcast over batch

        if added_batch:
            F = tf.squeeze(F, axis=0)  # (L*p, L*p)
        return F

    @staticmethod
    def _kron(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
        """
        Dense Kronecker product for 2-D tensors:
          A (m x n), B (p x q) -> (m*p x n*q)
        """
        A = tf.convert_to_tensor(A)
        B = tf.convert_to_tensor(B, dtype=A.dtype)
        a0, a1 = tf.shape(A)[0], tf.shape(A)[1]
        b0, b1 = tf.shape(B)[0], tf.shape(B)[1]
        # tensordot -> (a0, a1, b0, b1) then reshape to (a0*b0, a1*b1)
        K = tf.tensordot(A, B, axes=0)
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        K = tf.reshape(K, (a0 * b0, a1 * b1))
        return K

    # ---- Criterion (Psi) wiring (exactly like SoF) -------------------------

    def objective(self, Gamma_batch: tf.Tensor) -> tf.Tensor:
        """
        Return the scalar loss (A- or D-optimality) for Γ or a batch of Γ.
        If a batch is provided, returns a 1-D tensor of losses (length B).
        """
        Z = self.model_matrix(Gamma_batch)
        M = self.information_matrix(Z)
        M = self._regularize_information(M)   # naming aligned with SoF
        loss = self.psi.loss_from_M(M)   
        return loss

    def report(self, Gamma_batch: tf.Tensor) -> tf.Tensor:
        """
        Return the positive 'report' score corresponding to the criterion,
        e.g., det(M) for D-opt or 1/trace(M^{-1}) for A-opt (depending on Psi).
        """
        Z = self.model_matrix(Gamma_batch)
        M = self.information_matrix(Z)
        M = self._regularize_information(M)   # naming aligned with SoF
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
