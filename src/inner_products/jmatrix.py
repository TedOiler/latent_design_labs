# inner_products/jmatrix.py
from __future__ import annotations
from typing import List, Sequence, Tuple, Dict
import numpy as np

from bases.base import Basis  # your abstract base
# expects Basis-like objects to have:
#   - num_basis(self) -> int
#   - evaluate(self, i: int, t: float) -> float

class JBuilder:
    """
    Build cross-Gram matrix J for one or more (x_basis, b_basis) factor pairs.

    For F factors:
      J = block_diag(J_1, ..., J_F),  J_f in R^{Kx_f x Kb_f},
      where (J_f)_{ij} = ∫_0^1 φ_i(t) ψ_j(t) dt, computed by Gauss–Legendre quadrature.

    Returns:
      {
        "J": np.ndarray (Kx_total, Kb_total),
        "Kx_list": List[int],
        "Kb_list": List[int],
        "x_slices": List[slice],  # Γ column slices per factor
        "b_slices": List[slice],  # Φ column slices per factor
      }
    """

    def __init__(self, basis_pairs: Sequence[Tuple[Basis, Basis]], quad_points: int = 256):
        if not basis_pairs:
            raise ValueError("basis_pairs must contain at least one (x_basis, b_basis) pair.")
        self.basis_pairs: List[Tuple[Basis, Basis]] = list(basis_pairs)
        # 256 Gauss–Legendre nodes is cheap and very accurate for piecewise polynomials.
        self.quad_points = int(quad_points)

    # ---- quadrature helpers -------------------------------------------------

    def _gauss_legendre_01(self, n: int):
        """n-point Gauss-Legendre on [0, 1]: nodes t (n,), weights w (n,)."""
        # nodes/weights on [-1, 1]
        x, w = np.polynomial.legendre.leggauss(n)
        # map to [0, 1]
        t = 0.5 * (x + 1.0)
        w01 = 0.5 * w
        return t.astype(float), w01.astype(float)

    def _eval_basis_matrix(self, B: Basis, t: np.ndarray) -> np.ndarray:
        """Phi[t_idx, i] = basis_i(t[t_idx]) for i=0..K-1."""
        K = int(B.num_basis())
        Phi = np.empty((t.shape[0], K), dtype=float)
        for i in range(K):
            # vectorize per basis function across t
            Phi[:, i] = np.fromiter((B.evaluate(i, float(tt)) for tt in t), count=t.shape[0], dtype=float)
        return Phi

    def _cross_block(self, x_basis: Basis, b_basis: Basis) -> np.ndarray:
        """Compute J_block = ∫ Phi_x^T Phi_b dt via quadrature."""
        t, w = self._gauss_legendre_01(self.quad_points)
        Phi_x = self._eval_basis_matrix(x_basis, t)     # (nq, Kx)
        Phi_b = self._eval_basis_matrix(b_basis, t)     # (nq, Kb)
        # Weighted inner products: (Kx, nq) @ (nq, Kb)
        # Use weights by scaling rows of Phi_b (or Phi_x) before the matmul.
        Phi_b_w = Phi_b * w[:, None]
        J_block = Phi_x.T @ Phi_b_w                      # (Kx, Kb)
        return J_block

    @staticmethod
    def _block_diag(blocks: List[np.ndarray]) -> np.ndarray:
        Kx_total = int(sum(B.shape[0] for B in blocks))
        Kb_total = int(sum(B.shape[1] for B in blocks))
        J = np.zeros((Kx_total, Kb_total), dtype=float)
        x_off = 0
        b_off = 0
        for Blk in blocks:
            kx, kb = Blk.shape
            J[x_off:x_off + kx, b_off:b_off + kb] = Blk
            x_off += kx
            b_off += kb
        return J

    # ---- public -------------------------------------------------------------

    def build(self) -> Dict[str, object]:
        blocks: List[np.ndarray] = []
        Kx_list: List[int] = []
        Kb_list: List[int] = []

        for xB, bB in self.basis_pairs:
            J_blk = self._cross_block(xB, bB)
            if J_blk.ndim != 2:
                raise ValueError(f"Cross block must be 2D; got {J_blk.shape}")
            kx, kb = int(J_blk.shape[0]), int(J_blk.shape[1])

            # Minimal sanity: sizes must match num_basis() when available.
            if hasattr(xB, "num_basis") and int(xB.num_basis()) != kx:
                raise ValueError(f"x-basis num_basis()={xB.num_basis()} but J rows={kx}")
            if hasattr(bB, "num_basis") and int(bB.num_basis()) != kb:
                raise ValueError(f"b-basis num_basis()={bB.num_basis()} but J cols={kb}")

            blocks.append(J_blk)
            Kx_list.append(kx)
            Kb_list.append(kb)

        # Assemble block-diagonal J and slice metadata
        J = self._block_diag(blocks)

        x_slices: List[slice] = []
        b_slices: List[slice] = []
        x_acc = 0
        b_acc = 0
        for kx, kb in zip(Kx_list, Kb_list):
            x_slices.append(slice(x_acc, x_acc + kx))
            b_slices.append(slice(b_acc, b_acc + kb))
            x_acc += kx
            b_acc += kb

        return {
            "J": J,
            "Kx_list": Kx_list,
            "Kb_list": Kb_list,
            "x_slices": x_slices,
            "b_slices": b_slices,
        }
