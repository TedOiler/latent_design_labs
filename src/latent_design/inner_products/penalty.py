# inner_products/penalty.py
from __future__ import annotations
from typing import Sequence, List
import numpy as np

from latent_design.bases.base import Basis  # expects .num_basis() and .evaluate(i, t)

class PenaltyBuilder:
    """
    Build the block-diagonal parameter-direction roughness matrix R0 for
    Scalar-on-Function models using numeric second derivatives.

    For F factors with parameter bases b_basis[f], f=1..F:
      R0 = blkdiag(R_b1, ..., R_bF),  R_b = ∫ b''(t) b''(t)^T dt
    computed via Gauss–Legendre quadrature on [0, 1] and 3-point finite diffs.

    Notes
    -----
    * No derivative methods are required on Basis; everything is numeric.
    * Constant/linear bases naturally yield (near-)zero second derivatives,
      so the corresponding blocks become (near-)zero; we zero them under a tolerance.
    """

    def __init__(
        self,
        b_bases: Sequence[Basis],
        quad_points: int = 256,
        diff_h: float = 1e-4,
        boundary_scheme: str = "one_sided",  # ("one_sided" near 0/1; falls back if needed)
        zero_block_tol: float = 1e-12,
    ):
        if not b_bases:
            raise ValueError("b_bases must contain at least one parameter basis.")
        self.b_bases: List[Basis] = list(b_bases)
        self.quad_points = int(quad_points)
        self.h = float(diff_h)
        self.boundary_scheme = str(boundary_scheme)
        self.zero_block_tol = float(zero_block_tol)

    # ---- quadrature on [0,1] -----------------------------------------------
    def _gauss_legendre_01(self, n: int):
        x, w = np.polynomial.legendre.leggauss(n)  # nodes/weights on [-1,1]
        t = 0.5 * (x + 1.0)
        w01 = 0.5 * w
        return t.astype(float), w01.astype(float)

    # ---- numeric second derivatives for a basis over nodes t ----------------
    def _basis_second_derivatives(self, B: Basis, t: np.ndarray) -> np.ndarray:
        """
        Return Bpp with shape (nq, Kb) where Bpp[k, j] = d^2/dt^2 b_j(t_k)
        computed numerically via finite differences.
        """
        nq = int(t.shape[0])
        Kb = int(B.num_basis())
        Bpp = np.empty((nq, Kb), dtype=float)

        h = self.h
        for j in range(Kb):
            # Loop over nodes (explicit, still cheap at nq~256)
            for k in range(nq):
                tk = float(t[k])

                # central when safe
                if (tk - h) >= 0.0 and (tk + h) <= 1.0:
                    fph = B.evaluate(j, tk + h)
                    f0  = B.evaluate(j, tk)
                    fmh = B.evaluate(j, tk - h)
                    val = (fph - 2.0 * f0 + fmh) / (h * h)
                else:
                    if self.boundary_scheme == "one_sided":
                        # try forward near 0
                        if (tk + 2*h) <= 1.0:
                            f0   = B.evaluate(j, tk)
                            f1   = B.evaluate(j, tk + h)
                            f2   = B.evaluate(j, tk + 2*h)
                            val  = (f0 - 2.0 * f1 + f2) / (h * h)
                        # try backward near 1
                        elif (tk - 2*h) >= 0.0:
                            fm2  = B.evaluate(j, tk - 2*h)
                            fm1  = B.evaluate(j, tk - h)
                            f0   = B.evaluate(j, tk)
                            val  = (fm2 - 2.0 * fm1 + f0) / (h * h)
                        else:
                            # extreme edge: shrink h to whatever fits and do central
                            heff = min(max(tk, 1.0 - tk), h) or h
                            fph  = B.evaluate(j, min(1.0, tk + heff))
                            f0   = B.evaluate(j, tk)
                            fmh  = B.evaluate(j, max(0.0, tk - heff))
                            val  = (fph - 2.0 * f0 + fmh) / (heff * heff)
                    else:
                        # simple clipping central diff
                        fph = B.evaluate(j, min(1.0, tk + h))
                        f0  = B.evaluate(j, tk)
                        fmh = B.evaluate(j, max(0.0, tk - h))
                        val = (fph - 2.0 * f0 + fmh) / (h * h)

                Bpp[k, j] = float(val)

        return Bpp

    @staticmethod
    def _block_diag(blocks: List[np.ndarray]) -> np.ndarray:
        kb_total = int(sum(B.shape[0] for B in blocks))
        R0 = np.zeros((kb_total, kb_total), dtype=float)
        off = 0
        for Blk in blocks:
            kb = int(Blk.shape[0])
            R0[off:off + kb, off:off + kb] = Blk
            off += kb
        return R0

    # ---- public -------------------------------------------------------------
    def build(self) -> np.ndarray:
        """
        Compute and return R0 (Kb_total x Kb_total), block-diagonal across factors.
        """
        t, w = self._gauss_legendre_01(self.quad_points)
        blocks: List[np.ndarray] = []

        for B in self.b_bases:
            Kb = int(B.num_basis())
            if Kb <= 0:
                raise ValueError("Basis must have at least one function.")

            Bpp = self._basis_second_derivatives(B, t)  # (nq, Kb)
            # Weighted inner products of second derivatives:
            # R_b = ∑_k w_k * b''(t_k) b''(t_k)^T = Bpp^T diag(w) Bpp
            Bpp_w = Bpp * w[:, None]                    # (nq, Kb)
            Rb = Bpp.T @ Bpp_w                          # (Kb, Kb)

            # Zero small blocks (linear/constant parameter bases)
            if np.linalg.norm(Rb, ord="fro") < self.zero_block_tol:
                Rb[:] = 0.0

            blocks.append(Rb)

        return self._block_diag(blocks)
