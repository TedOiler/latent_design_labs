from __future__ import annotations
from typing import Tuple
import numpy as np
import tensorflow as tf

ArrayLike = np.ndarray | tf.Tensor

# -----------------------
# helpers (model-agnostic)
# -----------------------

def _to_tensor(x: ArrayLike, dtype=None) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return x if dtype is None else tf.cast(x, dtype)
    return tf.convert_to_tensor(x, dtype=dtype)

def _ensure_2d_rows(X: ArrayLike) -> tf.Tensor:
    X = _to_tensor(X)
    if X.shape.rank != 2:
        raise ValueError(f"Expected a 2D array with shape (runs, Kx); got {X.shape}")
    return X

# -----------------------
# core matrices & spectra
# -----------------------

def info_matrix(model, design: ArrayLike) -> tf.Tensor:
    """
    Compute information matrix M = Z^T Z for a given design.
    Squeezes a leading singleton batch if present -> (p, p).
    """
    X = _ensure_2d_rows(design)
    Z = model.model_matrix(X)               # (runs, p) or (1, runs, p)
    # If model returned a batched Z, squeeze it
    if Z.shape.rank == 3 and int(Z.shape[0]) == 1:
        Z = tf.squeeze(Z, axis=0)
    M = model.information_matrix(Z)         # (p, p) or (1, p, p)
    if M.shape.rank == 3 and int(M.shape[0]) == 1:
        M = tf.squeeze(M, axis=0)
    return M

def eigen_spectrum(M: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (eigvals_sorted_desc, eigvecs) for symmetric M.
    """
    M = _to_tensor(M, dtype=tf.float64)
    # force symmetry numerically
    M = 0.5 * (M + tf.linalg.matrix_transpose(M))
    w, v = tf.linalg.eigh(M)                      # ascending
    w = tf.reverse(w, axis=[-1])
    v = tf.reverse(v, axis=[-1])
    return w.numpy(), v.numpy()

def condition_number(M: ArrayLike, eps: float = 1e-15) -> float:
    """
    Spectral condition number κ2 = λ_max / max(λ_min, eps).
    Accepts eigenvalue arrays of shape (p,) or (1, p).
    """
    w, _ = eigen_spectrum(M)      # descending
    w = np.asarray(w)
    if w.ndim == 2 and w.shape[0] == 1:
        w = w[0]
    if w.ndim != 1:
        raise ValueError(f"condition_number expected a single matrix; got eigenvalues shape {w.shape}")
    lam_max = float(w[0])
    lam_min = float(max(w[-1], eps))
    return lam_max / lam_min

# -----------------------
# leverage & pred variance
# -----------------------

def leverage_diag(model, design: ArrayLike) -> np.ndarray:
    """
    Diagonal of the hat matrix H = Z (Z^T Z)^{-1} Z^T (unit-σ^2 scale).
    Shape: (runs,)
    """
    X = _ensure_2d_rows(design)

    # Z: (runs, p) — squeeze if model emits (1, runs, p)
    Z = model.model_matrix(X)
    if Z.shape.rank == 3 and int(Z.shape[0]) == 1:
        Z = tf.squeeze(Z, axis=0)

    # M: (p, p) — squeeze if model emits (1, p, p)
    M = model.information_matrix(Z)
    if M.shape.rank == 3 and int(M.shape[0]) == 1:
        M = tf.squeeze(M, axis=0)

    Minv = tf.linalg.inv(_to_tensor(M))     # (p, p)

    # h_i = z_i^T Minv z_i → compute t = Z @ Minv (runs, p), then rowwise dot with Z
    t = tf.matmul(Z, Minv)                  # (runs, p)
    h = tf.reduce_sum(t * Z, axis=1)        # (runs,)
    return h.numpy()

def prediction_variance(model, design: ArrayLike, new_rows: ArrayLike) -> np.ndarray:
    """
    Prediction variance (unit-σ^2) at user-supplied new rows.
    new_rows must be in the SAME raw space the model expects in model_matrix:
      - SoS: shape (q, Kx) numeric predictors
      - SoF: shape (q, Kx_total) Γ rows (one-run designs)
    Returns: (q,) np.ndarray of z^T (Z^T Z)^{-1} z
    """
    X = _ensure_2d_rows(design)
    Z = model.model_matrix(X)                     # (runs, p)
    M = model.information_matrix(Z)               # (p, p)
    Minv = tf.linalg.inv(_to_tensor(M))

    Xnew = _ensure_2d_rows(new_rows)
    Znew = model.model_matrix(Xnew)               # (q, p)

    tmp = tf.linalg.matvec(Minv, tf.transpose(Znew))   # (p, q)
    pv = tf.reduce_sum(tf.transpose(Znew) * tmp, axis=0)  # (q,)
    return pv.numpy()

# -----------------------
# efficiencies (criterion-agnostic)
# -----------------------

def _det_psd(M: tf.Tensor, eps: float = 0.0) -> tf.Tensor:
    # stable det via eigenvalues (all non-negative for Z^T Z)
    w = tf.linalg.eigvalsh(0.5 * (M + tf.linalg.matrix_transpose(M)))
    if eps:
        w = tf.maximum(w, tf.cast(eps, w.dtype))
    return tf.math.exp(tf.reduce_sum(tf.math.log(tf.maximum(w, tf.cast(1e-300, w.dtype)))))

def d_efficiency(model, design: ArrayLike, reference: ArrayLike, eps: float = 0.0) -> float:
    """
    D-efficiency of `design` vs `reference`:
      Eff_D = ( det(M_design) / det(M_ref) )^(1/p)
    """
    Md = info_matrix(model, design)
    Mr = info_matrix(model, reference)
    p = int(Md.shape[-1])
    det_d = _det_psd(_to_tensor(Md), eps=eps)
    det_r = _det_psd(_to_tensor(Mr), eps=eps)
    eff = tf.pow(det_d / det_r, 1.0 / float(p))
    return float(eff.numpy())

def a_efficiency(model, design: ArrayLike, reference: ArrayLike, eps: float = 0.0) -> float:
    """
    A-efficiency (using trace of inverse):
      Eff_A = ( trace(M_ref^{-1}) / trace(M_design^{-1}) )
    """
    Md = info_matrix(model, design)
    Mr = info_matrix(model, reference)
    if eps:
        I = tf.eye(int(Md.shape[-1]), dtype=Md.dtype)
        Md = Md + eps * I
        Mr = Mr + eps * I
    tr_d = tf.linalg.trace(tf.linalg.inv(_to_tensor(Md)))
    tr_r = tf.linalg.trace(tf.linalg.inv(_to_tensor(Mr)))
    eff = tr_r / tr_d
    return float(eff.numpy())
