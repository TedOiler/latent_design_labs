import numpy as np

def relative_efficiency(d1: np.ndarray, d2: np.ndarray, model) -> float:
    """
    Relative efficiency of d1 w.r.t. d2 under the model's current criterion.
    Returns Psi(M2) / Psi(M1).
    Interpretation (since we minimize Psi):
      - > 1.0  -> d1 is more efficient (better) than d2
      - = 1.0  -> equally efficient
      - < 1.0  -> d1 is less efficient (worse) than d2
    No numerical stabilization is applied.
    """
    X1 = np.asarray(d1, dtype=float)
    X2 = np.asarray(d2, dtype=float)

    Z1 = model.model_matrix(X1)
    Z2 = model.model_matrix(X2)

    M1 = model.information_matrix(Z1)
    M2 = model.information_matrix(Z2)

    psi1 = model.psi.from_M_np(M1)  # e.g., trace(inv(M1)) or 1/det(M1)
    psi2 = model.psi.from_M_np(M2)

    return psi2 / psi1