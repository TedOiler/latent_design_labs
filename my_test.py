#!/usr/bin/env python
"""
FoF + both penalties (λs, λt) — minimal NBDO smoke test.
Run from repo root: use the same env you installed the package into.
"""
from latent_design.utils.tf_env import initialize
initialize(seed=42, floatx="float64", inter_threads=1, intra_threads=1,
           deterministic=True, cpp_log_level="3")

# (optional) also set numpy/tf seeds locally if you like, but initialize already did it

import os, sys, time
import numpy as np

# --- If the editable install isn't picked up, fall back to source tree ---
try:
    from latent_design.bases.bspline import BSplineBasis
    from latent_design.models.fof import FunctionOnFunctionModel
    from latent_design.optimizers.nbdo import NBDO
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath("src"))
    from latent_design.bases.bspline import BSplineBasis
    from latent_design.models.fof import FunctionOnFunctionModel
    from latent_design.optimizers.nbdo import NBDO

import tensorflow as tf

def main():

    # --- Small bases so it runs fast ---
    x1 = BSplineBasis(degree=3, total_knots_num=4)
    b1 = BSplineBasis(degree=3, total_knots_num=4)
    x2 = BSplineBasis(degree=3, total_knots_num=4)
    b2 = BSplineBasis(degree=3, total_knots_num=4)

    # For λ_t > 0 you need a response basis
    resp_basis = BSplineBasis(degree=3, total_knots_num=4)

    model = FunctionOnFunctionModel(
        basis_pairs=[(x1, b1), (x2, b2)],
        criterion="A",            # A-opt: numerically robust for sanity checks
        intercept=True,
        lambda_s=0.1,
        lambda_t=0.1,
        response_basis=resp_basis,
        quad_points_penalty=64,   # keep integration light
        eps=1e-6,
        dtype=tf.float64,
    )

    print(f"Kx={model.Kx}, Kb={model.Kb}, p={model.p}, effective_p={model.effective_p}")

    # --- Tiny NBDO run ---
    runs = 30
    opt = NBDO(model=model, latent_dim=3, seed=42, verbose=True)


    opt.compute_train_set(num_designs=128, runs=runs, random_state=42)
    hist = opt.fit(epochs=15, batch_size=64, patience=3)

    report, design = opt.optimize(n_calls=6, n_random_starts=3)

    print("\n=== RESULTS ===")
    print("A-opt report:", report)
    print("Design shape:", design.shape)
    print("First row (rounded):", np.round(design[0], 4))
    print("report_num(design):", model.report_num(design))
    print(f"\nTimings: NBDO fit = {opt.time_fit_s:.3f}s | BO = {opt.time_bo_s:.3f}s | total = {opt.time_total_s:.3f}s")

if __name__ == "__main__":
    main()
