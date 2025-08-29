# cli/app.py
from __future__ import annotations

# Silence most TF C++ logs; we'll also hard-mute OS stderr in blocks below.
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import csv
import json
import os
import sys
import contextlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import numpy as np
import typer

# ---- Your package imports (adjust paths if needed) -------------------------
from latent_design.utils.tf_env import initialize as tf_initialize
from latent_design.models.sos import ScalarOnScalarModel
from latent_design.models.sof import ScalarOnFunctionModel
from latent_design.models.fof import FunctionOnFunctionModel

from latent_design.bases.bspline import BSplineBasis
from latent_design.bases.fourier import FourierBasis

from latent_design.optimizers.nbdo import NBDO

# ---------------------------------------------------------------------------
# Typer app as a GROUP (so `nbdo run ...` works)
# ---------------------------------------------------------------------------
app = typer.Typer(no_args_is_help=True, add_completion=False, help="NBDO CLI (single-run)")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MASTER_HEADER = ["section", "key", "value", "design"]  # last column named 'design'


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def make_run_id(tag: str | None = None) -> str:
    athens = timezone(timedelta(hours=3))
    ts = datetime.now(athens).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}__{tag}" if tag else ts


@contextlib.contextmanager
def _mute_os_stderr():
    """
    Mute OS-level STDERR (fd=2) within the block. This hides TF C++ noise like:
      'NodeDef mentions attribute use_unbounded_threadpool ...'
    Other Python exceptions will still raise.
    """
    try:
        orig_fd = sys.stderr.fileno()
    except Exception:
        # Non-standard env; just run without muting.
        yield
        return

    saved = os.dup(orig_fd)  # duplicate current fd 2
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, orig_fd)  # redirect fd 2 -> /dev/null
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, orig_fd)  # restore fd 2
        os.close(saved)


def _serialize_design(design: Any) -> str:
    """Serialize design (np.ndarray or list-like) into a compact JSON string for a single TSV cell."""
    try:
        import numpy as np  # lazy import
    except Exception:
        np = None  # type: ignore

    if np is not None and isinstance(design, np.ndarray):
        data = design.tolist()
    else:
        try:
            data = list(design)
        except Exception:
            data = design
    return json.dumps(data, separators=(",", ":"))


def append_master_tsv(tsv_path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    tsv_path = Path(tsv_path)
    file_exists = tsv_path.exists()
    ensure_dir(tsv_path.parent)

    with tsv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_HEADER, delimiter="\t", extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            safe_row = {k: str(row.get(k, "")) for k in MASTER_HEADER}
            writer.writerow(safe_row)


def make_meta_rows(run_id: str, **meta: Any) -> list[Dict[str, str]]:
    rows = [{"section": "meta", "key": "run_id", "value": run_id, "design": ""}]
    for k, v in meta.items():
        rows.append({"section": "meta", "key": str(k), "value": str(v), "design": ""})
    return rows


def make_metric_rows(**metrics: Any) -> list[Dict[str, str]]:
    return [{"section": "metric", "key": str(k), "value": str(v), "design": ""} for k, v in metrics.items()]


def make_design_row(design: Any) -> Dict[str, str]:
    return {"section": "design", "key": "", "value": "", "design": _serialize_design(design)}


def parse_int_list(csv_or_repeat: Optional[str], fallback_single: Optional[int], n_factors: int, name: str) -> List[int]:
    """
    Parse a comma-separated list for --kx-list/--kb-list/--deg-list.
    If not provided, replicate fallback_single n_factors times.
    """
    if csv_or_repeat:
        try:
            vals = [int(x.strip()) for x in csv_or_repeat.split(",") if x.strip() != ""]
        except Exception:
            raise typer.BadParameter(f"Invalid list for --{name}: {csv_or_repeat}")
        if len(vals) != n_factors:
            raise typer.BadParameter(f"--{name} must have exactly {n_factors} integers (got {len(vals)}).")
        return vals
    if fallback_single is None:
        raise typer.BadParameter(f"--{name} is required (or provide --{name.replace('-list','')} to replicate).")
    return [fallback_single] * n_factors


def build_bspline(K: int, degree: int) -> BSplineBasis:
    """Your BSplineBasis signature: (degree, total_knots_num).  K = degree + total_knots_num - 1."""
    total_knots_num = max(2, K - degree + 1)
    return BSplineBasis(degree=degree, total_knots_num=total_knots_num)


def safe_report(model_obj: Any, Gamma: Any) -> Optional[float]:
    """
    Ask the model for its official 'reported' value on Gamma using existing reporter APIs.
    No math here; we just try common reporter names.
    """
    cand_names = ["report_num", "report_value", "report"]
    for name in cand_names:
        fn = getattr(model_obj, name, None)
        if callable(fn):
            try:
                val = fn(Gamma)
                if hasattr(val, "numpy"):
                    val = val.numpy()
                try:
                    return float(val)
                except Exception:
                    try:
                        return float(val.reshape(()))
                    except Exception:
                        pass
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Root callback
# ---------------------------------------------------------------------------
@app.callback()
def _root_callback():
    pass


# ---------------------------------------------------------------------------
# Command: run
# ---------------------------------------------------------------------------
@app.command("run")
def run(
    # Common
    model: str = typer.Option(..., help="Model type: sos | sof | fof"),
    criterion: str = typer.Option("A", help="Design criterion to optimize: A or D"),
    runs: int = typer.Option(..., help="Number of runs in the design"),
    latent_dim: int = typer.Option(..., help="Latent dimension for the autoencoder"),
    epochs: int = typer.Option(..., help="Training epochs for NBDO"),
    n_calls: int = typer.Option(..., help="Bayesian optimization calls"),
    patience: int = typer.Option(100, help="Early stopping patience (epochs)"),
    batch_size: int = typer.Option(256, help="Training batch size"),
    seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, help="Progress bars only (Rich). Model/BO logs stay silent."),
    tag: str | None = typer.Option(None, help="Optional tag appended to filename"),
    results_dir: str = typer.Option("results", help="Directory to save the master TSV"),

    # SoS
    kx: Optional[int] = typer.Option(None, help="[SoS/FoF/SoF] For SoS: number of scalar factors (Kx)."),
    sos_order: int = typer.Option(2, help="[SoS] Polynomial order (if applicable)"),

    # Functional models (SoF/FoF)
    factors: int = typer.Option(1, help="[SoF/FoF] Number of functional predictors"),
    kx_list: Optional[str] = typer.Option(None, help="[SoF/FoF] Comma list of Kx per predictor, e.g. '5,4'"),
    kb: Optional[int] = typer.Option(None, help="[SoF/FoF] If given, replicate to all predictors"),
    kb_list: Optional[str] = typer.Option(None, help="[SoF/FoF] Comma list of Kb per predictor, e.g. '6,3'"),
    deg: Optional[int] = typer.Option(2, help="[SoF/FoF] If given, replicate to all predictors"),
    deg_list: Optional[str] = typer.Option(None, help="[SoF/FoF] Comma list of degrees per predictor, e.g. '2,3'"),
    intercept: bool = typer.Option(True, help="[SoF/FoF] Include intercept"),

    # Penalties
    lambda_s: float = typer.Option(0.0, help="[SoF/FoF] Penalty weight on s-direction (β-basis)"),
    lambda_t: float = typer.Option(0.0, help="[FoF] Penalty weight on t-direction (response basis)"),

    # Training set
    num_designs: int = typer.Option(1000, help="Training set size for NBDO (compute_train_set)"),
):
    """
    Run the chosen model + NBDO + BO once and save a single TSV with:
    - meta rows (args, timings from NBDO)
    - metric rows (reported A and D on the final design)
    - one design row (full design JSON in 'design' column)
    """
    import logging
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.FATAL)
    except Exception:
        pass

    # init env (determinism/devices as your tf_env dictates)
    tf_initialize()

    ensure_dir(results_dir)
    run_id = make_run_id(tag)
    out_path = Path(results_dir) / f"{run_id}.tsv"  # keep the consistent suffix

    model_key = model.lower().strip()
    criterion = criterion.upper().strip()
    if criterion not in {"A", "D"}:
        raise typer.BadParameter("criterion must be 'A' or 'D'")

    nbdo: Optional[NBDO] = None  # for safe cleanup in finally
    try:
        # ------------------ build & run (mute OS-level TF C++ stderr) ------------------
        if model_key == "sos":
            if kx is None or kx <= 0:
                raise typer.BadParameter("[SoS] --kx must be a positive integer (number of factors).")

            with _mute_os_stderr():
                mdl = ScalarOnScalarModel(Kx=int(kx), criterion=criterion, order=int(sos_order))
                nbdo = NBDO(model=mdl, latent_dim=latent_dim, seed=seed, verbose=verbose)
                nbdo.compute_train_set(num_designs=num_designs, runs=runs)
                nbdo.fit(epochs=epochs, patience=patience, batch_size=batch_size)
                rep_value, Gamma = nbdo.optimize(n_calls=n_calls)
                Gamma = np.round(Gamma, 3)
                
                # evaluate reported A & D on final design
                A_rep = safe_report(ScalarOnScalarModel(Kx=int(kx), criterion="A", order=int(sos_order)), Gamma)
                D_rep = safe_report(ScalarOnScalarModel(Kx=int(kx), criterion="D", order=int(sos_order)), Gamma)

        elif model_key == "sof":
            # parse lists for Kx/Kb/deg across predictors
            Kx_list = parse_int_list(kx_list, kx, factors, "kx-list")
            Kb_list = parse_int_list(kb_list, kb, factors, "kb-list")
            Deg_list = parse_int_list(deg_list, deg, factors, "deg-list")

            pairs = []
            for Kx_i, Kb_i, deg_i in zip(Kx_list, Kb_list, Deg_list):
                bx = build_bspline(Kx_i, deg_i)
                bb = build_bspline(Kb_i, deg_i)
                pairs.append((bx, bb))

            with _mute_os_stderr():
                mdl_kwargs = dict(basis_pairs=pairs, criterion=criterion, intercept=intercept)
                if lambda_s and lambda_s > 0:
                    # your SoF API used lambda_penalty in earlier notes
                    mdl_kwargs["lambda_penalty"] = float(lambda_s)

                mdl = ScalarOnFunctionModel(**mdl_kwargs)
                nbdo = NBDO(model=mdl, latent_dim=latent_dim, seed=seed, verbose=verbose)
                nbdo.compute_train_set(num_designs=num_designs, runs=runs)
                nbdo.fit(epochs=epochs, patience=patience, batch_size=batch_size)
                rep_value, Gamma = nbdo.optimize(n_calls=n_calls)
                Gamma = np.round(Gamma, 3)

                A_rep = safe_report(ScalarOnFunctionModel(basis_pairs=pairs, criterion="A", intercept=intercept,
                                                          **({"lambda_penalty": float(lambda_s)} if lambda_s > 0 else {})), Gamma)
                D_rep = safe_report(ScalarOnFunctionModel(basis_pairs=pairs, criterion="D", intercept=intercept,
                                                          **({"lambda_penalty": float(lambda_s)} if lambda_s > 0 else {})), Gamma)

        elif model_key == "fof":
            Kx_list = parse_int_list(kx_list, kx, factors, "kx-list")
            Kb_list = parse_int_list(kb_list, kb, factors, "kb-list")
            Deg_list = parse_int_list(deg_list, deg, factors, "deg-list")

            pairs = []
            for Kx_i, Kb_i, deg_i in zip(Kx_list, Kb_list, Deg_list):
                bx = build_bspline(Kx_i, deg_i)
                bb = build_bspline(Kb_i, deg_i)
                pairs.append((bx, bb))

            with _mute_os_stderr():
                response_basis = FourierBasis(n_harmonics=3, include_constant=True) if (lambda_t and lambda_t > 0) else None

                mdl = FunctionOnFunctionModel(
                    basis_pairs=pairs,
                    criterion=criterion,
                    intercept=intercept,
                    response_basis=response_basis,
                    lambda_s=(float(lambda_s) if lambda_s > 0 else None),
                    lambda_t=(float(lambda_t) if lambda_t > 0 else None),
                )

                nbdo = NBDO(model=mdl, latent_dim=latent_dim, seed=seed, verbose=verbose)
                nbdo.compute_train_set(num_designs=num_designs, runs=runs)
                nbdo.fit(epochs=epochs, patience=patience, batch_size=batch_size)
                rep_value, Gamma = nbdo.optimize(n_calls=n_calls)
                Gamma = np.round(Gamma, 3)

                def _fof_with(crit: str):
                    return FunctionOnFunctionModel(
                        basis_pairs=pairs,
                        criterion=crit,
                        intercept=intercept,
                        response_basis=response_basis,
                        lambda_s=(float(lambda_s) if lambda_s > 0 else None),
                        lambda_t=(float(lambda_t) if lambda_t > 0 else None),
                    )

                A_rep = safe_report(_fof_with("A"), Gamma)
                D_rep = safe_report(_fof_with("D"), Gamma)

        else:
            raise typer.BadParameter("model must be one of: sos, sof, fof")

        # ------------------ timings from NBDO ------------------
        nbdo_time_sec = (nbdo.time_fit_s if nbdo and nbdo.time_fit_s is not None else 0.0)
        bo_time_sec = (nbdo.time_bo_s if nbdo and nbdo.time_bo_s is not None else 0.0)
        total_time_sec = (nbdo.time_total_s if nbdo and nbdo.time_total_s is not None else (nbdo_time_sec + bo_time_sec))

        # ------------------ write single master TSV ------------------
        meta_rows = make_meta_rows(
            run_id,
            status="success",
            optimized=criterion,
            model=model_key,
            runs=runs,
            latent_dim=latent_dim,
            epochs=epochs,
            n_calls=n_calls,
            patience=patience,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            nbdo_time_sec=f"{nbdo_time_sec:.6f}",
            bo_time_sec=f"{bo_time_sec:.6f}",
            total_time_sec=f"{total_time_sec:.6f}",
        )
        metric_rows = make_metric_rows(
            A_opt="NA" if A_rep is None else f"{A_rep:.12g}",
            D_opt="NA" if D_rep is None else f"{D_rep:.12g}",
        )
        design_row = make_design_row(Gamma)

        append_master_tsv(out_path, [*meta_rows, *metric_rows, design_row])
        typer.echo(f"Wrote master TSV → {out_path}")

    except Exception as e:
        # Best-effort error record
        append_master_tsv(out_path, make_meta_rows(run_id, status="error", error_message=str(e)))
        raise
    finally:
        try:
            if nbdo is not None:
                nbdo.clear_memory()
            else:
                from tensorflow.keras.backend import clear_session
                clear_session()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Allow `python -m cli.app` or `python cli/app.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
