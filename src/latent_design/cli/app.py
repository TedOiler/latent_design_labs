# cli/app.py (v2 – simplified shape)
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Any, Dict
import subprocess, shlex

import os, sys, contextlib
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

@contextlib.contextmanager
def _mute_os_stderr():
    """
    Temporarily redirect the OS-level stderr (fd=2) to /dev/null.
    Useful to silence noisy TensorFlow C++ messages.
    Python exceptions will still raise, but their tracebacks will be hidden
    while the context is active—so keep the wrapped blocks small.
    """
    try:
        orig_fd = sys.stderr.fileno()
    except Exception:
        # e.g., running without a real TTY — just do nothing
        yield
        return

    saved = os.dup(orig_fd)            # duplicate current stderr
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, orig_fd)      # replace stderr with /dev/null
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, orig_fd)        # restore original stderr
        os.close(saved)


import json
import importlib.resources as ir
import typer

from latent_design.utils.tf_env import initialize as tf_initialize
from latent_design.bases.bspline import BSplineBasis
from latent_design.bases.fourier import FourierBasis
from latent_design.bases.polynomial import PolynomialBasis
from latent_design.models.sos import ScalarOnScalarModel
from latent_design.models.sof import ScalarOnFunctionModel
from latent_design.models.fof import FunctionOnFunctionModel
from latent_design.optimizers.nbdo import NBDO
from latent_design.diagnostics.core import info_matrix, eigen_spectrum, condition_number, leverage_diag

app = typer.Typer(no_args_is_help=True, add_completion=False)

# ---------- enums & dataclasses ----------
class Criterion(str, Enum):
    A = "A"
    D = "D"

class Family(str, Enum):
    bspline = "bspline"
    fourier = "fourier"
    poly = "poly"
    none = "none"

@dataclass
class BasisSpec:
    family: Family
    degree: Optional[int] = None       # bspline/poly
    knots: Optional[int] = None        # bspline
    harmonics: Optional[int] = None    # fourier
    include_const: bool = True         # fourier

@dataclass
class PredictorSpec:
    x: BasisSpec
    b: BasisSpec

@dataclass
class ResponseSpec:
    y: Optional[BasisSpec] = None      # FoF only; None means no response basis

@dataclass
class Penalties:
    lambda_s: float = 0.0
    lambda_t: float = 0.0

@dataclass
class RunSpec:
    model: str
    criterion: Criterion
    runs: int
    latent_dim: int
    epochs: int
    n_calls: int
    patience: int = 100
    batch_size: int = 256
    seed: int = 42
    verbose: bool = False
    tag: Optional[str] = None
    results_dir: str = "results"
    num_designs: int = 1000
    intercept: bool = True
    kx: Optional[int] = None
    sos_order: int = 1
    predictors: List[PredictorSpec] = field(default_factory=list)
    response: ResponseSpec = field(default_factory=ResponseSpec)
    penalties: Penalties = field(default_factory=Penalties)


# ---------- tiny IO helpers ----------
def _read_text_from_path_or_resource(path: Path, resource_pkg: str, resource_subdir: str | None = None) -> str:
    """
    Read text from a filesystem path if it exists; otherwise, try to read
    from package resources under `resource_pkg[/resource_subdir]`.
    """
    p = Path(path)
    if p.exists():
        return p.read_text()
    name = p.name
    try:
        base = ir.files(resource_pkg)
        if resource_subdir:
            base = base.joinpath(resource_subdir)
        data = (base / name).read_text(encoding="utf-8")
        return data
    except Exception as e:
        raise typer.BadParameter(f"Config not found: {path}")


def read_config(path: Path) -> Dict[str, Any]:
    text = _read_text_from_path_or_resource(path, "latent_design.cli", resource_subdir="cfgs")
    return json.loads(text)

def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    # overrides format: key=val (supports shallow keys used in practice)
    for ov in overrides:
        if "=" not in ov:
            raise typer.BadParameter(f"Invalid override '{ov}', expected key=value")
        k, v = ov.split("=", 1)
        # naive coercion (int/float/bool), else str
        if v.isdigit():
            val: Any = int(v)
        else:
            try:
                val = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    val = (v.lower() == "true")
                else:
                    val = v
        cfg[k] = val
    return cfg

# ---------- builders ----------
def build_basis(b: BasisSpec):
    if b.family == Family.fourier:
        return FourierBasis(n_harmonics=int(b.harmonics or 3), include_constant=bool(b.include_const))
    if b.family == Family.bspline:
        deg = int(b.degree or 3)
        knots = int(b.knots or 12)
        return BSplineBasis(degree=deg, total_knots_num=knots)
    if b.family == Family.poly:
        return PolynomialBasis(degree=int(b.degree or 3))
    if b.family == Family.none:
        return None
    raise typer.BadParameter(f"Unknown basis family: {b.family}")

def build_model(spec: RunSpec):
    if spec.model == "sos":
        if spec.kx is None:
            raise typer.BadParameter("For 'sos', please provide 'kx' in the config (or -o kx=...).")
        return ScalarOnScalarModel(
            Kx=int(spec.kx),
            criterion=spec.criterion.value,
            order=int(spec.sos_order),
        )
    if spec.model == "sof":
        pairs = [(build_basis(p.x), build_basis(p.b)) for p in (spec.predictors or [])]
        kwargs = dict(basis_pairs=pairs, criterion=spec.criterion.value, intercept=spec.intercept)
        if spec.penalties.lambda_s > 0:
            kwargs["lambda_penalty"] = float(spec.penalties.lambda_s)
        return ScalarOnFunctionModel(**kwargs)
    if spec.model == "fof":
        pairs = [(build_basis(p.x), build_basis(p.b)) for p in (spec.predictors or [])]
        rbasis = build_basis(spec.response.y) if spec.response and spec.response.y else None
        return FunctionOnFunctionModel(
            basis_pairs=pairs,
            criterion=spec.criterion.value,
            intercept=spec.intercept,
            response_basis=rbasis,
            lambda_s=(spec.penalties.lambda_s or None),
            lambda_t=(spec.penalties.lambda_t or None),
        )
    raise typer.BadParameter("model must be one of: sos, sof, fof")

# ---------- runner ----------
def run_once(spec: RunSpec):

    with _mute_os_stderr():
        tf_initialize()
        mdl = build_model(spec)

        nbdo = NBDO(model=mdl, latent_dim=spec.latent_dim, seed=spec.seed, verbose=spec.verbose)
        nbdo.compute_train_set(num_designs=spec.num_designs, runs=spec.runs)
        nbdo.fit(epochs=spec.epochs, patience=spec.patience, batch_size=spec.batch_size)
        report_value, Gamma = nbdo.optimize(n_calls=spec.n_calls)

        # diagnostics (unchanged)
        M = info_matrix(mdl, Gamma)
        eig, _ = eigen_spectrum(M)
        kappa2 = condition_number(M)
        h = leverage_diag(mdl, Gamma)
        A_rep, D_rep = None, None
        try:
            if spec.model == "sos":
                if spec.kx is None:
                    raise typer.BadParameter("For 'sos' reporters, 'kx' must be set (use -o kx=...).")
                A_rep = ScalarOnScalarModel(
                    Kx=int(spec.kx),
                    criterion="A",
                    order=int(spec.sos_order),
                ).report(Gamma)
                D_rep = ScalarOnScalarModel(
                    Kx=int(spec.kx),
                    criterion="D",
                    order=int(spec.sos_order),
                ).report(Gamma)

            elif spec.model == "sof":
                pairs = [(build_basis(p.x), build_basis(p.b)) for p in (spec.predictors or [])]
                kwargs = dict(basis_pairs=pairs, intercept=spec.intercept)
                if spec.penalties.lambda_s > 0:
                    kwargs["lambda_penalty"] = float(spec.penalties.lambda_s)

                A_rep = ScalarOnFunctionModel(criterion="A", **kwargs).report(Gamma)
                D_rep = ScalarOnFunctionModel(criterion="D", **kwargs).report(Gamma)

            elif spec.model == "fof":
                pairs = [(build_basis(p.x), build_basis(p.b)) for p in (spec.predictors or [])]
                rbasis = build_basis(spec.response.y) if spec.response and spec.response.y else None
                kwargs = dict(basis_pairs=pairs, intercept=spec.intercept, response_basis=rbasis)
                if spec.penalties.lambda_s > 0:
                    kwargs["lambda_s"] = float(spec.penalties.lambda_s)
                if spec.penalties.lambda_t > 0:
                    kwargs["lambda_t"] = float(spec.penalties.lambda_t)

                A_rep = FunctionOnFunctionModel(criterion="A", **kwargs).report(Gamma)
                D_rep = FunctionOnFunctionModel(criterion="D", **kwargs).report(Gamma)
        except Exception:
            pass

    # write (keep your existing TSV writer, or output JSONL)
    out_dir = Path(spec.results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    row = dict(
        model=spec.model, criterion=spec.criterion.value, runs=spec.runs, tag=spec.tag or "",
        A_opt=(None if A_rep is None else float(A_rep)),
        D_opt=(None if D_rep is None else float(D_rep)),
        lambda_min=float(eig[-1]), lambda_max=float(eig[0]), kappa2=float(kappa2),
        leverage_min=float(h.min()), leverage_max=float(h.max()), leverage_mean=float(h.mean()),
        nbdo_time_s=nbdo.time_fit_s, bo_time_s=nbdo.time_bo_s, total_time_s=nbdo.time_total_s,
        design=Gamma.tolist(),
    )
    with out_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    typer.echo(f"Wrote → {out_path}")

# ---------- CLI commands ----------
@app.command("run")
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to JSON config (single or sweep)"),
    override: List[str] = typer.Option(None, "--override", "-o", help="Shallow key=val overrides"),
):
    cfg = read_config(config)

    # ---- helpers (local to keep your file unchanged elsewhere) ----
    def load_basis(d: Dict[str, Any] | None) -> Optional[BasisSpec]:
        if not d:
            return None
        return BasisSpec(
            family=Family(d.get("family", "bspline")),
            degree=d.get("degree"),
            knots=d.get("knots"),
            harmonics=d.get("harmonics"),
            include_const=bool(d.get("include_const", True)),
        )

    def to_spec(d: Dict[str, Any]) -> RunSpec:
        preds: List[PredictorSpec] = []
        for pd in d.get("predictors", []):
            preds.append(PredictorSpec(x=load_basis(pd["x"]), b=load_basis(pd["b"])))  # type: ignore

        resp = d.get("response", {})
        response = ResponseSpec(y=load_basis(resp.get("y"))) if resp else ResponseSpec(None)

        return RunSpec(
            model=d["model"],
            criterion=Criterion(d.get("criterion", "A")),
            runs=int(d["runs"]),
            latent_dim=int(d["latent_dim"]),
            epochs=int(d["epochs"]),
            n_calls=int(d["n_calls"]),
            patience=int(d.get("patience", 100)),
            batch_size=int(d.get("batch_size", 256)),
            seed=int(d.get("seed", 42)),
            verbose=bool(d.get("verbose", False)),
            tag=d.get("tag"),
            results_dir=d.get("results_dir", "results"),
            num_designs=int(d.get("num_designs", 1000)),
            intercept=bool(d.get("intercept", True)),

            # NEW:
            kx=(int(d["kx"]) if "kx" in d else int(d["Kx"]) if "Kx" in d else None),
            sos_order=int(d.get("sos_order", d.get("order", 1))),

            predictors=preds,
            response=response,
            penalties=Penalties(lambda_s=float(d.get("lambda_s", 0.0)),
                                lambda_t=float(d.get("lambda_t", 0.0))),
        )

    # parse overrides into a dict once; then merge appropriately per case
    overrides_dict: Dict[str, Any] = {}
    if override:
        overrides_dict = apply_overrides({}, override)  # reuse your existing coercion

    # ---- Case A: top-level list -> many experiments ----
    if isinstance(cfg, list):
        for i, exp in enumerate(cfg):
            merged = {**exp, **overrides_dict} if overrides_dict else exp
            spec = to_spec(merged)
            if not spec.tag:
                spec.tag = f"exp{i}"
            run_once(spec)
        return

    # ---- Case B: sweep dict with defaults/experiments ----
    if isinstance(cfg, dict) and "experiments" in cfg:
        defaults = cfg.get("defaults", {})
        sweep_tag = cfg.get("sweep_tag")
        if overrides_dict:
            defaults = {**defaults, **overrides_dict}
        exps = cfg["experiments"]
        if not isinstance(exps, list):
            raise typer.BadParameter("'experiments' must be a list")
        for i, exp in enumerate(exps):
            merged = {**defaults, **exp}
            # auto-tag if requested and not present on the experiment
            if sweep_tag and not merged.get("tag"):
                merged["tag"] = f"{sweep_tag}_exp{i}"
            spec = to_spec(merged)
            run_once(spec)
        return

    # ---- Case C: single experiment dict (original behavior) ----
    if isinstance(cfg, dict):
        single = {**cfg, **overrides_dict} if overrides_dict else cfg
        spec = to_spec(single)
        run_once(spec)
        return

    raise typer.BadParameter("Config must be a dict (single/sweep) or a list of experiment dicts.")

@app.command("batch")
def batch(
    file: Path = typer.Option(..., "--file", "-f", help="Path to a cmds.txt file"),
    stop_on_error: bool = typer.Option(True, help="Stop at the first failing command"),
):
    """
    Run multiple CLI commands from a text file, one per line.
    Lines starting with '#' or blank lines are ignored.
    Each line is executed in a fresh subprocess to avoid TF session state.
    """
    try:
        text = _read_text_from_path_or_resource(file, "latent_design.cli", resource_subdir="cfgs")
    except Exception:
        # fall back to filesystem error if nothing else worked
        text = Path(file).read_text()
    lines = text.splitlines()

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        typer.echo(f"→ [{lineno}] {line}")
        # Use the shell so you can write lines exactly like you would in the terminal:
        # e.g., `nbdo run -c fof.json -o tag=paper1 -o runs=12`
        result = subprocess.run(line, shell=True)
        if result.returncode != 0:
            typer.echo(f"✗ Command {lineno} failed with exit code {result.returncode}")
            if stop_on_error:
                raise typer.Exit(result.returncode)

    typer.echo("✔ Batch complete.")

def main():
    app()

if __name__ == "__main__":
    main()
