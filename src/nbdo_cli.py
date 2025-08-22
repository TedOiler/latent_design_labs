from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich import print as rprint

from models.sos import ScalarOnScalarModel
from optimizers.nbdo import NBDO


app = typer.Typer(help="NBDO — minimal CLI to generate and dump train/val sets")


def _save_csv(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, arr, delimiter=",", fmt="%.6f")


def _timestamp_dir(root: Path, prefix: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = root / f"{run_id}_{prefix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.command("dump")
def dump(
    runs: int = typer.Option(..., "--runs", "-n", help="number of experimental runs (n)"),
    kx: int = typer.Option(..., "--kx", "-k", help="number of features (Kx) for the model"),
    num_designs: int = typer.Option(..., "--num-designs", "-L", help="number of candidate designs (L)"),
    result: str = typer.Option("", "--result", "-r", help="any user-provided string to store in result.json"),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed"),
    out_dir: Path = typer.Option(Path("runs"), "--out-dir", help="folder to save files"),
) -> None:
    """
    Generate train/validation sets using your NBDO.compute_train_set and save them.
    No extra computation: result.json includes only your --result string and metadata.
    """
    model = ScalarOnScalarModel(Kx=kx)
    nbdo = NBDO(model=model, latent_dim=2)

    nbdo.compute_train_set(num_designs=num_designs, runs=runs, random_state=seed)

    run_dir = _timestamp_dir(out_dir, "dump")
    _save_csv(run_dir / "train_set.csv", nbdo.train_set)
    _save_csv(run_dir / "val_set.csv", nbdo.val_set)

    meta = {
        "user_result": result,          # exactly what you passed in
        "runs": runs,
        "kx": kx,
        "num_designs": num_designs,
        "seed": seed,
        "input_dim": getattr(nbdo, "input_dim", None),
        "train_shape": list(nbdo.train_set.shape),
        "val_shape": list(nbdo.val_set.shape),
    }
    (run_dir / "result.json").write_text(json.dumps(meta, indent=2))

    rprint(f"[bold green]✓ Saved[/] {run_dir/'train_set.csv'} and {run_dir/'val_set.csv'}")
    rprint(f"[bold]Meta[/]: {run_dir/'result.json'}")

if __name__ == "__main__":
    app()