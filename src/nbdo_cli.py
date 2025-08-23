from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import typer
from rich import print as rprint
import subprocess

from models.sos import ScalarOnScalarModel
from optimizers.nbdo import NBDO


app = typer.Typer(help="NBDO — CLI for hyperparameter analysis and experiment management")


def _save_csv(path: Path, arr: np.ndarray) -> None:
    """Save numpy array to CSV with 6 decimal precision."""
    np.savetxt(path, arr, delimiter=",", fmt="%.6f")


def _timestamp_dir(root: Path, prefix: str) -> Path:
    """Create timestamped directory for experiment results."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = root / f"{run_id}_{prefix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_training_history(history, path: Path) -> None:
    """Save training history metrics to JSON."""
    if history is not None:
        # Extract key metrics from training history
        metrics = {
            "loss": [float(x) for x in history.history.get("loss", [])],
            "val_loss": [float(x) for x in history.history.get("val_loss", [])] if "val_loss" in history.history else None,
            "lr": [float(x) for x in history.history.get("lr", [])] if "lr" in history.history else None,
        }
        path.write_text(json.dumps(metrics, indent=2))


@app.command("dump")
def dump(
    runs: int = typer.Option(..., "--runs", "-n", help="number of experimental runs (n)"),
    kx: int = typer.Option(..., "--kx", "-k", help="number of features (Kx) for the model"),
    num_designs: int = typer.Option(..., "--num-designs", "-L", help="number of candidate designs (L)"),
    result: str = typer.Option("", "--result", "-r", help="any user-provided string to store in result.json"),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir", help="folder to save files"),
) -> None:
    """
    Generate train/validation sets using NBDO.compute_train_set and save them.
    No extra computation: result.json includes only your --result string and metadata.
    """
    model = ScalarOnScalarModel(Kx=kx)
    nbdo = NBDO(model=model, latent_dim=2)

    nbdo.compute_train_set(num_designs=num_designs, runs=runs, random_state=seed)

    run_dir = _timestamp_dir(out_dir, "dump")
    _save_csv(run_dir / "train_set.csv", nbdo.train_set)

    meta = {
        "user_result": result,          # exactly what you passed in
        "runs": runs,
        "kx": kx,
        "num_designs": num_designs,
        "seed": seed,
        "input_dim": getattr(nbdo, "input_dim", None),
        "train_shape": list(nbdo.train_set.shape),
    }
    (run_dir / "result.json").write_text(json.dumps(meta, indent=2))

    rprint(f"[bold green]✓ Saved[/] {run_dir/'train_set.csv'}")
    rprint(f"[bold]Meta[/]: {run_dir/'result.json'}")


@app.command("experiment")
def experiment(
    runs: int = typer.Option(..., "--runs", "-n", help="number of experimental runs (n)"),
    kx: int = typer.Option(..., "--kx", "-k", help="number of features (Kx) for the model"),
    num_designs: int = typer.Option(..., "--num-designs", "-L", help="number of candidate designs (L)"),
    latent_dim: int = typer.Option(..., "--latent-dim", "-d", help="latent space dimensionality"),
    max_layers: Optional[int] = typer.Option(None, "--max-layers", help="maximum number of autoencoder layers"),
    alpha: float = typer.Option(0.0, "--alpha", help="LeakyReLU alpha parameter"),
    latent_act: str = typer.Option("tanh", "--latent-act", help="latent layer activation"),
    output_act: str = typer.Option("tanh", "--output-act", help="output layer activation"),
    epochs: int = typer.Option(100, "--epochs", help="training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", help="training batch size"),
    patience: int = typer.Option(50, "--patience", help="early stopping patience"),
    n_calls: int = typer.Option(30, "--n-calls", help="BO optimization calls"),
    acq_func: str = typer.Option("EI", "--acq-func", help="acquisition function"),
    acq_optimizer: str = typer.Option("sampling", "--acq-optimizer", help="acquisition optimizer"),
    n_random_starts: int = typer.Option(5, "--n-random-starts", help="BO random starts"),
    result: str = typer.Option("", "--result", "-r", help="user-provided result identifier"),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir", help="folder to save files"),
) -> None:
    """
    Run complete NBDO experiment: train autoencoder and optimize design.
    Saves comprehensive results for hyperparameter analysis.
    """
    # Create model and optimizer
    model = ScalarOnScalarModel(Kx=kx)
    nbdo = NBDO(
        model=model, 
        latent_dim=latent_dim,
        max_layers=max_layers,
        alpha=alpha,
        latent_space_act=latent_act,
        output_layer_act=output_act,
        seed=seed
    )

    # Generate training data
    nbdo.compute_train_set(num_designs=num_designs, runs=runs, random_state=seed)
    
    # Train autoencoder
    history = nbdo.fit(
        epochs=epochs,
        batch_size=batch_size,
        patience=patience
    )
    
    # Optimize design
    optimal_criterion, optimal_design = nbdo.optimize(
        n_calls=n_calls,
        acq_func=acq_func,
        acq_optimizer=acq_optimizer,
        n_random_starts=n_random_starts,
        verbose=False
    )

    # Create results directory
    run_dir = _timestamp_dir(out_dir, "experiment")
    
    # Save training data
    _save_csv(run_dir / "train_set.csv", nbdo.train_set)
    
    # Save optimal design
    _save_csv(run_dir / "optimal_design.csv", optimal_design)
    
    # Save training history
    _save_training_history(history, run_dir / "training_history.json")
    
    # Save comprehensive metadata
    meta = {
        "user_result": result,
        "model_params": {
            "runs": runs,
            "kx": kx,
            "num_designs": num_designs,
        },
        "nbdo_params": {
            "latent_dim": latent_dim,
            "max_layers": max_layers,
            "alpha": alpha,
            "latent_space_act": latent_act,
            "output_layer_act": output_act,
            "seed": seed,
        },
        "training_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
        },
        "optimization_params": {
            "n_calls": n_calls,
            "acq_func": acq_func,
            "acq_optimizer": acq_optimizer,
            "n_random_starts": n_random_starts,
        },
        "results": {
            "input_dim": nbdo.input_dim,
            "num_layers": nbdo.num_layers,
            "hidden_widths": getattr(nbdo, "hidden_widths", None),
            "optimal_criterion": float(optimal_criterion),
            "optimal_design_shape": list(optimal_design.shape),
            "train_shape": list(nbdo.train_set.shape),
            "final_loss": float(history.history["loss"][-1]) if history else None,
            "epochs_trained": len(history.history["loss"]) if history else None,
        },
        "search_history": {
            "latent_vars": [list(x) for x in nbdo.search_history] if nbdo.search_history else None,
            "objective_vals": [float(x) for x in nbdo.eval_history] if nbdo.eval_history else None,
        }
    }
    
    (run_dir / "result.json").write_text(json.dumps(meta, indent=2))
    
    # Clean up memory
    nbdo.clear_memory()

    rprint(f"[bold green]✓ Experiment completed[/]")
    rprint(f"[bold]Results saved to[/]: {run_dir}")
    rprint(f"[bold]Optimal criterion[/]: {optimal_criterion:.6f}")
    rprint(f"[bold]Training epochs[/]: {len(history.history['loss'])}")


@app.command("hyperparameter-sweep")
def hyperparameter_sweep(
    config_file: Path = typer.Argument(..., help="JSON config file with hyperparameter ranges"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir", help="folder to save results"),
    seed_base: int = typer.Option(42, "--seed-base", help="base seed for experiments"),
) -> None:
    """
    Run hyperparameter sweep based on JSON configuration file.
    Useful for systematic exploration of parameter space.
    """
    if not config_file.exists():
        raise typer.BadParameter(f"Config file {config_file} does not exist")
    
    config = json.loads(config_file.read_text())
    
    # Extract parameter ranges
    runs_range = config.get("runs", [8, 12, 16])
    kx_range = config.get("kx", [3, 5, 7])
    latent_dim_range = config.get("latent_dim", [2, 4, 6])
    max_layers_range = config.get("max_layers", [None, 3, 5])
    alpha_range = config.get("alpha", [0.0, 0.1, 0.2])
    epochs_range = config.get("epochs", [50, 100, 200])
    batch_size_range = config.get("batch_size", [16, 32, 64])
    patience_range = config.get("patience", [20, 50, 100])
    n_calls_range = config.get("n_calls", [20, 50, 100])
    
    total_experiments = (
        len(runs_range) * len(kx_range) * len(latent_dim_range) * 
        len(max_layers_range) * len(alpha_range) * len(epochs_range) * 
        len(batch_size_range) * len(patience_range) * len(n_calls_range)
    )
    
    rprint(f"[bold]Starting hyperparameter sweep[/] with {total_experiments} experiments")
    
    experiment_count = 0
    for runs in runs_range:
        for kx in kx_range:
            for latent_dim in latent_dim_range:
                for max_layers in max_layers_range:
                    for alpha in alpha_range:
                        for epochs in epochs_range:
                            for batch_size in batch_size_range:
                                for patience in patience_range:
                                    for n_calls in n_calls_range:
                                        experiment_count += 1
                                        seed = seed_base + experiment_count
                                        
                                        rprint(f"[cyan]▶ Experiment {experiment_count}/{total_experiments}[/]")
                                        rprint(f"  runs={runs}, kx={kx}, latent_dim={latent_dim}, max_layers={max_layers}")
                                        rprint(f"  alpha={alpha}, epochs={epochs}, batch_size={batch_size}")
                                        rprint(f"  patience={patience}, n_calls={n_calls}, seed={seed}")
                                        
                                        try:
                                            # Run experiment
                                            model = ScalarOnScalarModel(Kx=kx)
                                            nbdo = NBDO(
                                                model=model,
                                                latent_dim=latent_dim,
                                                max_layers=max_layers,
                                                alpha=alpha,
                                                seed=seed
                                            )
                                            
                                            nbdo.compute_train_set(num_designs=64, runs=runs, random_state=seed)
                                            history = nbdo.fit(epochs=epochs, batch_size=batch_size, patience=patience)
                                            optimal_criterion, optimal_design = nbdo.optimize(n_calls=n_calls, verbose=False)
                                            
                                            # Save results
                                            run_dir = _timestamp_dir(out_dir, f"sweep_{experiment_count:04d}")
                                            
                                            _save_csv(run_dir / "optimal_design.csv", optimal_design)
                                            _save_training_history(history, run_dir / "training_history.json")
                                            
                                            # Save metadata
                                            meta = {
                                                "experiment_id": experiment_count,
                                                "hyperparameters": {
                                                    "runs": runs,
                                                    "kx": kx,
                                                    "latent_dim": latent_dim,
                                                    "max_layers": max_layers,
                                                    "alpha": alpha,
                                                    "epochs": epochs,
                                                    "batch_size": batch_size,
                                                    "patience": patience,
                                                    "n_calls": n_calls,
                                                    "seed": seed,
                                                },
                                                "results": {
                                                    "optimal_criterion": float(optimal_criterion),
                                                    "final_loss": float(history.history["loss"][-1]),
                                                    "epochs_trained": len(history.history["loss"]),
                                                }
                                            }
                                            
                                            (run_dir / "result.json").write_text(json.dumps(meta, indent=2))
                                            
                                            rprint(f"  [green]✓ Criterion: {optimal_criterion:.6f}[/]")
                                            
                                            nbdo.clear_memory()
                                            
                                        except Exception as e:
                                            rprint(f"  [red]✗ Failed: {str(e)}[/]")
                                            continue
    
    rprint(f"[bold green]✓ Hyperparameter sweep completed[/]")
    rprint(f"[bold]Results saved to[/]: {out_dir}")


@app.command("batch")
def batch(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Text file with one full command per line"),
    stop_on_error: bool = typer.Option(False, help="Stop at first failing command"),
) -> None:
    """
    Run each command from the given text file, exactly as written.
    - Empty lines and lines starting with '#' are ignored.
    - Each command runs in its own subprocess.
    """
    lines = file.read_text().splitlines()
    ran = 0
    for idx, raw in enumerate(lines, 1):
        cmd = raw.strip()
        if not cmd or cmd.startswith("#"):
            continue
        rprint(f"[cyan]▶ Running [{idx}]:[/] {cmd}")
        res = subprocess.run(cmd, shell=True)
        ran += 1
        if res.returncode != 0:
            rprint(f"[red]✗ Command [{idx}] exited with code {res.returncode}[/]")
            if stop_on_error:
                raise typer.Exit(res.returncode)
    rprint(f"[bold green]✓ Done[/] ({ran} command{'s' if ran != 1 else ''} run)")


if __name__ == "__main__":
    app()