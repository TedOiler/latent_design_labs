"""latent_design.cli

Package entry for the CLI. Re-exports the Typer app and main() so
`from latent_design.cli import app, main` works cleanly in tests and docs.
"""

from .app import app, main  # noqa: F401

__all__ = ["app", "main"]

