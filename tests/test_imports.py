"""
Test that all modules can be imported correctly.
"""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import latent_design
    assert hasattr(latent_design, '__version__')
    assert latent_design.__version__ == "0.1.0"


def test_bases_import():
    """Test that basis modules can be imported."""
    from latent_design.bases import Basis, BSplineBasis, PolynomialBasis, FourierBasis
    assert Basis is not None
    assert BSplineBasis is not None
    assert PolynomialBasis is not None
    assert FourierBasis is not None


def test_criteria_import():
    """Test that criteria modules can be imported."""
    from latent_design.criteria import (
        AOptimality, DOptimality, GOptimality, IOptimality
    )
    assert AOptimality is not None
    assert DOptimality is not None
    assert GOptimality is not None
    assert IOptimality is not None


def test_models_import():
    """Test that model modules can be imported."""
    from latent_design.models import FOF, SOF, SOS
    assert FOF is not None
    assert SOF is not None
    assert SOS is not None


def test_optimizers_import():
    """Test that optimizer modules can be imported."""
    from latent_design.optimizers import NBDO
    assert NBDO is not None


def test_diagnostics_import():
    """Test that diagnostic modules can be imported."""
    from latent_design.diagnostics import core, sampling, viz
    assert core is not None
    assert sampling is not None
    assert viz is not None


def test_utils_import():
    """Test that utility modules can be imported."""
    from latent_design.utils import backend, efficiency
    assert backend is not None
    assert efficiency is not None


def test_cli_import():
    """Test that CLI module can be imported."""
    from latent_design.cli import main, app
    assert main is not None
    assert app is not None
