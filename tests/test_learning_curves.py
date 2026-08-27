"""
Regression tests for learning-curve functions in utils.py.

R14 (C6): exponential_func is the production model for both duration and turn error.
_piecewise_func_legacy is retained for reference only — it must NOT be called
from any production fitting path.

Exponential model: value(b) = D_infty + (D_0 - D_infty) * exp(-k * (b - 1))
  At b=1: value = D_0 (no decay yet — b is 1-based)
  As b→∞: value → D_infty
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.utils import exponential_func, _piecewise_func_legacy


class TestExponentialFunc:
    def test_at_b1_returns_D0(self):
        """At b=1 the exponential term is exp(0)=1, so result equals D_0."""
        result = exponential_func(b=1, D_infty=10.0, D_0=50.0, k=0.5)
        assert result == pytest.approx(50.0)

    def test_converges_to_D_infty(self):
        """At large b the function converges to D_infty."""
        result = exponential_func(b=1000, D_infty=10.0, D_0=50.0, k=10.0)
        assert result == pytest.approx(10.0, abs=1e-6)

    def test_monotone_decreasing_when_D0_gt_Dinfty(self):
        bs = np.arange(1, 20)
        vals = exponential_func(bs, D_infty=10.0, D_0=50.0, k=0.3)
        assert np.all(np.diff(vals) < 0)

    def test_monotone_increasing_when_D0_lt_Dinfty(self):
        bs = np.arange(1, 20)
        vals = exponential_func(bs, D_infty=50.0, D_0=10.0, k=0.3)
        assert np.all(np.diff(vals) > 0)

    def test_k0_is_constant(self):
        """k=0 means no learning — value stays at D_0."""
        bs = np.arange(1, 10)
        vals = exponential_func(bs, D_infty=10.0, D_0=50.0, k=0.0)
        assert np.all(vals == pytest.approx(50.0))

    def test_flat_when_D0_equals_Dinfty(self):
        bs = np.arange(1, 10)
        vals = exponential_func(bs, D_infty=30.0, D_0=30.0, k=1.0)
        assert np.all(vals == pytest.approx(30.0))


class TestPiecewiseFuncLegacy:
    """
    _piecewise_func_legacy is NOT used in the production pipeline (R14).
    These tests preserve its behaviour as a regression anchor.
    """

    def test_at_b0_returns_e0(self):
        result = _piecewise_func_legacy(b=0, e_infty=0.1, e0=0.5, alpha=0.1)
        assert result == pytest.approx(0.5)

    def test_converged_returns_e_infty(self):
        result = _piecewise_func_legacy(b=10, e_infty=0.1, e0=0.5, alpha=0.1)
        # e0 - alpha*10 = 0.5 - 1.0 = -0.5 → clipped to e_infty=0.1
        assert result == pytest.approx(0.1)

    def test_intermediate_value(self):
        # e0 - alpha*2 = 0.5 - 0.2 = 0.3 (above e_infty=0.1, below e0=0.5)
        result = _piecewise_func_legacy(b=2, e_infty=0.1, e0=0.5, alpha=0.1)
        assert result == pytest.approx(0.3)

    def test_not_called_from_production_path(self):
        """
        Verifies _piecewise_func_legacy is NOT imported or called from gen_*.py scripts.
        Any production call to this function violates R14.
        """
        import glob
        import ast

        gen_scripts = glob.glob(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "gen_*.py")
        )
        for script_path in gen_scripts:
            with open(script_path, "r") as f:
                source = f.read()
            assert "_piecewise_func_legacy" not in source, (
                f"{script_path} references _piecewise_func_legacy — "
                "this function must not appear in the production pipeline (R14)."
            )
