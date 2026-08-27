"""
Smoke test for every figure script in ``scripts/plot_*.py``.

Runs each script end-to-end with a temporary ``-s/--save_path`` and asserts it
exits 0 and writes at least one PDF. This is a *smoke* test: it guards against
import/refactor breakage (the plot-script refactor pass), not pixel-level figure
identity. Output is not numerically compared (some panels use unseeded jitter via
``plot_jittered_scatter``, so byte-identical output is not expected).

Run in the project conda env (``m_maze``) so the package and figure-data caches
are importable, e.g.::

    PYTHONPATH=<repo> python -m pytest tests/test_plot_scripts_smoke.py
"""

import os
import sys
import glob
import subprocess

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")

_PLOT_SCRIPTS = sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(_SCRIPTS_DIR, "plot_*.py"))
)


def test_plot_scripts_discovered():
    """All 19 figure scripts are present (guards against an empty parametrization)."""
    assert len(_PLOT_SCRIPTS) == 20, _PLOT_SCRIPTS


@pytest.mark.parametrize("script", _PLOT_SCRIPTS)
def test_plot_script_runs(script, tmp_path):
    """
    Each ``plot_*.py`` runs to completion and writes a PDF.

    The script is invoked with ``cwd=scripts/`` (so ``import config`` resolves),
    ``MPLBACKEND=Agg`` (headless), and ``-s <tmp>`` so no manuscript output is
    touched. Asserts exit code 0 and at least one ``*.pdf`` in the temp dir.
    """
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    save_dir = str(tmp_path) + os.sep

    result = subprocess.run(
        [sys.executable, script, "-s", save_dir],
        cwd=_SCRIPTS_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{script} exited {result.returncode}\n--- stderr ---\n{result.stderr[-3000:]}"
    )
    pdfs = glob.glob(os.path.join(str(tmp_path), "*.pdf"))
    assert pdfs, f"{script} produced no PDF in {tmp_path}"
