"""
Render every ``scripts/plot_*.py`` figure into one output directory (paper/Overleaf export).

Convenience driver that runs all figure scripts in parallel with a shared ``-s`` save path.
Defaults to the in-repo, gitignored ``config.PAPER_FIGURES_DIR`` (R5: no machine-specific
external default); override ``-s`` to point at an external Overleaf tree if desired.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python update_overleaf_plots.py                 # -> ../paper_figures/
    python update_overleaf_plots.py -s /path/to/overleaf/figures
"""
import argparse
import subprocess
from glob import glob
from pathlib import Path

import config

parser = argparse.ArgumentParser(description="Render all plot_*.py figures into one directory.")
parser.add_argument("-s", "--save_path", type=str, default=str(config.PAPER_FIGURES_DIR),
                    help="Directory to save the figure PDFs (default: in-repo paper_figures/).")
args = parser.parse_args()

Path(args.save_path).mkdir(parents=True, exist_ok=True)

# Launch every figure script in its own subprocess with the shared save path.
process_list = []
scripts_list = glob("./plot_*.py")
for script in scripts_list:
    cmd = ["python", script, "-s", args.save_path]
    proc = subprocess.Popen(cmd)
    process_list.append(proc)

for proc in process_list:
    proc.wait()
