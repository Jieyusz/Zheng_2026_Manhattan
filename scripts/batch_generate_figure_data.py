"""
Regenerate all figure data by running every scripts/gen_*.py.

Ordering (R7)
-------------
Cross-script dependencies force a four-phase order:

1. ``gen_count_df.py`` writes ``data/figure_data/Acortical_learning_count_df.csv``,
   which ``gen_acortical_learning.py``, ``gen_ac_generalization.py`` and
   ``gen_ac_mem.py`` read — so it runs serially FIRST.
2. The remaining producer scripts (now including ``gen_acortical_learning.py``)
   run in parallel.  Most fitting producers no longer fit inline; they write
   ``"<base> fit input"`` payloads instead (via ``utils.save_curve_fit_input``).
3. ``gen_curve_fits.py`` reads every ``"<base> fit input"`` payload and writes the
   corresponding ``"<base> fit results"`` with one shared bootstrap seed, so it
   runs serially AFTER the producers.  (``gen_wildtype_two_day_data.py`` keeps its
   grouped ``fit_two_data_df`` inline and is unaffected.)
4. ``gen_endotaxis.py`` reads the ``"Mask A example manifest"`` written by
   ``gen_wildtype_two_day_data.py`` (via ``load_all_figure_data()``), so it runs serially
   LAST.  It needs the manifest rather than recomputing the selection because the Mask-A
   example animals are drawn by a seeded ``np.random.choice`` in that script; it then
   reloads the live Session from the ``DataLoader`` by name (R8).  Its Mask-D example is
   recomputed directly (the last three wildtype Day-1 sessions) and needs no cache.

Usage
-----
    python batch_generate_figure_data.py [--overwrite | --no-overwrite]

Run from the ``scripts/`` directory (the gen scripts use ``import config`` and
write to ``config.SAVE_DIR``).
"""
import argparse
import subprocess
import sys
from glob import glob

# Must run FIRST: writes the learning-count CSV that gen_acortical_learning.py,
# gen_ac_generalization.py and gen_ac_mem.py read (R7).
PREREQUISITE_SCRIPTS = ["./gen_count_df.py"]
# Must run AFTER producers: reads every "<base> fit input" payload they write and
# produces the "<base> fit results" caches with one shared seed (R7).
CURVE_FIT_SCRIPT = "./gen_curve_fits.py"
# Must run LAST: reads the "Mask A example manifest" written by
# gen_wildtype_two_day_data.py via load_all_figure_data() (R7). Only the example animal's
# identity comes from the manifest; the Session itself is reloaded from the DataLoader (R8).
FINAL_SCRIPTS = ["./gen_endotaxis.py"]
# Validation-only: the trained per-animal RL simulations confirm that the model-free
# staircase drawn in the error-propagation figure (produced analytically by
# gen_error_propagation.py) is reproduced by a real trained agent. They are NOT figure
# inputs -- nothing loads their keys -- and they need the (uncommitted) raw trajectories,
# so they are excluded from the batch. Run them manually to re-validate; the equivalence
# is also pinned data-free in tests/test_rl_error_propagation.py.
VALIDATION_SCRIPTS = ["./gen_rl_simulation.py", "./gen_rl_turn_simulation.py"]

parser = argparse.ArgumentParser(description="Generate all figure data")
parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction,)

args = parser.parse_args()
overwrite = args.overwrite
overwrite_str = "--overwrite" if overwrite else "--no-overwrite"

# 1. Run the prerequisites serially; fail loudly so dependents never run against
#    a missing or stale CSV.
for script in PREREQUISITE_SCRIPTS:
    print(f"[1/4] Running prerequisite {script} first...")
    subprocess.run([sys.executable, script, overwrite_str], check=True)

# 2. Launch the producer scripts (everything that is not a prerequisite, the
#    central curve-fit step, or a final consumer) in parallel.
deferred = (set(PREREQUISITE_SCRIPTS) | {CURVE_FIT_SCRIPT} | set(FINAL_SCRIPTS)
            | set(VALIDATION_SCRIPTS))
producer_scripts = [s for s in sorted(glob("./gen_*.py")) if s not in deferred]
print(f"[2/4] Launching {len(producer_scripts)} producer scripts in parallel...")
process_list = [subprocess.Popen([sys.executable, s, overwrite_str]) for s in producer_scripts]
failures = [s for proc, s in zip(process_list, producer_scripts) if proc.wait() != 0]
if failures:
    raise RuntimeError(f"figure-data generation failed for: {', '.join(failures)}")

# 3. Derived data from producer output: central curve fitting (writes the
#    "<base> fit results" + bootstrap params caches) and, from the two-day paired
#    params, the Day2/Day1 ratio-CI tables.
print(f"[3/4] Running central curve fitting {CURVE_FIT_SCRIPT}...")
subprocess.run([sys.executable, CURVE_FIT_SCRIPT, overwrite_str], check=True)

# 4. Run the final consumer scripts serially, after their inputs exist.
for script in FINAL_SCRIPTS:
    print(f"[4/4] Running consumer {script} last...")
    subprocess.run([sys.executable, script, overwrite_str], check=True)
