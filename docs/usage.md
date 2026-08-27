# Manhattan Maze — Nomenclature Reference

The naming conventions used throughout the analysis code, the figures, and the
`data/figure_data/` keys: how masks are represented as graphs, and how bouts, traverses
and sorties are named. Read this to interpret a figure panel or a cached data key.

For input data schemas and units see [`data_contracts.md`](data_contracts.md); for the
manuscript equations, symbols, and units see
[`notation_guide.md`](notation_guide.md); for the top-level overview and
reproduction instructions see the [README](../README.md).

## Nomenclature

### Graph representations of masks

1. Corridor graph
    - nodes: all corridors in the mask
    - edges: all holes connecting corridors
    - Mask A, B, C are isomorphic at this level
2. Tile graph
    - nodes: all tiles in the mask
    - edges: all holes and borders between the tiles
    - Mask A, B, C has the same number of nodes

The corridor graph is **bipartite** (horizontal corridors 0–10 ↔ vertical corridors 11–21;
holes connect only across the two sets). Non-learning random-walk baselines on either graph —
expected completion steps and corridor errors for a forward-biased first-order Markov walker —
are provided by `manhattan_maze.random_walk` (`walker_metrics`).

### Bout naming convention

The two water ports:
- Home port (H): located in the home cage
- Out port (O): located at the end of the maze, central corridor on the top layer

Bout naming format: "starting port - ending port"
1. Basic references
   - "H-H": home port to home port
   - "H-O": home port to out port
   - "O-H": out port to home port
   - "O-O": out port to out port
2. Categories
   - "H*": bouts that start from the home port
   - "O*": bouts that start from the out port (homebound)
   - Traverses: "H-O", "O-H" (bouts that end at the opposite port)
     - "H-O": outbound traverse
     - "O-H": homebound traverse
   - Sorties: "H-H", "O-O" (bouts that end at the same port)
   - Note: traverses are most likely rewarded at the end, but not necessarily —
     the mouse might go directly into the home cage.

### Manuscript-aligned accessors

The published per-bout/per-session quantities have descriptive method names that map
directly to the manuscript symbols (full definitions and units in
[`notation_guide.md`](notation_guide.md)):

| Method / attribute | Manuscript symbol | Returns |
|--------------------|-------------------|---------|
| `Bout.get_duration_s(sleep_threshold=5)` | $D_{a,b}$ | Sleep-thresholded traverse/bout duration (seconds) |
| `Bout.get_turn_error_rate()` | $E_{a,b}$ | Approach-conditioned turn error rate in [0, 1] (raises for Mask D). `include="all"`/`"first"` are deprecated |
| `Bout.get_corridor_transition_matrix(normalize=True)` | $T^{(q)}$ | 22×22 directed corridor transition matrix |
| `Session.get_bottleneck_choice_ratio()` | $R_{c_b,c_j}$ | Per-journey bottleneck-choice ratio (Mask D) |
| `Session.get_three_traverse_similarity_matrix()` | $J_{O,O}, J_{H,H}, J_{O,H'}$ | Tuple `(j_oo, j_hh, j_oh_prime)` of adjusted-Jaccard matrices |
| `Session.slice_to_journeys()` | journeys | Sub-sessions split at traverse boundaries (some slices are sortie-only) |
| `Session.reward_interval_seconds` | reward interval | Inter-reward intervals (seconds) |
| `random_walk.walker_metrics(mask, beta, unit)` | $\bar\tau(\beta)$, $\mathcal{E}(\beta)$ | Random-walker completion time & expected corridor errors (`sec:walker`) |

The exponential learning-curve fits use `delta` ($\delta$, duration) and `epsilon`
($\epsilon$, turn error) as the learning-rate parameters; see
`scripts/config.py` (`CURVE_FIT_SPECS`).
