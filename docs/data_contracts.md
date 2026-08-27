# Data Contracts — Manhattan Maze

This document describes every core data object in the pipeline: its required columns/fields,
units, allowed values, and known failure modes. It is intended as a reference for refactoring,
testing, and onboarding. **Do not change numerical behaviour without updating this document.**

---

## Table of Contents

- [Coordinate System](#coordinate-system)
- [§3 bout\_df](#3-bout_df)
- [§4 tiles\_df](#4-tiles_df)
- [§5 corridors\_df](#5-corridors_df)
- [§6 Mask object](#6-mask-object)
- [§7 metadata CSV](#7-metadata-csv)
- [§10 Session.reward\_interval\_seconds](#10-sessionreward_interval_seconds)
- [§12 figure\_data files](#12-figure_data-files)
- [§13 Random-walk completion metrics](#13-random-walk-completion-metrics-random_walkpy)

Section numbers are those of the full contract and are referenced from docstrings
throughout the code, so they are kept stable. The sections covering inputs that are not
distributed with this repository — raw trajectories, the processed-bout cache, room-status
files, reward files and the `manual_fixes` JSON — are omitted here.

---

## Coordinate System

All maze coordinates follow a 2D (col, row) / (x, y) convention.
The maze is modelled as two logical floors stacked on top of each other.

| Symbol | Meaning | Range (default maze\_size=11) |
|--------|---------|-------------------------------|
| `x` (col) | horizontal position, increases East | 0 – 10 |
| `y` (row) | vertical position, increases North | 0 – 10 |
| `z` (floor) | 0 = bottom floor (horizontal corridors), 1 = top floor (vertical corridors) | 0 or 1 |
| `tile` index | `x + y*size + z*size²` | 0 – 241 |
| `corridor` index | horizontal (z=0): `y` (0–10); vertical (z=1): `x + size` (11–21) | 0 – 21 |

**Home port**: `(x=0, y=5, z=0)` → tile 55, corridor 5.  
**Out port**: `(x=5, y=9, z=1)` → tile 225, corridor 16.

These defaults are set in `DataLoader.__init__` and are propagated to every `Mask`.
Changing them redefines what counts as a traverse and invalidates all learning metrics.

---

## 3. `bout_df`

**The fundamental unit of trajectory data.**  
Stored inside `processed_bout_df_list` and also accessible as `Bout.bout_df`.

### Schema

| Column | dtype | Allowed values | Units |
|--------|-------|---------------|-------|
| `in_frame` | `int64` | ≥ 0 | absolute video frame number |
| `out_frame` | `int64` | ≥ `in_frame` | absolute video frame number |
| `discrete_loc` | `tuple(int, int)` | `(col, row)` with col, row ∈ [0, 10] | maze grid position |

### Row semantics

Each row represents the animal occupying a single maze tile continuously.
- `in_frame`: first frame the animal was assigned to this tile.
- `out_frame`: last frame before moving to the next tile.
- `discrete_loc`: `(col, row)` — a Python `tuple` of two Python or NumPy `int`.

**`out_frame` of row k must equal `in_frame` of row k+1 minus 1** (no gaps in frame coverage)
after QC processing. Minor violations (off-by-one) may remain at bout boundaries.

### Minimum length

A valid bout has **≥ 3 rows** (`_check_no_empty_bouts` enforces `min_length=3`).

### Start / end constraints (post-QC)

- Row 0: `discrete_loc` must be `home_pos = (0, 5)` or `out_pos = (5, 9)`.
- Row −1: `discrete_loc` must be `home_pos` or `out_pos`.
- A bout starting and ending at the **same** port is a sortie (H-H or O-O).
- A bout starting at one port and ending at the other is a traverse (H-O or O-H).

### Movement constraints (post-QC)

Between any two consecutive rows, `discrete_loc` must change by exactly 1 in either `col`
or `row`, but not both (no diagonal moves). Changes in both dimensions simultaneously indicate
a turn passing through a hole and should be split into two rows by the QC pipeline.

### Failure modes

- `discrete_loc` stored as a `list` instead of `tuple` in older cached files;
  `bouts_to_tiles_format` creates tuples natively.
- Frame overlap between consecutive bouts can occur if `_check_first_and_last_coords`
  extends an endpoint by extrapolating with a negative duration.
- NaN-valued `in_frame`/`out_frame` can appear after `_add_coords` if `new_vals` is
  incomplete; downstream `tiles_df` construction will fail with a type error.

---

## 4. `tiles_df`

**Derived from `bout_df` inside `Bout._build_tiles_df`.**  
Accessed as `Bout.tiles_df`.  
Each row represents the animal on a **tile** (a maze cell with explicit floor assignment).

### Schema

| Column | dtype | Allowed values | Units |
|--------|-------|---------------|-------|
| `in_frame` | `int64` | ≥ 0 | absolute video frame |
| `out_frame` | `int64` | ≥ `in_frame` | absolute video frame |
| `tile` | `int64` | 0 – 241 (for size=11) | tile index = `x + y*11 + z*121` |
| `x` | `int64` | 0 – 10 | column (East) |
| `y` | `int64` | 0 – 10 | row (North) |
| `z` | `int64` | 0 or 1 | floor (0=horizontal, 1=vertical) |

### Construction rule

At every hole in the mask, the mouse transitions between floors (z=0 ↔ z=1).
`_build_tiles_df` detects this by comparing the direction of movement before and after each
tile. If the floor changes, the tile is **split at the midpoint frame** into two rows:
one for each floor. This means `len(tiles_df) ≥ len(bout_df)`.

### Invariants

- `tile = x + y*size + z*size²`, consistent with `utils.xyz_to_ti`.
- `x`, `y`, `z` are always derivable from `tile` via `utils.ti_to_xyz`.
- Tile 55 = home tile `(0,5,z=0)`. Tile 225 = out tile `(5,9,z=1)`.
- Row count is `len(bout_df) + n_holes_visited` (each hole adds one split row).

### Failure modes

- If `bout_df` has fewer than 2 rows, `_build_tiles_df` raises `ValueError` ("must have at
  least two tiles"). This should not occur after `_check_no_empty_bouts`.
- `z_xy` infers floor from whether two consecutive positions share the same column (vertical =
  z=1) or the same row (horizontal = z=0). If a bout begins at the first tile, the floor of
  tile 0 is inferred from tile 1 — if the first two tiles are at the same position,
  `z_xy` returns 0 by default.

---

## 5. `corridors_df`

**Derived from `tiles_df` inside `Bout._build_corridors_df`.**  
Accessed as `Bout.corridors_df`.  
Each row represents one continuous run in a single corridor.

### Schema

| Column | dtype | Allowed values | Units |
|--------|-------|---------------|-------|
| `in_frame` | `int64` | ≥ 0 | absolute video frame |
| `out_frame` | `int64` | ≥ `in_frame` | absolute video frame |
| `corridor` | `int64` | 0 – 21 (for size=11) | corridor index |

### Corridor index encoding

| Range | Axis | Formula |
|-------|------|---------|
| 0 – 10 | horizontal (z=0) | `y` value of the corridor |
| 11 – 21 | vertical (z=1) | `x + 11` |

Home corridor = 5. Out corridor = 16.

### Construction rule

Consecutive tiles in the same corridor are condensed into a single row.
The `in_frame` is taken from the first tile in the run; `out_frame` from the last.

### Invariants

- `len(corridors_df) ≤ len(tiles_df)`.
- Consecutive rows always have different `corridor` values (consecutive same-corridor
  runs are merged by `df_condense_consecutive_repeats`).
- Corridors in range 0 – 10 correspond to horizontal traversal (floor z=0).
- Corridors in range 11 – 21 correspond to vertical traversal (floor z=1).

### Failure modes

- A `bout_idx` column is added dynamically to `corridors_df` by `Session.concat_corridors_df`
  (not present on the raw `Bout.corridors_df`). Code that reads this column will fail if
  called on a `Bout` directly rather than through `Session.concat_corridors_df`.

---

## 6. `Mask` object

**Source**: `data/masks/holes_{name}.npy` for hole geometry.
**Optional special source**: `data/masks/special/{canonical_name}.json` for mask-specific scientific annotations.
**Instantiated by**: `DataLoader._load_mask`.

The `Mask` object represents the physical maze geometry and the graph derived from that geometry. It should not contain analysis logic. Mask-specific scientific annotations, such as the Mask-D bottleneck structure or the adjusted-Jaccard correction, are stored in `Mask.special`.

### Attributes

| Attribute                      | Type                                                                          | Description                                                                                                                                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                         | `str`                                                                         | Exact configuration name used by the data loader, e.g. `'O'`, `'A'`, `'B'`, `'C'`, `'D'`, `'E'`, `'F'`, `'AO'`, `'A_flipped'`, `'C_flipped'`, `'D_flipped'`, `'DT'`.                                    |
| `canonical_name`               | `str`                                                                         | Base mask identity after removing variant suffixes where appropriate, e.g. `'D'` for `'D'`, and optionally for `'D_flipped'` if it shares the same scientific topology after coordinate transformation. |
| `variant`                      | `str \| None`                                                                 | Variant label such as `'flipped'`, `'reduced'`, or `'test'`. `None` for canonical masks.                                                                                                                |
| `coordinate_frame`             | `str`                                                                         | Coordinate frame of the mask. Allowed values: `'full_11x11'` or `'reduced_9x9'`.                                                                                                                        |
| `size`                         | `int`                                                                         | Maze grid dimension. Always 11 in production full-frame masks; 9 after `remove_outskirts()`.                                                                                                            |
| `holes_coords`                 | `np.ndarray` shape `(n_holes, 2)` dtype `int`                                 | Hole coordinates as `[[col, row], ...]` in the current `coordinate_frame`.                                                                                                                              |
| `holes_list`                   | `list[tuple[int, int]]`                                                       | Same hole coordinates as a list of `(col, row)` tuples.                                                                                                                                                 |
| `home_pos`                     | `tuple[int, int]`                                                             | Home port coordinate `(col, row)` in the current `coordinate_frame`. In the full frame this is `(0, 5)`.                                                                                                |
| `out_pos`                      | `tuple[int, int]`                                                             | Out/reward port coordinate `(col, row)` in the current `coordinate_frame`. In the full frame this is `(5, 9)`.                                                                                          |
| `home_tile`                    | `int`                                                                         | Tile index of the home port. In the full frame this is 55.                                                                                                                                              |
| `out_tile`                     | `int`                                                                         | Tile index of the out/reward port. In the full frame this is 225.                                                                                                                                       |
| `home_corridor`                | `int`                                                                         | Corridor index of the home port. In the full frame this is 5.                                                                                                                                           |
| `out_corridor`                 | `int`                                                                         | Corridor index of the out/reward port. In the full frame this is 16.                                                                                                                                    |
| `tiles_adj_mat`                | `np.ndarray` shape `(2 * size * size, 2 * size * size)` dtype `bool` or `int` | Symmetric tile adjacency matrix. Entries indicate adjacency between individual tiles across the two maze layers.                                                                                        |
| `tiles_shortest_distances`     | `np.ndarray` shape `(2 * size * size, 2 * size * size)` dtype `float64`       | All-pairs shortest-path distances between tiles. Infinite distance indicates disconnected nodes.                                                                                                        |
| `tiles_shortest_path`          | `list[int]`                                                                   | One shortest tile path from `home_tile` to `out_tile`, including endpoints.                                                                                                                             |
| `corridors_adj_mat`            | `np.ndarray` shape `(22, 22)` dtype `bool` or `int`                           | Symmetric corridor adjacency matrix. Entries indicate allowed transitions between corridors through holes.                                                                                              |
| `corridors_shortest_distances` | `np.ndarray` shape `(22, 22)` dtype `float64`                                 | All-pairs shortest-path distances between corridors.                                                                                                                                                    |
| `corridors_shortest_path`      | `list[int]`                                                                   | One shortest corridor path from `home_corridor` to `out_corridor`, including endpoints.                                                                                                                 |
| `correct_turns_outbound`       | `dict[tuple[int, int], str]`                                                  | Mapping from hole coordinate to correct allocentric turn direction for outbound H→O traverses. Directions are `'N'`, `'S'`, `'E'`, `'W'`.                                                               |
| `correct_turns_homebound`      | `dict[tuple[int, int], str]`                                                  | Mapping from hole coordinate to correct allocentric turn direction for homebound O→H traverses.                                                                                                         |
| `special`                      | `dict \| None`                                                                | Optional mask-specific scientific specification loaded from JSON. For Mask D, this stores bottleneck, biclique, and adjusted-Jaccard information. `None` for ordinary masks.                            |

### Indexing and coordinate conventions

* Hole and port coordinates are always `(col, row)`, not `(row, col)`.
* Tile IDs are integer indices over the two maze layers.
* Corridor IDs are integer indices in `[0, 21]`.
* Internal array indices are 0-based.
* Manuscript-facing traverse numbers and bout numbers are 1-based.
* `traverse_idx` means 0-based internal index.
* `traverse_number` means 1-based display/manuscript number.
* `bout_idx` means 0-based internal bout index.
* `bout_number` means 1-based display/manuscript bout number.
* The symbol (b) is reserved for bout in the manuscript and should not be used as a generic code variable.

### Invariants

* `holes_coords` contains only valid maze positions: `col ∈ [0, size - 1]`, `row ∈ [0, size - 1]`.
* `holes_coords` contains no duplicate coordinates.
* `holes_list == [tuple(x) for x in holes_coords]`.
* `tiles_adj_mat` is symmetric.
* `corridors_adj_mat` is symmetric.
* `tiles_adj_mat[i, j] = 1` iff tiles `i` and `j` are adjacent within a layer or connected across layers by a valid hole.
* `corridors_adj_mat[i, j] = 1` iff corridors `i` and `j` are connected through a valid hole.
* `tiles_shortest_path[0] == home_tile`.
* `tiles_shortest_path[-1] == out_tile`.
* `corridors_shortest_path[0] == home_corridor`.
* `corridors_shortest_path[-1] == out_corridor`.
* `corridors_shortest_distances[home_corridor, out_corridor]` equals the minimum number of corridor transitions required to solve the maze.
* `correct_turns_outbound` and `correct_turns_homebound` are defined in the same coordinate frame as `holes_coords`.
* If `coordinate_frame == 'reduced_9x9'`, all coordinates and graph products must be recomputed or explicitly transformed. Full-frame and reduced-frame graph products must not be mixed.

### Failure modes

* If `data/masks/holes_{name}.npy` is missing for a mask listed in `Config_label_list`, `DataLoader` raises `FileNotFoundError` at construction.
* If a special JSON file is declared for a mask but missing, `DataLoader` raises `FileNotFoundError` unless special-mask loading is explicitly disabled.
* If `remove_outskirts()` is called, the returned mask has `coordinate_frame == 'reduced_9x9'`, `variant == 'reduced'`, and a name such as `"{original_name}_reduced"`.
* Code that mixes full-frame and reduced-frame masks must raise a clear error rather than silently producing incorrect corridor indices.
* A reduced mask should not be inserted into `DataLoader.masks` under the original full-frame key.
* Mask-specific analyses must fail loudly if called on a mask without the required `special` fields.

### Mask-D special specification

Mask D has additional scientific structure and must not be handled by universal constants scattered across the codebase. Its special annotations are stored in:

```text
data/masks/special/mask_d.json
```

The generic `Mask` object loads this file into `mask.special` when `canonical_name == 'D'`.

Required Mask-D special fields:

```
{
  "canonical_name": "D",
  "applies_to": ["D"],
  "topology": {
    "description": "Mask D contains two highly connected subgraphs connected through bottleneck corridor(s).",
    "bottleneck_corridors": [],
    "left_subgraph_corridors": [],
    "right_subgraph_corridors": [],
    "excluded_corridors": []
  },
  "adjusted_jaccard": {
    "enabled": true,
    "n_guaranteed_transitions": 3,
    "scope": "Mask-D transition-similarity analysis only",
    "description": "Subtracts guaranteed task-structure transitions before computing adjusted Jaccard similarity."
  },
  "transition_analysis": {
    "uses_directional_transitions": true,
    "homebound_matrix_is_transposed_for_retrace_similarity": true
  }
}
```

The corridor lists should be filled from the existing values in `manhattan_maze/mask_d.py`. They must not be re-derived independently in multiple scripts.

### Mask-D-specific functions

Mask-D-only logic belongs in a separate module, for example:

```text
src/manhattan_maze/mask_d.py
```

This module should contain pure functions that consume a generic `Mask` object with a valid `mask.special` field. The generic `Mask` class should not grow Mask-D-only methods.

Recommended functions:

```python
def validate_mask_d(mask: Mask) -> None:
    """Fail if `mask` is not canonical Mask D or lacks required Mask-D special fields."""

def get_mask_d_bottleneck_corridors(mask: Mask) -> list[int]:
    """Return Mask-D bottleneck corridor IDs from `mask.special`."""

def get_mask_d_subgraph_corridors(mask: Mask) -> dict[str, list[int]]:
    """Return Mask-D left/right subgraph corridor IDs from `mask.special`."""

def get_mask_d_adjusted_jaccard_correction(mask: Mask) -> int:
    """Return the Mask-D guaranteed-transition correction, currently 3."""

def compute_mask_d_adjusted_jaccard(..., mask: Mask) -> float:
    """Compute Mask-D adjusted Jaccard using the correction declared in `mask.special`."""

def compute_mask_d_bottleneck_preference(..., mask: Mask) -> float:
    """Compute Mask-D bottleneck preference using bottleneck corridors declared in `mask.special`."""
```

These functions must raise a clear error when called on non-D masks unless explicitly designed to support a transformed Mask-D variant such as `D_flipped`.

### Mask-D invariants

* Mask-D adjusted Jaccard must not apply a universal hard-coded `-3` correction.
* The correction is read from `mask.special["adjusted_jaccard"]["n_guaranteed_transitions"]`.
* The correction applies only to Mask-D transition-similarity analyses.
* Non-D masks must use standard Jaccard unless a mask-specific correction is explicitly defined.
* Homebound retracing similarity uses a direction-reversed or transposed homebound transition representation, corresponding to (J_{\mathrm{O,H'}}) in the manuscript.
* If `D_flipped` is included under the Mask-D family, the JSON must explicitly specify whether the same corridor IDs apply or whether a coordinate/corridor transformation is required.

### Tests

Required tests:

1. Loading Mask D attaches a non-null `mask.special`.
2. Loading non-D masks leaves `mask.special is None`, unless they have their own special JSON.
3. Mask-D special JSON contains all required fields.
4. `validate_mask_d(mask_d)` passes for Mask D.
5. `validate_mask_d(mask_a)` fails clearly for Mask A.
6. Adjusted Jaccard for Mask D uses `n_guaranteed_transitions == 3`.
7. Adjusted Jaccard for non-D masks does not silently subtract 3.
8. `D_flipped` behavior is explicitly tested if it is treated as part of the Mask-D family.
9. Full-frame and reduced-frame masks cannot be mixed in the same graph or corridor-index analysis.

---

## 7. Metadata CSV

**Path**: `data/manhattan_metadata_published.csv` (included).  
**Read by**: `DataLoader._get_metadata`.

### Schema

| Column | dtype | Allowed values | Notes |
|--------|-------|---------------|-------|
| `Nickname` | `str` | `"{Animal}_{experiment_code}"` e.g. `"Z9_t1"` | Unique identifier for one recording session |
| `Sex` | `str` | `'M'`, `'F'` | Animal sex |
| `Age` | `int64` | > 0 | Age in days at experiment start |
| `Genotype` | `str` | `'BL6J'`, `'HO'`, `'WT'` | `BL6J` = wildtype control; `HO` = acortical; `WT` = wildtype |
| `Room` | `str` | free text | Recording room identifier |
| `Condition` | `str` | `'Single_north'`, etc. | Housing condition |
| `Config_label_list` | `str` | comma-separated mask names e.g. `"O, A"` | Ordered list of mask configurations applied in this recording |
| `Notes` | `str` / NaN | free text | Experimenter notes; not used computationally |

**Derived column** added by `_get_metadata`:

| Column | Type | Rule |
|--------|------|------|
| `Animal` | `str` | `Nickname.split("_")[0]` — the animal identifier |

### `Config_label_list` parsing

`DataLoader._get_mask_labels` parses this column as:
```python
mask_labels = field.strip('][').split(', ')
```
The space after the comma is significant. A missing space (e.g. `"O,A"`) will produce
incorrect mask names and silently use wrong configurations.

### Invariants

- Every `Nickname` must have exactly one matching `verified_room_status` file and one
  `raw_trajectory` file.
- The number of mask labels in `Config_label_list` must equal the number of
  `"Configuration_*"` rows in the corresponding room-status file.
- `Age` values for the same animal across multiple recordings should increase monotonically.

### Failure modes

- Trailing whitespace in `Config_label_list` entries produces mask names like `"A "` that
  will not match any key in `DataLoader.masks`, raising `KeyError`.
- The `Unnamed: 8` column (an artefact of LibreOffice CSV export) is silently retained;
  `drop_unnamed_column` only removes `Unnamed: 0`.

---

## 10. `Session.reward_interval_seconds`

**Computed by**: `Session.extract_rwd_intervals_array`.  
**Stored as**: `Session.reward_interval_seconds` (units: **seconds**). The former
`Session.rwd_int_array` remains as a deprecated read-only alias returning the same array;
new code, exports, and plot labels must use `reward_interval_seconds`.

### Schema

`np.ndarray` of shape `(n_rewards,)` dtype `float64`.  
Each element is the time elapsed between consecutive reward events.

### Units

- Values are in **seconds** (frame differences divided by `FPS=30`), made explicit by the
  attribute name (C7/R3). The earlier "minutes" code comments were a documentation bug and
  have been removed.
- Downstream first-hour reward counts therefore use a **3600 s** threshold, e.g.
  `first_hour_rewards = len(rt[rt < 3600])` in `gen_wildtype_two_day_data.py` and the
  `np.cumsum(...) < 3600` form in `gen_acortical_learning.py`.

### Invariants (as currently computed)

- Values are in **seconds** (frame differences divided by `FPS=30`).
- The first element is the time from `Session.first_frame` to the first reward frame.
- Subsequent elements are inter-reward intervals.
- If `reward_df` is empty, `reward_interval_seconds` is an empty list `[]` (not a numpy array).

### Failure modes

- Returns `[]` (Python list, not ndarray) when `reward_df` is empty. Code that calls
  `.shape` or numpy operations on the result will fail.
- NaN in `rwd_2` column is silently excluded, so the event count may be lower than
  expected if many rewards went unconsumed.

---

## 12. `figure_data` files

**Directory**: `data/figure_data/`.  
**Produced by**: `utils.save_modular_data`.  
**Read by**: `utils.load_all_figure_data` (called at the top of every `plot_*.py` script).

### File formats by Python type

| Serialisation | Extension | Used for |
|---------------|-----------|---------|
| `np.ndarray` | `.npy` | Per-animal metric arrays (duration, turn error rate, etc.) |
| `pd.DataFrame` | `.parquet` | Tabular data (learning count df, animal masks df, memory metrics) and all example-session/bout tables |
| Any other object (lists, dicts, fitted-curve tuples) | `.pkl` | Fit results, fit inputs, similarity matrices |

**No live objects (R8).** Caches must not embed live `Session` / `Bout` / `Trajectory`
objects: they fail to unpickle the moment a class moves, and the back-reference chain
(`Bout.session` -> `Trajectory` -> every other session + the full mask library) makes them
enormous — a four-bout cache was 14.7 MB for 54 KB of actual bout data. Example sessions and
traverses are therefore exported as flat tables by `manhattan_maze.plot_data`; see
*Example-session tables* below.

The one permitted exception is `masks.pkl`, listed in `manhattan_maze.io.R8_PICKLE_ALLOWLIST`
because `Mask` / `MaskDSpecial` must be *called* as objects downstream (`mask.plot`,
`mask.tiles_shortest_distances`), including by the array-based renderers, which take a `Mask`
argument. Any other object pickle in the directory is a bug — it would fail to unpickle
once the class moved.

### Example-session tables

Each family of example sessions/bouts is exported as four keys, `"<base> {suffix}"`, written by
`utils.get_example_session_tables` (sessions) or `utils.get_example_bout_tables` (individual
bouts). Bases: `Mask A example`, `Mask D example`, `Acortical A example`, `Control A example`,
`Swap example`, `Overnight traverse example`, `Acortical E traverse example`,
`Acortical mem traverse example`.

| Suffix | Rows | Key columns |
|---|---|---|
| `bout steps` | one per `bout_df` row | `example, bout_idx, step, col, row` |
| `tile steps` | one per `tiles_df` row (longer: turn cells split per floor) | `example, bout_idx, step, in_frame, out_frame, tile, tile_distance` |
| `bout meta` | one per bout | `example, bout_idx, traverse_idx`/`label`, `bout_type, duration_s, start_frame, start_time_s, n_steps, is_ho, is_oh, cum_duration_s, session_idx` |
| `manifest` | one per session/bout | `cache, example, cohort_row, animal_name, session_idx, first_frame, last_frame, fps, mask_name, n_bouts, in_maze_end_s, session_span_s` |

Conventions that matter:

- **`example` is a positional index**, matching the `config.py` selectors
  (`MASK_A_EXAMPLE_ID`, `MASK_D_EXAMPLE_ID`, …), which index the old pickled lists. The animal
  name is a separate column.
- **Frames, not seconds.** `in_frame`/`out_frame` are absolute `int64` video frames and `fps`
  lives in the manifest, because consumers use four different time origins over the same rows
  (a sliced segment's own first frame, the parent session's first frame, the bout's *last*
  frame for `plot_tile_seq(inverse=True)`). The renderer picks the origin.
- **The superset is exported** — every bout of every example session — so the `config.py`
  selectors (`MASK_A_SEGMENT_BOUTS`, `MASK_D_MOTIF_TRAVERSES`, `EXAMPLE_TRAVERSE_INDICES`,
  `ACORTICAL_A_SEGMENT_BOUTS`, …) stay authoritative at *plot* time. The old
  `"… example segment"` / `"… example traverses"` caches are row selections on these.
- **Sorties carry `traverse_idx == -1`**, an explicit sentinel rather than a null: a null in an
  `int64` column promotes it to `float64` and traverse numbers render as `Trav.#1.0`.
- **Nothing is pre-binned or pre-derived.** The distance-over-time frame and the speed
  histogram's step-time point process are reconstructed at plot time by
  `utils.derive_tile_distance_table` / `utils.derive_step_times` / `utils.derive_bout_path_table`,
  because caching them too would store the same numbers three times. Regression tests assert
  each derived frame equals its extractor's output on a live session.

### Naming convention

File names are human-readable keys: `"{Genotype} {Mask} {metric}.{ext}"`.  
Examples:
- `Acortical Mask A duration.npy`
- `Wildtype A traverse duration.npy`
- `Control A duration fit input.pkl` (tidy frame + fit metadata; produced by `utils.save_curve_fit_input`, consumed by `gen_curve_fits.py`)
- `Control A duration fit results.pkl` (bootstrap fit tuple; written by `gen_curve_fits.py`)
- `Control A duration bootstrap params.parquet` (raw per-iteration bootstrap parameter draws; written by `gen_curve_fits.py`)
- `Wildtype two day duration param ratios.parquet` (Day2/Day1 within-subject curve-derived ratio CIs; per-metric, plus the combined `Wildtype two day param ratios`) — the `param ratios` key names are retained for cache compatibility
- `Wildtype D corridor transition matrices.npy` / `Wildtype D corridor adjacency.npy` (outskirt-removed 18×18 corridor space; written by `gen_endotaxis.py`)

The key string is used as a `dict` key in `figure_data_dict` after `load_all_figure_data`.

**Plain-CSV side artifacts.** A few DataFrames are additionally mirrored to plain
`.csv` for human inspection, alongside their canonical `save_modular_data` copy:
`Acortical_learning_count_df.csv` (mirrors the `"Acortical learning count df"` frame;
written by `gen_count_df.py` and read back by the acortical producers as the
learning-count source) and `Acortical_animal_learned_masks_df.csv` (mirrors
`"Acortical animal learned masks df"`). These `.csv` files are not loaded by
`load_all_figure_data` (which keys off the `save_modular_data` names); they are
convenience exports.

### Common array shapes

| Key pattern | Shape | dtype | Notes |
|-------------|-------|-------|-------|
| `"* traverse duration"` | `(n_animals, n_traverses)` | `float64` | seconds; NaN-padded for shorter sessions |
| `"* turn error rate"` | `(n_animals, n_traverses)` | `float64` | fraction [0, 1], approach-conditioned; NaN if no scored crossings in bout |
| `"* traverse speed"` | `(n_animals, n_traverses)` | `float64` | tiles/s |
| `"* tile error"` | `(n_animals, n_traverses)` | `float64` | count |
| `"* corridor error"` | `(n_animals, n_traverses)` | `float64` | count |
| `"* sortie counts"` | `(n_animals, n_rewards)` | `float64` | count of sorties between rewards |
| `"* sortie count by direction"` | `(n_animals, 2)` | `float64` | mean sorties per journey, split by starting port — columns `[H-H home-start, O-O out-start]` (from `utils.sorties_per_journey_by_direction`) |
| `"* reward intervals"` | `(n_animals, n_rewards)` | `float64` | seconds (from `get_slice_stats`, not the per-session `reward_interval_seconds`) |
| `"* first hour reward"` | `(n_animals,)` | `int` | count of rewards within the first 3600 s of the session; stored as a Python list → `.pkl` |
| `"* first journey timing"` | `(n_animals, 2)` | `float64` | in-maze seconds, columns `[start→first bottleneck, last bottleneck exit→first reward]` (Mask D) |
| `"* tiles per corridor"` | `(n_animals,)` | `float64` | scalar per animal |
| `"* average traverse similarity"` | `(n_animals,)` | `float64` | modified Jaccard, [0, 1] |

### Fitted curve tuple (`.pkl`)

`fit_results = (bs, ds, summary_df, bootstrap_curves)` where:
- `bs`: `np.ndarray` of bout indices (x-values)
- `ds`: `np.ndarray` of observed metric values (y-values)
- `summary_df`: `pd.DataFrame` with columns `["Parameter", "Estimate", "CI_lower", "CI_upper"]`
  (parameter names: `D_infty`, `D_0`, `delta` for duration; `E_infty`, `E_0`, `epsilon` for turn error)
- `bootstrap_curves`: `(x_grid, ci_lower, central_curve, ci_upper)` — each a `np.ndarray` of shape `(100,)`

### Bootstrap parameter draws, ratios, and Mask-D transitions

- **`"<base> bootstrap params"`** (`.parquet`): the raw per-iteration bootstrap parameter
  draws, one column per parameter — input to cross-group ratio CIs
  (`utils.bootstrap_ratio_ci` / `bootstrap_summary_ratio_ci`). The two-day key
  `"Wildtype two day {metric} bootstrap params"` is a `{(session, mask): DataFrame}`
  dict whose rows are **aligned across groups by a shared animal resample** (within-subject).
- **`"Wildtype two day {metric} param ratios"`** and the combined **`"Wildtype two day param ratios"`**
  (`.parquet`): tidy Day2/Day1 **curve-derived** ratio CIs (the `param ratios` key is kept
  for cache compatibility), columns
  `[Metric, Parameter, Session, Mask, Ratio, CI_lower, CI_upper, sat_frac_num, sat_frac_den, mask_frac]`;
  the across-combinations summary row uses `Session = Mask = "all"`. The three diagnostic
  columns are per-row saturation/masking fractions (see `docs/ratio_ci_method.md`):
  `sat_frac_num`/`sat_frac_den` = fraction of the numerator/denominator's *raw* fit-parameter
  draws sitting on a bound (for the asymptote row this is the `X_infty` saturation that
  motivates using `X_late`; for the rate row the delta/epsilon at-bound fraction); `mask_frac`
  = fraction of paired bootstrap iterations dropped as non-finite when forming that ratio.
  The independent-cohort tables (`"Acortical A genotype param ratios"`,
  `"Acortical generalization param ratios"`) carry the same diagnostic columns with
  `Comparison` in place of `Session`/`Mask`.
- **`"Wildtype two day {metric} tidy"`** (`.parquet`): the tidy per-traverse frame
  (columns `Animal, Session, Mask, b, Value`) written by `gen_wildtype_two_day_data.py`,
  consumed by `gen_curve_fits.py` for the model-free late-window robustness check.
- **`"Wildtype two day {metric} ratio robustness"`** and the combined
  **`"Wildtype two day ratio robustness"`** (`.parquet`): diagnostic table cross-checking
  the shipped estimator; columns `[Metric, Comparison, Xlate_ratio, modelfree_ratio,
  rate_masked, rate_masked_hi, rate_unmasked, rate_unmasked_hi]` — the curve-derived
  `X_late` ratio vs a fully model-free late-window ratio, and the rate ratio (median +
  97.5th pct) with vs without boundary NaN-masking. Does not feed any figure.
- **`"Wildtype D corridor transition matrices"`** (`.npy`): shape `(n_sessions, size, 18, 18)`,
  `arr[s, slice, end, start] = P(start → end)` in the outskirt-removed display corridor order
  (the same index space as the endotaxis schematic). **`"Wildtype D corridor adjacency"`** (`.npy`)
  is the matching `18×18` static corridor graph. Both consumed by `plot_d_full` / `plot_d_supp`
  via `utils.select_d_transition_dict` and `plot_utils.plot_corridor_transition_schematic`.
- **`"Wildtype Mask D goal transition array"`** (`.npy`): shape `(n_animals, 20)`, the
  per-reward bottleneck goal-choice sequence (from `get_slice_stats(unit="bottleneck choice")`,
  NaN-padded to 20). **Distinct** from the corridor-transition matrices above — this is the
  scalar choice-preference series feeding the bottleneck-choice panel, not the full transition
  matrix (written by `gen_wildtype_d_data.py`).
- **Additional curve-fit ratio tables.** Beyond the two-day and independent-cohort tables above,
  `gen_curve_fits.py` writes several more ratio tables that share the same tidy ratio-table schema
  (`Ratio, CI_lower, CI_upper` + the `sat_frac_*`/`mask_frac` diagnostics): `"* mask param ratios"`,
  `"Wildtype day21 mask BC param ratios"`, `"Mask D {metric} genotype param ratios"`, the
  generalization genotype-ratio tables, and the per-animal `"<base> per-animal params"` (`.parquet`,
  Control A only). Their table→panel mapping is documented in
  [`ratio_ci_method.md`](ratio_ci_method.md#table--panel-map) ("Table → panel map").

### Error-propagation and model-free-RL arrays

Written by `gen_error_propagation.py`, consumed by `plot_error_propagation_supp.py`.
These compare the per-position animal error against a model-free-RL null; full model and
figure walk-through in [`rl_error_propagation.md`](rl_error_propagation.md).

- **`"Wildtype A corridor error by position"`** (`.pkl`): `dict`
  `{"n_pos": int, "n_animals": {dir: int}, "H-O": arr, "O-H": arr}` where each `arr` is
  `(n_pos, n_traverses)` corridor-error rate indexed by distance-to-reward and traverse,
  per travel direction.
- **`"Wildtype A hole error by position"`** (`.pkl`): `dict` `{"H-O": [...], "O-H": [...]}`,
  one per-hole `(n_animals, n_traverses)` first-decision turn-error array per hole (list
  ordered close→far), per direction.
- **`"Mask A RL corridor staircase"`** (`.npy`): `(n_pos − 1, n_traverses)` — the closed-form
  model-free-RL backward-propagation prediction for corridor error (from
  `rl_model.analytic_rl_staircase`, `dead_end_last=True` pins the start corridor to 0).
- **`"Mask A RL turn staircase"`** (`.npy`): `(n_holes, n_traverses)` — the same closed-form
  prediction for per-hole turn error (no forced dead-end position).
- **Validation-only** (not figure inputs, and **excluded from `batch_generate_figure_data.py`**):
  `"Mask A model-free RL error by position"` / `"Mask A model-free RL turn error by position"`
  (`.pkl` dicts), the stochastic-simulation cross-checks of the closed-form staircase, written by
  `gen_rl_simulation.py` / `gen_rl_turn_simulation.py`.

### Model-comparison and Mask-A first-journey diagnostics

- **`"Mask D model comparison"`** (`.pkl`): `dict` with keys `"H-O"`, `"O-H"`, `"meta"`.
  Each direction dict holds `q_err` and `q_bn`, both `(n_seeds, L) = (20, 25)` `float64`:
  `q_err` is the self-play Q-learning **corridor-error rate** per traverse (mean over
  `N_WALKS=300` readout rollouts, `error_type="rate"`), `q_bn` is `P(gateway → bottleneck)`
  ∈ [0, 1] per traverse. `meta` carries the run scalars `L=25`, `BN_SIZE=10`, `n_seeds=20`,
  `gamma=0.9`, `alpha=0.5`, `gateway={"H-O":19,"O-H":12}`, `bottleneck=1`, and `e_half`
  (`{dir: float}`, the β=0.5 memoryless-null per-step error rate `E/(2E+L)` used as the
  traverse-1 anchor for the parameter-free Endotaxis step computed inline in the plot).
  Written by `gen_maskd_model_comparison.py`, consumed by `plot_algorithm.py` (fig:algo, Panels A–D).
- **`"{Wildtype|Acortical|Control} A first journey forward bias"`** (`.npy`): `(3, 18)` `float64` —
  row 0 = normalized position along the merged pre-reward first journey, on a **closed** grid
  `np.linspace(0, 1, 18)` (so the extreme points are exactly the journey's start and end),
  row 1 = cohort-mean forward-bias / directional-persistence β̂ ∈ [0, 1] (0.5 = memoryless
  walker), row 2 = SE across animals. **Not smoothed:** each point is a single β̂ fit over the
  decisions within ±10% of it, so the 20%-wide window is the only smoothing and the curve's
  bandwidth is exactly that. Only **fully-supported** positions are reported (`mode="valid"`,
  the `utils.moving_average` convention also used for the smoothed lines of `fig:ac_mem_gen` A):
  a window must fit entirely inside the journey, so the two grid points at each end are NaN and
  **14 of the 18 points carry data, x ≈ 0.118–0.882**. Pass `mode="same"` to the estimator to
  additionally evaluate the truncated end windows and span a full 0→1, at the cost of end points
  resting on ~half the decisions. Adjacent points share ~70% of their decisions and are therefore
  **correlated** — readable as a continuous estimate, but not as independent samples. Also NaN
  where < 2 animals contribute to a point; an animal contributes only where its window holds ≥ 2
  scorable decisions, and animals whose collapsed first journey is shorter than 6 corridors are
  excluded outright. All three are
  produced by the one estimator, `utils.first_journey_forward_bias_curve` (cohort layer in
  `analysis.py`, the per-decision β̂ in `random_walk.reversal_decisions` +
  `random_walk.forward_bias_mle`).
  Cohorts differ by group and are *not* a common selection rule: Wildtype is Day-1 Mask-A BL6J
  intersected with the two-day animals (n=25, `gen_wildtype_two_day_data.py`); Acortical (n=4)
  and Control (n=3) are the strict-first Mask-A learners (`gen_acortical_learning.py`).
  Consumed by `plot_oa_supp.py` (Panel I, Wildtype only) and `plot_ac_oa_supp.py` (Panel I,
  all three overlaid against the 0.5 null).
- **`"Wildtype two day first journey forward bias"`** (`.parquet`): tidy frame, one row per
  (session, mask, grid point) — 13 cells × 18 points = **234 rows**. Columns: `Session`
  (`int`; 1 = Day 1, 2–5 = Day2-1…Day2-4, the same numbering as
  `analysis.get_two_day_data_df` and the `"Wildtype two day {data_type} fit results"` keys),
  `Mask` (`str`; Day 1 is Mask A only, each Day-2 session carries A/B/C), `n_animals` (`int`;
  the cell's cohort size, 6–11 on Day 2 and 25 on Day 1), `x` (grid position, `linspace(0, 1, 18)`),
  `beta` (cohort-mean β̂) and `se` (SE across animals). Estimator, window, edge handling and NaN
  policy are **identical** to `"{Wildtype|Acortical|Control} A first journey forward bias"` above
  — the same `utils.first_journey_forward_bias_curve` at its defaults — so the same caveats apply
  (14 of 18 points populated per cell, x ≈ 0.118–0.882; adjacent points ~70% correlated). By
  construction the `Session == 1, Mask == "A"` rows **reproduce that `.npy` key exactly**, same
  cohort and same defaults, which is the cross-check on this table.
  ⚠️ **The Day-2 rows are not a latent-learning readout.** "First journey" there means only
  *before this session's first reward*; those animals already earned rewards on Day 1, so the
  Day-2 curves measure how forward-biased an experienced animal is on re-entering a maze, not
  reward-free structure learning. Only the Day-1 row supports the latent-learning reading.
  Written by `gen_wildtype_two_day_data.py`; not used by any manuscript figure.
- **`"Wildtype A hole by hole error rate"`** (`.pkl`): `dict` `{"H-O": [...], "O-H": [...]}`;
  each value is a list of length `n_holes` (ordered close→far from the corridor start), each
  element an `(n_animals, n_traverses/2)` `float64` approach-conditioned per-hole turn-error
  rate ∈ [0, 1] (NaN-padded). **Diagnostic** — written by `gen_wildtype_two_day_data.py`;
  no active plot consumer.
- **`"Overnight traverse example …"`** (`.parquet`, four keys): the Day-1-first /
  Day-1-late (10th / 21st) / Day-2.1-first traverses of each example animal, as flat tables
  (see *Example-session tables*). The former nested `[outbound, homebound][set][animal]`
  structure survives as the `direction` / `set_idx` / `animal_idx` columns on `bout meta` and
  `manifest`; `label` holds the traverse index. Written by `gen_wildtype_two_day_data.py`,
  consumed by `plot_day2.py` (fig:day2 example strips), which draws `animal_idx == 0` only.

### Single-north housing keys

The `"Single north …"` family (written by `gen_wildtype_d_data.py` for the single-north
housing condition, sessions Day-1 D and Day-2 D/A/C) follows the per-metric patterns above:
`"Single north {condition} reward intervals"` / `"… sortie counts"` (`(n, 30)`),
`"Single north {condition} traverse duration"` / `"… traverse turn error rate"` (`(n, 40)`;
the two Mask-D conditions omit turn error rate), and
`"Single north pooled sortie count by direction"` (`(n_sessions, 2)`, columns `[H-H, O-O]`,
pooled across all single-north sessions — pseudoreplicated, so treated as a paired
within-session comparison).

### Invariants

- NaN in metric arrays indicates that an animal did not reach that traverse count.
  Arrays are always right-padded with NaN (not left-padded), so `array[:, 0]` is always
  the first traverse across all animals.
- `reward intervals` in figure_data (from `Session.get_slice_stats`) are in **seconds**
  and represent inter-traverse intervals measured by in-maze bout timing — distinct from
  `Session.reward_interval_seconds` (see §10).
- Bootstrap confidence intervals use the **median** of bootstrap parameter estimates, not the
  original-data fit, as the central curve. The original-data fit is not stored separately.

### Failure modes

- `save_modular_data` with `overwrite=False` silently skips writing without checking if the
  file contents are current. Stale `.npy`/`.pkl` files are indistinguishable from fresh ones.
- `load_all_figure_data` loads every file in the directory. An incorrectly named or corrupted
  file silently produces a wrong dict key or raises an exception for all subsequent figures.
- `.pkl` files hold plain containers (lists, dicts, fitted-curve tuples) plus the allowlisted
  `masks.pkl`. Only `masks.pkl` would fail to unpickle after a class move; everything else is
  format-stable.
- Keys are filename *stems*, so the same key written in two formats (a stale `.pkl` left beside
  its `.npy`/`.parquet` replacement) would resolve by directory order. `load_all_figure_data`
  now raises on duplicate stems rather than silently serving the stale one.

---

## 13. Random-walk completion metrics (`random_walk.py`)

**Produced by**: `manhattan_maze/random_walk.py` — `completion_time`,
`expected_corridor_errors`, `effective_forward_bias`, `walker_metrics`.
**Consumed by**: manuscript methods `sec:walker`. **Not persisted** to `data/figure_data/`
(on-demand scalars, no file schema). Documented here because they are manuscript-facing
numbers pinned by `tests/test_random_walk.py` — **do not change numerical behaviour without
updating this document and that test.**

### Model and units

- **Forward bias `beta`** — dimensionless, `(0, 1]`. `beta = 1/2` = memoryless (uniform) walk;
  `beta = 1` = never reverses. Passed to `graph.first_order_average_steps` /
  `first_order_transition_matrix` as `probability`.
- **`completion_time`** (`walker_metrics["completion_time"]`) — expected number of graph
  transitions from start to goal. Units: corridor transitions (`unit="corridor"`) or tile
  transitions (`unit="tile"`). Ports/graph read from the `Mask` object.
- **`expected_corridor_errors`** (`walker_metrics["expected_errors"]`) — expected
  distance-increasing steps per traverse (corridor errors on the corridor graph; tile errors
  on the tile graph). Distance-to-goal from `mask.corridors_shortest_distance[goal, :]`.
- **`effective_forward_bias`** — solves `E(beta_hat) = observed corridor error` on the monotone
  `E(beta)` (default search `[0.5, 1.0]`). Units: dimensionless bias.
- **Bipartite identity**: on the corridor graph, `E(beta) = (tau(beta) - L)/2` with `L` the
  shortest-path length in holes (every edge changes goal distance by exactly one).

### Reference values (Home → Out, corridor graph)

| Graph | `tau(1/2)` | `tau(1)` | `E(1/2)` | `E(1)` |
|---|---|---|---|---|
| P₁₀ (Masks A–C) | 81 | 9 | 36 | 0 |
| Mask D | 166.75 | 92.587 | 80.875 | 43.793 |

---

## Quick Reference: Critical Scientific Constants

| Constant | Value | Location | Effect if changed |
|----------|-------|----------|-------------------|
| `FPS` | 30 | `DataLoader.__init__` | All durations, reward intervals |
| `home_coordinates` | `(0, 5, 0)` | `DataLoader.__init__` | Bout type classification |
| `out_coordinates` | `(5, 9, 1)` | `DataLoader.__init__` | Bout type classification |
| `sleep_threshold` | 5 s | `Bout.get_duration_s` | All duration values |
| `min_frames_per_cell` | 1 | `DataLoader.__init__` | Cell sequence condensation |
| `maze_size` | 11 | `DataLoader.__init__` → `Mask` | All tile/corridor indices |
| Guaranteed transitions (adjusted Jaccard) | 3 (Mask D only; 0 otherwise) | `utils.transition_vec_similarity(n_guaranteed_transitions=…)` | Similarity matrices |
| Bootstrap iterations | 1000 | `gen_*.py` call sites | CI width reproducibility |
| Bootstrap random seed | not fixed globally | varies per script (`np.random.seed(0)`) | Reproducibility |
| Forward-bias convention (random walk) | β=1/2 memoryless, β=1 never-reverse | `random_walk.py` | Completion-time / corridor-error baselines (see §13) |
