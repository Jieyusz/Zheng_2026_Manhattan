# Manhattan Maze — Notation and Symbol Guide

A standalone reference for the mathematical notation used in the Manhattan Maze
manuscript. It collects every equation and defines every symbol, with units. Intended for
readers, coauthors, and reviewers; it is self-contained and does not require the code.

## Conventions

- **Traverse index $b$** is a positive integer with $b=1$ the first completed traverse.
- **Tile coordinates** are 0-based, $(x, y, z)$ with $x,y \in [0,10]$ and $z \in \{0,1\}$
  ($z=0$ horizontal layer, $z=1$ vertical layer).
- **Corridor indices** run 0–21 (0–10 horizontal, 11–21 vertical).
- **Indices are local to each equation.** In the learning-curve models (Eqs. 1–2) $a$ is the
  animal and $b$ the traverse; in the similarity matrices (Eqs. 7–10) the row/column indices are
  $r$ and $s$ (traverse-within-direction). Note $s$ also serves as the session index in the
  relative-magnitude estimates (Eqs. 12–15) — its meaning is local to each equation. The
  bottleneck corridor is always written with a subscripted corridor label $c_b$, distinct from
  the traverse index $b$.
- **Units** are seconds throughout (never frames or minutes).

---

## 1. Equation Inventory

### Learning-curve models

**Eq. 1 — Traverse-duration learning curve** (population-level exponential decay of traverse
duration over successive traverses, with per-traverse animal residuals):

$$D_{a,b} = D_{\infty} + \left(D_0 - D_{\infty}\right)\exp\left[-\delta(b-1)\right] + \xi^D_{a,b}$$

**Eq. 2 — Turn-error-rate learning curve** (same form for turn error rate):

$$E_{a,b} = E_{\infty} + \left(E_0 - E_{\infty}\right)\exp\left[-\epsilon(b-1)\right] + \xi^E_{a,b}$$

Fit independently per condition by nonlinear least squares; parameter uncertainty is estimated
by animal-level bootstrap. Bounds and initial values are listed in the glossary below.

### Effect-size (relative-magnitude) estimates

**Eq. 11 — Parameter ratio per bootstrap replicate** (fold-change of a fitted parameter
$\theta$ between conditions $A$ and $B$):

$$R_\theta^{(\ell)} = \frac{\theta_A^{(\ell)}}{\theta_B^{(\ell)}}$$

**Eq. 12 — Day-2 / Day-1 parameter ratio** (per session $s$, mask $m$, replicate $\ell$):

$$R_{\theta,s,m}^{(\ell)} = \frac{\theta_{\mathrm{Day~2},s,m}^{(\ell)}}{\theta_{\mathrm{Day~1}}^{(\ell)}}$$

**Eq. 13 — Median Day-2 ratio** (across sessions/masks within a replicate):

$$R_{\theta,\mathrm{median}}^{(\ell)} = \mathrm{median}_{s,m}\left(R_{\theta,s,m}^{(\ell)}\right)$$

**Eq. 14 — Last observed traverse per session:**

$$b_{\max}^{(s)} = \max\{\,b : X_s(b)\ \mathrm{was\ observed}\,\}$$

**Eq. 15 — Shared late-traverse threshold** (latest traverse observed in every compared
session $\mathcal{S}$):

$$b_{\mathrm{late}} = \min_{s \in \mathcal{S}} b_{\max}^{(s)}$$

**Eq. 16 — Data-anchored late performance** (fitted curve evaluated at $b_{\mathrm{late}}$;
reported as $D_{\mathrm{late}}$ / $E_{\mathrm{late}}$, used instead of the weakly-identified
asymptotes; $k$ denotes the corresponding rate, $\delta$ for $D$ and $\epsilon$ for $E$):

$$X_{\mathrm{late}} = X_\infty + (X_0 - X_\infty)\exp\!\left[-k\,(b_{\mathrm{late}} - 1)\right]$$

### Directional transitions in Mask D

**Eq. 3 — Journey-level transition counts** (total corridor $c_i\to c_j$ transitions across all
bouts $q$ in a journey):

$$N_{i,j} = \sum_q T^{(q)}_{i,j}$$

**Eq. 4 — Bottleneck choice ratio** (measured at the bottleneck-adjacent corridor $c_g$, the
direction-appropriate neighbour of the bottleneck, as the fraction of transitions out of $c_g$
that enter the bottleneck $c_b$; uniform-choice chance is $1/|\mathcal{N}^{+}(c_g)|$):

$$R_{c_g,c_b} = \frac{N_{c_g,c_b}}{\displaystyle\sum_{k \in \mathcal{N}^{+}(c_g)} N_{c_g,k}}$$

### Traverse similarity in Mask D

**Eq. 5 — Adjusted Jaccard similarity** between two traverses' directed corridor-transition
sets, minus the 3 transitions guaranteed by the Mask D bottlenecks:

$$J\!\left(T^{(p)},T^{(q)}\right) =
\frac{
\displaystyle\sum_{i,j} \mathbb{I}\!\left(T^{(p)}_{i,j} > 0 \;\mathrm{and}\; T^{(q)}_{i,j} > 0\right) - 3
}{
\displaystyle\sum_{i,j} \mathbb{I}\!\left(T^{(p)}_{i,j} > 0 \;\mathrm{or}\; T^{(q)}_{i,j} > 0\right) - 3
}$$

$J=0$ means two traverses shared only the obligatory transitions; $J=1$ means identical directed
transition sets. The $-3$ correction is specific to Mask D (three guaranteed transitions).

**Eq. 6 — Similarity matrix types** for a session with $m$ outbound and $n$ homebound traverses:

$$J_{\mathrm{O,O}} \in \mathbb{R}^{m \times m}, \qquad J_{\mathrm{H,H}} \in \mathbb{R}^{n \times n}, \qquad J_{\mathrm{O,H'}} \in \mathbb{R}^{m \times n}$$

**Eq. 7 — Outbound self-similarity entry** ($T^{(\mathrm{O}_r)}$ is the $r$-th outbound traverse):

$$J_{\mathrm{O,O}}(r,s) = J\!\left(T^{(\mathrm{O}_r)},T^{(\mathrm{O}_s)}\right)$$

**Eq. 8 — Homebound self-similarity entry** ($T^{(\mathrm{H}_s)}$ is the $s$-th homebound traverse):

$$J_{\mathrm{H,H}}(r,s) = J\!\left(T^{(\mathrm{H}_r)},T^{(\mathrm{H}_s)}\right)$$

**Eq. 9 — Reversed homebound transition matrix** (transpose, to compare in the outbound
direction):

$$T^{(\mathrm{H'}_s)}(i,j) = T^{(\mathrm{H}_s)}(j,i)$$

**Eq. 10 — Outbound–homebound cross-similarity entry:**

$$J_{\mathrm{O,H'}}(r,s) = J\!\left(T^{(\mathrm{O}_r)},T^{(\mathrm{H'}_s)}\right)$$

### First-order Markov walker (completion time, corridor errors, forward-bias estimate)

Non-learning baseline of `sec:walker` (implemented in `manhattan_maze/random_walk.py`).
Equation numbers are assigned at manuscript compile time; `\label`s are given for reference.

**Transition rule** (`eq:transition`) — forward bias $\beta \in (0,1]$; the state is a
directed edge $(j \to i)$ ("at node $i$, arrived from $j$"); move to a neighbour $l$ of $i$
($A_{il}=1$):

$$
S_{(i\to l),\,(j\to i)} \propto
\begin{cases} \beta, & l \neq j \ (\text{forward}),\\ 1-\beta, & l = j \ (\text{reversal}), \end{cases}
$$

normalised over the neighbours of $i$ (forced reversal at a dead end). Collected over all
directed edges, $S$ is column-stochastic.

**First-passage recurrence** (`eq:recurrence`) — edge-states indexed $u, v$; goal node $e$
made absorbing (every state stepping into $e$):

$$
\tau_u = \begin{cases} 0, & u \text{ enters } e,\\ 1 + \sum_v S_{vu}\,\tau_v, & \text{otherwise.} \end{cases}
$$

**Fundamental-matrix solution** (`eq:solution`) — over the transient states,
$\boldsymbol\tau' = (\mathbf I - S'^\top)^{-1}\mathbf 1$, with fundamental matrix
$\mathbf F = (\mathbf I - S'^\top)^{-1}$. The completion time $\bar\tau(\beta)$ averages these
over the directed edges entering the start node.

**Memoryless special case** (`eq:memoryless`, $\beta = 1/2$) — the chain reduces to a walk
over nodes with $S_{ij} = A_{ij}/\sum_k A_{kj}$.

**Corridor-error identity** — the corridor graph is bipartite, so every step changes the goal
distance by one and the expected corridor errors satisfy
$\mathcal{E}(\beta) = [\bar\tau(\beta) - L]/2$, with $L$ the shortest-path length in holes.
Dividing by the walk length gives the per-step **corridor error rate**
$\rho(\beta) = \mathcal{E}(\beta)/\bar\tau(\beta) = \tfrac12\!\left(1 - L/\bar\tau(\beta)\right)$,
a fraction in $[0,\tfrac12]$ that approaches the memoryless chance level $\tfrac12$ as the walk
lengthens. ($\mathcal{E}$ is the raw count and $\rho$ its per-step rate; neither uses the
turn-error symbol $E$.)

**Empirical forward bias** (`eq:prev`, `eq:betahat`) — a direct estimate of $\beta$ from an animal's
own path, read from the model's reversal probability rather than by inverting the error curve. At an
interior node of degree $g=\sum_l A_{il}$, the transition rule gives the reversal probability

$$p_{\mathrm{rev}}(\beta,g) = \frac{1-\beta}{(1-\beta)+(g-1)\beta},$$

which decreases monotonically from the memoryless value $1/g$ to $0$ at $\beta=1$. Writing $c_t$ for the
corridor at step $t$ of the run-length-collapsed trajectory, a decision is a reversal when
$c_{t+1}=c_{t-1}$ (the start and end corridors are excluded). Over $N_{\mathrm{dec}}$ corridor decisions
of which $N_{\mathrm{rev}}$ are reversals, with $g_t=\deg(c_t)$ the degree at decision $t$, the empirical
forward bias $\hat\beta$ matches the model's expected reversals to the observed count,

$$\sum_{t=1}^{N_{\mathrm{dec}}} p_{\mathrm{rev}}(\hat\beta,g_t)=N_{\mathrm{rev}},$$

a single monotone root-find. On the
linear track ($g=2$) this collapses to $\hat\beta = 1-N_{\mathrm{rev}}/N_{\mathrm{dec}}$; at
degree-$>2$ junctions (Mask D) the degree-general form is used. Because $p_{\mathrm{rev}}$ is monotone
in $\beta$, $\hat\beta$ is always defined and needs no goal, so it applies to reward-free sorties and to
Mask D, where $\rho(\hat\beta)=$ observed rate has no solution.

Reference values (Home → Out, corridor graph): P₁₀ — $\bar\tau(1/2)=81$, $\bar\tau(1)=9$,
$\mathcal{E}(1/2)=36$, $\mathcal{E}(1)=0$, $\rho(1/2)=0.44$, $\rho(1)=0$; Mask D —
$\bar\tau(1/2)=166.75$, $\bar\tau(1)=92.587$, $\mathcal{E}(1/2)=80.875$, $\mathcal{E}(1)=43.793$,
$\rho(1/2)=0.49$, $\rho(1)=0.47$.

**Related null model.** A second, learning-based null — the model-free-RL
error-propagation staircase (backward propagation of error from the reward, indexed by
distance-to-reward $d$ and traverse $b$) — is documented separately in
`docs/rl_error_propagation.md` (implemented in `manhattan_maze/rl_model.py`,
`analytic_rl_staircase`). Its notation is self-contained there and not repeated in this
guide. The model-free-RL agent has a discount and a learning rate, but its greedy (argmax)
readout depends only on the *order* in which positions acquire reward value, not on those
magnitudes; they therefore do not affect the reported prediction — on Mask A or on the Mask-D
self-play of `fig:algo` — and are omitted from the Methods. Consequently $\gamma$ and $\alpha$
appear as manuscript symbols only for the endotaxis gain and goal-learning rate (see *Endotaxis
model parameters* below).

---

## 2. Symbol Glossary

### Index symbols

| Symbol | Meaning | Range |
|--------|---------|-------|
| $a$ | Animal index (Eqs. 1–2) | Integer ≥ 1 |
| $b$ | Traverse index (Eqs. 1–2) | Positive integer, $b=1$ first traverse |
| $r$ | Similarity-matrix row index: the $r$-th traverse within a direction (Eqs. 7–10) | Positive integer |
| $s$ | Session index (Eqs. 12–15); similarity-matrix column index (Eqs. 7–10) | Positive integer |
| $i$, $j$ | Corridor indices (row = source, column = destination) | 0–21 |
| $k$ | Off-diagonal offset in similarity analysis; also the generic rate placeholder in Eq. 16 | Integer ≥ 0 |
| $m$ | Number of outbound traverses in a session (dimension of $J_{\mathrm{O,O}}$) | Integer ≥ 0 |
| $n$ | Number of homebound traverses in a session (dimension of $J_{\mathrm{H,H}}$) | Integer ≥ 0 |
| $p$, $q$ | Bout/traverse indices for pairwise similarity (Eq. 5) | Integer ≥ 0 |
| $u$, $v$ | Directed-edge state indices of the first-order walker (state $(j\to i)$) | Integer ≥ 0 |
| $\ell$ | Bootstrap replicate index | Integer ≥ 1 |
| $t$ | Position/step index along a corridor trajectory ($c_t$); also the corridor-step counter in `fig:algo` E–G | Integer ≥ 1 |

### Duration model (Eq. 1)

| Symbol | Meaning | Type | Units |
|--------|---------|------|-------|
| $D_{a,b}$ | Traverse duration of animal $a$ on traverse $b$ | Raw data | seconds; capped at 5 s per tile |
| $D_0$ | Fitted initial traverse duration at $b=1$ | Fitted parameter | seconds |
| $D_\infty$ | Fitted asymptotic traverse duration as $b \to \infty$ | Fitted parameter | seconds |
| $\delta$ | Non-negative exponential decay (learning) rate for duration | Fitted parameter | 1/traverse |
| $\xi^D_{a,b}$ | Per-animal per-traverse residual from the population curve | Residual | seconds |
| $\sigma_D^2$ | Assumed residual variance for the duration model | Model assumption | seconds² |

Bounds: $2 \leq D_\infty \leq 60$, $5 \leq D_0 \leq 800$, $0.01 \leq \delta \leq 1$.
Initial values: $D_\infty = 20$, $D_0 = 200$, $\delta = 0.1$.

### Turn-error model (Eq. 2)

| Symbol | Meaning | Type | Units |
|--------|---------|------|-------|
| $E_{a,b}$ | Turn error rate of animal $a$ on traverse $b$ | Raw data | Fraction 0–1 (first-decision per hole: fraction of distinct decision holes whose first shortest-path-corridor crossing turned the wrong way) |
| $E_0$ | Fitted initial turn error rate at $b=1$ | Fitted parameter | Dimensionless |
| $E_\infty$ | Fitted asymptotic turn error rate | Fitted parameter | Dimensionless |
| $\epsilon$ | Non-negative exponential decay (learning) rate for error | Fitted parameter | 1/traverse |
| $\xi^E_{a,b}$ | Per-animal per-traverse residual for the error model | Residual | Dimensionless |
| $\sigma_E^2$ | Assumed residual variance for the error model | Model assumption | Dimensionless |

Bounds: $0.001 \leq E_\infty \leq 0.5$, $0.1 \leq E_0 \leq 1$, $0.01 \leq \epsilon \leq 1$.
Initial values: $E_\infty = 0.1$, $E_0 = 0.5$, $\epsilon = 0.1$.

$E_{a,b}$ is the **first-decision-per-hole, approach-conditioned** rate: for each distinct decision
hole on the shortest path, only the *first* crossing entered on the shortest-path corridor is scored,
and $E_{a,b}$ is the fraction of those first decisions made in the wrong direction. From that approach
exactly two turn outcomes are possible (one correct), so the chance level is exactly 0.5. Scoring one
first decision per hole makes the denominator the geometry-fixed count of distinct decision holes:
reversals that re-cross a hole cannot inflate it, and every traverse is directly comparable. This is
`Bout.count_error(include="first")`, the default. The older pooled scoring is deprecated —
`include="approach"` pools every crossing (endogenous denominator, inflated by reversals) and
`include="all"` additionally counts forced errors from perpendicular-corridor crossings whose correct
direction is unreachable; both emit a `DeprecationWarning`.

### Relative-magnitude estimates (Eqs. 11–16)

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\theta$ | Any fitted parameter compared across conditions ($D_0, D_\infty, \delta, E_0, E_\infty, \epsilon$) | (as parameter) |
| $R_\theta^{(\ell)}$ | Fold-change of $\theta$ between two conditions, per replicate | Ratio |
| $R_{\theta,s,m}^{(\ell)}$ | Day-2/Day-1 ratio for session $s$, mask $m$ | Ratio |
| $R_{\theta,\mathrm{median}}^{(\ell)}$ | Median of $R_{\theta,s,m}$ across sessions/masks within a replicate | Ratio |
| $m$ | Mask condition index (this section) | — |
| $b_{\max}^{(s)}$ | Highest observed traverse number in session $s$ | Traverse # |
| $b_{\mathrm{late}}$ | Latest traverse number observed in every compared session | Traverse # |
| $\mathcal{S}$ | Set of sessions in a comparison | — |
| $D_{\mathrm{late}}$, $E_{\mathrm{late}}$ | Data-anchored late performance (curve at $b_{\mathrm{late}}$) | seconds / dimensionless |

### Transition matrix and choice ratio (Eqs. 3–4)

| Symbol | Meaning | Units |
|--------|---------|-------|
| $T^{(q)}$ | Directed corridor transition matrix for bout $q$ | 22×22 integer matrix |
| $T^{(q)}_{i,j}$ | Count of $c_i \to c_j$ transitions in bout $q$ | Count ≥ 0 |
| $c_i$, $c_j$ | Corridor labels | Integer 0–21 |
| $c_b$ | Bottleneck corridor (fixed by mask geometry) | Integer 0–21 |
| $c_g$ | Bottleneck-adjacent corridor where the choice is scored (direction-specific neighbour of $c_b$: outbound 19, homebound 12) | Integer 0–21 |
| $N_{i,j}$ | Sum of $T^{(q)}_{i,j}$ across all bouts in one journey | Count |
| $R_{c_g,c_b}$ | Choice ratio at $c_g$ toward the bottleneck $c_b$ (uniform chance $1/|\mathcal{N}^{+}(c_g)|$) | Fraction 0–1 |
| $\mathcal{N}^{+}(c_g)$ | Set of corridors reachable in one step from $c_g$ | Set of corridor indices |

### Adjusted Jaccard similarity (Eqs. 5–10)

| Symbol | Meaning | Units |
|--------|---------|-------|
| $J(T^{(p)}, T^{(q)})$ | Adjusted Jaccard similarity between traverses $p$ and $q$ | Dimensionless, 0–1 |
| $\mathbb{I}(\cdot)$ | Indicator function (1 if true, else 0) | — |
| $J_{\mathrm{O,O}}$ | $m \times m$ self-similarity among outbound traverses | Dimensionless matrix |
| $J_{\mathrm{H,H}}$ | $n \times n$ self-similarity among homebound traverses | Dimensionless matrix |
| $J_{\mathrm{O,H'}}$ | $m \times n$ cross-similarity: outbound vs. reversed homebound | Dimensionless matrix |
| $T^{(\mathrm{H'}_s)}$ | Transpose of $T^{(\mathrm{H}_s)}$; converts homebound → outbound direction | 22×22 integer matrix |

### First-order Markov walker (`sec:walker`)

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\beta$ | Forward bias of the walker ($1/2$ = memoryless, $1$ = never reverse) | Dimensionless, $(0,1]$ |
| $A$ | Maze adjacency matrix ($A_{ij}=1$ if nodes $i,j$ are connected) | 0/1 matrix |
| $S$ | Column-stochastic transition matrix over directed-edge states | — |
| $e$ | Goal (absorbing) node | node index |
| $\tau_u$ | Expected steps to first reach $e$ from state $u$ | steps |
| $\boldsymbol\tau'$ | Transient hitting-time vector | steps |
| $\mathbf F$ | Fundamental matrix $(\mathbf I - S'^\top)^{-1}$ | — |
| $\bar\tau(\beta)$ | Completion time from the start node (averaged over entering edges) | steps (corridor or tile transitions) |
| $\mathcal{E}(\beta)$ | Expected corridor errors (distance-increasing steps) | count per traverse |
| $\rho(\beta)$ | Corridor error **rate** $=\mathcal{E}(\beta)/\bar\tau(\beta)$ | Fraction 0–1 (chance ~0.5) |
| $\hat\beta$ | Empirical forward bias, matching model reversals to observed ($\sum_{t=1}^{N_{\mathrm{dec}}} p_{\mathrm{rev}}(\hat\beta,g_t)=N_{\mathrm{rev}}$, `eq:betahat`); on the linear track $\hat\beta=1-N_{\mathrm{rev}}/N_{\mathrm{dec}}$ | Dimensionless, $(0,1]$ |
| $L$ | Shortest-path length from start to goal | holes |
| $p_{\mathrm{rev}}(\beta,g)$ | Reversal probability at an interior node of degree $g$ (distinct from bout index $p$) | Fraction 0–1 |
| $g$ | Node (corridor) degree, $g=\sum_l A_{il}$; $g_t=\deg(c_t)$ at decision $t$ | Integer ≥ 1 |
| $c_t$ | Corridor occupied at position $t$ of the run-length-collapsed trajectory (sequence; vs. label $c_i$) | corridor index |
| $N_{\mathrm{dec}}$ | Number of corridor decisions scored (start and end corridors excluded) | Count |
| $N_{\mathrm{rev}}$ | Number of reversals ($c_{t+1}=c_{t-1}$) among those decisions | Count |

### Endotaxis model parameters

| Symbol | Meaning | Value used |
|--------|---------|-----------|
| $\gamma$ | Gain — controls propagation of activity through the learned map | 0.21 |
| $\theta$ | Threshold — determines when map-cell connections are updated | 0.2 |
| $\alpha$ | Goal-learning rate — updates goal-cell synapses after reward | 0.2 |
| decay | Synaptic decay rate — set to 0 to retain connections | 0 |

The endotaxis goal-learning rate $\alpha$ is distinct from the behavioral learning rates
$\delta$ (duration) and $\epsilon$ (turn error).

### Model error-rate predictions (`eq:endo_maskA`, `eq:endo_maskD`, `eq:rl_maskA`)

Analytic predictions from the candidate models, plotted against the mouse error curves.
Corridor error rate uses $\rho$, turn error rate uses $E$ (matching the reserved data
symbols); where a model predicts the same value for both metrics they are written jointly
(e.g. $\rho_{\text{Endo}} = E_{\text{Endo}}$).

| Symbol | Meaning | Value |
|--------|---------|-------|
| $b$ | Traverse index ($b=1$ the first traverse) | Integer $\ge 1$ |
| $d$ | Distance to reward at a decision position | holes |
| $\rho_{\text{RL}}(d,b)$, $E_{\text{RL}}(d,b)$ | Model-free RL staircase: corridor / turn error rate | $0.5$ if $b \le d$, else $0$ |
| $\rho_{\text{Endo}}(d,b)$, $E_{\text{Endo}}(d,b)$ | Endotaxis Mask A: corridor / turn error rate | $0.5$ if $b=1$, else $0$ |
| $\rho_{\text{Endo}}(b)$ | Endotaxis Mask D corridor error rate | $\rho(\tfrac12)$ if $b=1$, else $0$ |
| $R_{\text{Endo}}(b)$ | Endotaxis Mask D bottleneck choice ratio | $1/\lvert\mathcal{N}^{+}(c_g)\rvert$ if $b=1$, else $1$ |
| $c_g$ | Gateway corridor into the bottleneck | corridor index |
| $\mathcal{N}^{+}(c_g)$ | Out-neighbours of $c_g$ ($\lvert\mathcal{N}^{+}(c_g)\rvert = 5$, so chance $0.2$) | set |

---

## 3. Additional notation

### Maze graph notation

| Notation | Meaning |
|----------|---------|
| $\mathrm{P}_k$ | Path graph with $k$ corridors and $k-1$ holes (P₂ = Mask O, P₁₀ = Masks A/B/C, P₄ = Masks E/F) |
| $\mathrm{K}_{4,4}$ | Complete bipartite graph (biclique), 4+4 corridors, all-to-all connected |
| Mask D graph | $\mathrm{K}_{4,4} + \mathrm{P}_3 + \mathrm{K}_{4,4} + \mathrm{P}_2$ |
| Bipartite corridor graph | Corridors split into horizontal (0–10) and vertical (11–21) sets; holes connect only across sets, so every edge changes goal distance by exactly 1 |
| $(x, y, z)$ | Tile coordinates: $x,y \in [0,10]$, $z \in \{0,1\}$ |
| H | Home port location $(0, 5, 0)$ |
| O | Out port location $(5, 9, 1)$ |

### Performance metrics

| Term | Meaning | Units |
|------|---------|-------|
| Reward interval | Time between successive rewards (in-maze time); the first interval spans the start of the first bout to the first reward | seconds |
| Traverse duration | Active traversal time, capped at 5 s/tile to remove immobility | seconds |
| Turn error rate | Fraction of distinct decision holes whose first shortest-path-corridor crossing turned off the shortest-path direction (first-decision per hole, approach-conditioned; chance 0.5) | Fraction 0–1 |
| Corridor error | Steps into a corridor farther from the goal port | Count per traverse |
| Corridor error rate | Fraction of steps that increase goal distance ($\mathcal{E}/\bar\tau$) | Fraction 0–1 (chance ~0.5) |
| Tile error | Steps (tile resolution) farther from goal: wrong turns, overshooting, backtracking | Count per traverse |
| Tiles per corridor | Mean tiles traveled within a single corridor visit | Count |
| Relative performance | Mean in a later session ÷ mean of the same mouse's first-10 Day-1 traverses | Ratio |

### Trajectory vocabulary

| Term | Definition |
|------|-----------|
| Session | A continuous period under one fixed mask configuration. |
| Bout | A trajectory segment that starts and ends at water ports. |
| Traverse | A bout between different ports (H→O or O→H). |
| Sortie | A bout between the same port (H→H or O→O). |
| Journey | All sorties from one port plus the subsequent traverse from that port. |
