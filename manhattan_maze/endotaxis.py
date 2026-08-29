
"""Endotaxis model for corridor-graph learning in Mask D.

The core learning routines (``map_lin``, ``Learn_Mouse_tr``) and their symbol
conventions are adapted from the original Endotaxis implementation by Markus
Meister: https://github.com/markusmeister/Endotaxis-2023
(see also Zhang & Meister, "Endotaxis," 2024). The short variable names
(``ga`` = gain gamma, ``th`` = threshold theta, ``al`` = goal-learning rate
alpha, ``de`` = synaptic decay) are kept verbatim from that upstream repo so the
code stays diff-comparable with the source; they are intentionally NOT renamed.
"""

import numpy as np
from manhattan_maze import plot_utils
import matplotlib.pyplot as plt

# Endotaxis learning parameters live in scripts/config.py
# (config.ENDOTAXIS_LEARNING_PARAMETERS); callers pass them into Learn_Mouse_tr.

def remove_out_skirt(array):
    """
    Input: any numpy array or list
    """
    array = np.copy(array)
    array -= 1
    array -= (array > 10).astype(array.dtype) * 2
    return array

def extract_corridor_seq(session, remove_skirt=True):
    '''
    Input: session
    Output: a list of corridor visits, in a sequence (remove outskirt)
    '''
    # remove the first corridor as it's the same as the previous bout
    corridors = session.concat_corridors_df().corridor.to_numpy(copy=True)
    # convert the corridors based on the 9x9 mask
    if not remove_skirt:
        return corridors
    else:
        return remove_out_skirt(corridors)

def map_lin(M,ga):
    '''
    Return V = map output for every point cell activation
    V[:,j] = map output with agent at j
    '''
    return np.linalg.inv(1/ga*np.eye(len(M))-M)

def Learn_Mouse_tr(A, P, ga,th,al,de,bi=False, goal=13):
    '''
    Learning both map and targets, using mouse trajectory but save the time info
    ga = gain
    th = threshold
    al = goal learning rate
    de = synaptic decay (not used here)
    rs = random seed
    bi = bidirectional synapse update?
    '''

    def set_graph():
        '''
        make changes to graph
        '''
        nonlocal N # binds N to the variable just outside this function
        if s==0:
            F[0][goal]=1 # signal appears at goal

    def learn_map():
        '''
        update the map synapses
        '''
        for j in np.where(u>th)[0]: # all cells with pre-before > th
            for i in notj[j]:
                if v[i]>th: # if post-after > th
                    M[i,j]=1 # potentiate that map synapse
                    if bi: # bidirectional update
                        M[j,i]=1 # same for the reverse synapse
                else:
                    M[i,j]*=dmm # let that map synapse decay
                    if bi: # bidirectional update
                        M[j,i]*=dmm # same for the reverse synapse

    def learn_goal():
        '''
        update the goal synapses
        '''
        for k in range(m): # for every goal
            e=G[k].dot(v) # goal signal
            f=F[k][p] # feature signal
            d=f-e # difference between feature signal and its prediction from the goal synapses
            if d>0: # if feature exceeds prediction
                G[k]+=al*d*v/v.dot(v) # increment those synapses by something prop to d and v
            else:
                G[k]*=np.exp(-de*v) # let that synapse decay prop to v

    n=len(A)
    F=np.zeros((1,n)) # Feature strength, single goal
    m=len(F) # number of goals
    N=[np.nonzero(a==1)[0] for a in A.T] # N[i] = list of nodes you can step to from node i
    notj=[[i for i in range(n) if i != j] for j in range(n)] # list of all i that are not j
    dmm=np.exp(-de) # multiplier for map synapse decay

    Ns=[] # save adjacencies
    Ms=[] # save the map synapses
    Gs=[] # save goal synapses
    M=np.zeros((n,n)) # erase map synapses
    G=np.zeros((1,n)) # erase goal synapses, single goal

    p=int(P[0]) # start point of the walk
    v=np.linalg.inv(1/ga*np.eye(n)-M)[:,p] # compute map output
    le = len(P) # length of the walk
    for s in range(le): # for each step along the walk
        set_graph() # change the graph if needed
        u=v # previous map output
        p=int(P[s]) # choose next node in random walk
        v=np.linalg.inv(1/ga*np.eye(n)-M)[:,p] # compute new map output
        learn_map() # update map synapses
        learn_goal() # update goal synapses
        Ns+=[N.copy()] # save adjacencies
        Ms+=[M.copy()] # save map synapses
        Gs+=[G.copy()] # save goal synapses

    return Ns,Ms,Gs

def analytic_endotaxis_step(n_positions, n_traverses, chance=0.5, dead_end_last=False):
    """Closed-form endotaxis error prediction: the synchronized one-step drop.

    The map-based analogue of :func:`manhattan_maze.rl_model.analytic_rl_staircase`, and
    the curve drawn in the endotaxis column of ``plot_error_propagation_supp.py`` (computed
    inline there — this is a parameter-free analytic function, not a figure-data cache).

    Endotaxis gets the SAME no-signal treatment as the model-free Q-agent: before the first
    reward there is no goal signal, so every readout is a random walk -> ``chance`` at every
    position on traverse 1 (column 0), identical to the RL staircase's first column. But
    endotaxis learns the whole graph MAP from movement, and one exploratory traverse covers
    every corridor of the (linear) Mask-A path, so a single reward then tags the goal and the
    gradient is correct at EVERY position at once: ``0`` from traverse 2 onward, all positions
    together (not the back-to-front frontier of the RL staircase). Direction-independent.

    Verified against the endotaxis simulation (:func:`random_walk_complete` +
    :func:`Learn_Mouse_tr` + :func:`endo_gradient_walk`) on the committed Mask-A geometry in
    ``tests/test_endotaxis_error_propagation.py``.

    Parameters
    ----------
    n_positions : int
        Number of path positions (rows), ordered close -> far (reward row already dropped).
    n_traverses : int
        Number of traverses (columns); column 0 = traverse 1 = pre-reward = chance everywhere.
    chance : float, default 0.5
        Pre-signal error level (binary decision on the linear path).
    dead_end_last : bool, default False
        If True, pin the far (start-corridor) position to 0 for all traverses -- the forced
        one-way dead end scored correct from traverse 1 (matches ``analytic_rl_staircase``;
        the corridor row needs it, the turn row does not).

    Returns
    -------
    np.ndarray, shape (n_positions, n_traverses)
        ``chance`` in column 0, ``0`` elsewhere; far row pinned to 0 when ``dead_end_last``.
    """
    step = np.zeros((n_positions, n_traverses))
    if n_traverses:
        step[:, 0] = chance
    if dead_end_last and n_positions:
        step[-1] = 0.0
    return step


def random_walk_complete(A, start, seed=0):
    """Random walk that ends once every directed edge of ``A`` has been covered.

    Adapted from the original Endotaxis implementation by Markus Meister
    (``RandomWalkComplete``, https://github.com/markusmeister/Endotaxis-2023); modified for
    this repo to draw from a local ``numpy.random.Generator`` (``default_rng``) instead of the
    global ``np.random.seed``. This is the pre-reward EXPLORATION whose length (it must cover
    every edge) guarantees one traverse wires the full endotaxis map. VERIFICATION-ONLY: it
    validates :func:`analytic_endotaxis_step`; the figure uses the analytic form.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=float)
    neighbors = [np.nonzero(a == 1)[0] for a in A.T]
    remaining = A.copy()
    path = [int(start)]
    n_edges, covered = int(remaining.sum()), 0
    while covered < n_edges:
        i = path[-1]
        j = int(rng.choice(neighbors[i]))
        if remaining[j, i] == 1:
            remaining[j, i] = 0
            covered += 1
        path.append(j)
    return path


def endo_gradient_walk(A, signal, start, goal, max_steps=500):
    """Noiseless gradient navigation on a learned goal signal (VERIFICATION-ONLY).

    Steps to the neighbour with the highest goal signal (any positive gradient is used -- no
    noise), never immediately reversing unless the node is a dead end. This is the
    deterministic, no-noise analogue of Meister's ``Steps`` readout for a graph with cycles
    (a pure argmax can 2-cycle at a plateau; the no-U-turn rule resolves it). Returns the
    visited corridor sequence; used to validate :func:`analytic_endotaxis_step`.
    """
    A = np.asarray(A, dtype=float)
    neighbors = [np.nonzero(a == 1)[0] for a in A.T]
    p, prev, path = int(start), -1, [int(start)]
    for _ in range(max_steps):
        if p == goal:
            break
        nb = neighbors[p]
        cand = nb[nb != prev] if (len(nb) > 1 and prev in nb) else nb
        prev, p = p, int(cand[np.argmax(signal[cand])])
        path.append(p)
    return np.asarray(path)


def draw_walk(axes, Ps, Ss, start_time=0, end_time=None, cmap="plasma", start_corr=None, end_corr=None):
    """
    plot_yl: plot the marker points
    """

    if end_time is None:
        end_time = len(Ps)
    Ps1=Ps[start_time:end_time]
    Ss1=Ss[start_time:end_time]
    if start_corr is None:
        start_corr=Ps1[0]
    if end_corr is None:
        end_corr=Ps1[-1]
    # set nan value to be black. .copy() matters: plt.get_cmap returns the *registered*
    # colormap object when handed a Colormap (rather than a name), so set_bad would mutate
    # the global instance for the rest of the process -- every later panel drawn with that
    # colormap would inherit a black bad-value. Copying keeps the change local.
    new_cmap = plt.get_cmap(cmap).copy()
    new_cmap.set_bad(color="black")
    Ss1 = np.array(Ss1)
    masked_signal = np.ma.masked_where(np.abs(Ss1-np.log(0.1))<1e-5, Ss1)

    axes[0].imshow(masked_signal, interpolation='nearest',aspect='auto', cmap=new_cmap, )
    axes[0].scatter(Ps1, np.arange(len(Ps1)), color="white", s=1)
    # color the "Bouts" strip (axes[-1]) by bout type; the caller draws any step-index
    # markers on the walk itself (axes[0]).
    # find when it hits the start and goal corridor to color sortie and traverses
    h_indices = np.where(Ps1==start_corr)[0]
    o_indices = np.where(Ps1==end_corr)[0]
    plot_utils.color_bouts_from_indices(axes[-1], h_indices, o_indices)

    for ax in axes:
        ax.set_ylim(top=0, bottom=len(Ps1)-1)
    axes[1].yaxis.set_visible(False)
    axes[1].set_xticks([])
    axes[1].set_xlabel("Bouts")

    axes[0].set_xlabel("Corridor")
    axes[0].set_ylabel("Steps")


def format_d_corridor_order_ticks(ax,):
    ax.set_xticks([0, 8, 17])
    ax.set_xticklabels(["Home", "Bottleneck",
                        "Out"])

def expand_signal(S):
    return np.array([S[-1]]+list(S)+[S[0]])

def draw_goal(ax,i, Ss, plot_shortest=False, shortest_indices=None, cmap="plasma", markersize=8):
    n=len(Ss[i])
    ax.plot(range(n),Ss[i], c="black", linewidth=1, zorder=0)
    ax.scatter(range(n), Ss[i], c=Ss[i], cmap=cmap, s=markersize, zorder=5, )
    ax.set_xlabel("Corridor")
    ax.set_ylabel("Goal (log)")
    ax.set_xlim(-0.5,n-0.5)
    ax.set_ylim(bottom=Ss[i].min(), top=Ss[i].max())
    if plot_shortest:
        assert shortest_indices is not None, "Must input shortest path indices"
        ax.scatter(shortest_indices, Ss[i, shortest_indices], edgecolor="red", facecolor="None", zorder=10, linewidth=0.5, label="Shortest path", s=markersize)
