"""Graph algorithms on the maze: Floyd-Warshall, shortest paths, and Markov-walk models.

Split out of utils.py; see docs.
"""
import numpy as np
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path
from manhattan_maze.utils import map_array_based_on_ref

__all__ = ['floyd_warshall', 'zero_order_average_steps', 'first_order_transition_matrix', 'first_order_average_steps', 'find_shortest_path', 'find_sub_sequences', 'get_average_step_matrix', 'get_from_to_node_steps', 'zero_order_P_occupied', 'zero_order_simulation', 'first_order_simulation', 'first_order_P_occupied', 'convert_markov_steps']

def floyd_warshall(adj_mat):
    """
    All-pairs shortest-path distances via the Floyd–Warshall algorithm.

    Parameters
    ----------
    adj_mat : np.ndarray, shape (n, n)
        Directed adjacency matrix. ``adj_mat[i, j] = 1`` encodes a directed
        edge from node j to node i (column → row convention).

    Returns
    -------
    np.ndarray, shape (n, n)
        Shortest-path distance matrix.  ``result[i, j]`` is the minimum
        number of hops from node j to node i.  Unreachable pairs retain
        distance n (the initialisation ceiling).

    Notes
    -----
    Uses the closed-form weight initialisation ``(1 − I)·n − A·(n − 1)``
    so that connected pairs start at 1 and unconnected pairs at n.
    Pinning this formula in tests guards against a scipy-based reimplementation
    that handles directed graphs differently.
    """
    n = len(adj_mat)
    shortest_path = (1 - np.eye(n)) * n - adj_mat * (n - 1)
    for node_idx in range(n):
        shortest_path = np.minimum(
            shortest_path,
            np.repeat(shortest_path[:, node_idx][..., None], n, 1)
            + np.repeat(shortest_path[node_idx, :][None, ...], n, 0),
        )
    return shortest_path


def zero_order_average_steps(A):
    """
    Mean first-passage time T[e, s] for a zero-order Markov random walk.

    Computes the expected number of steps for a walker starting at node s to
    first reach node e, assuming uniform random choice among neighbors
    (no memory of the previous step).

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix of the graph.  Must have no all-zero rows or columns
        (i.e., no isolated nodes).

    Returns
    -------
    np.ndarray, shape (n, n)
        T[e, s] = expected steps from s to e.  Diagonal entries are 0.

    Notes
    -----
    Written by MM. Uses matrix inversion of the sub-transition matrix after
    removing the absorbing end node.
    """
    n=A.shape[0]
    S=A/np.sum(A,axis=0) # transition matrix
    T=[]
    for e in range(n): # index of end node
        R=np.delete(np.delete(S,e,axis=0),e,axis=1) # remove row and column e from the transition matrix
        U=np.linalg.inv(np.identity(n-1)-R.T).dot(np.ones(n-1)) # solve
        U=np.insert(U,e,0) # add the missing row back to the solution
        T+=[U] # add this as a row of T
    return np.array(T) # row index is for end nodes


def first_order_transition_matrix(A, probability=1):
    """
    Build the 2-state (non-reversing) transition matrix for a first-order walk.

    Constructs an extended state space where each state is a directed edge (j→i)
    rather than a node, encoding the constraint that the walker avoids reversing
    unless forced (a dead end).

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix of the graph.  Must have no isolated nodes.
    probability : float, default 1
        Probability of continuing forward (not reversing) at each step.
        Must be > 0.

    Returns
    -------
    B : np.ndarray, shape (fm, fm)
        Column-stochastic transition matrix for the 2-state graph,
        where fm is the number of directed edges.
    node : np.ndarray, shape (n, n)
        node[i, j] = index of the 2-state node representing the step j→i.
    prec : list of list of int
        prec[i] = list of nodes that have edges into node i.
    succ : list of list of int
        succ[j] = list of nodes reachable from node j.
    pair : list of tuple[int, int]
        pair[r] = (i, j) — the directed edge j→i encoded by 2-state node r.

    Raises
    ------
    ValueError
        If ``probability`` is 0.
    """
    n = A.shape[0]
    prec = [[j for j in range(n) if A[i, j]] for i in range(n)]  # list of precursors for each state
    succ = [[i for i in range(n) if A[i, j]] for j in range(n)]  # list of successors for each state
    pair = []  # pair[r]=(i,j) says that 2-state node r is the step from j to i
    node = np.zeros((n, n), dtype=int)  # node[i,j] is the index of the node corresponding to step (i,j)
    fm = 0  # number of nodes in the 2-state graph (forward)
    if probability == 0:  # if probability is 0, then this does not work
        raise ValueError('probability must be greater than 0')
    for j in range(n):  # compile the list of nodes of the 2-state graph
        for i in succ[j]:
            pair += [(i, j)]
            node[i, j] = fm
            fm += 1
    B = np.zeros((fm, fm))  # make transition matrix for 2-state graph
    for q in range(fm):  # for every node in 2-state graph jik
        i, j = pair[q]  # step from j to i corresponding to that node
        if succ[i] == [j]:  # i is a dead end, have to step back to j
            r = node[j, i]  # node corresponding to return to j
            B[r, q] = 1  # add transition
        else:
            for succ_node in succ[i]:  # if one can go from i to succ_node
                f = node[succ_node, i]  # node corresponding to step from i to succ_node
                if succ_node != j:  # not returning to j
                    B[f, q] = probability  # forward transition (j → i → succ_node)
                if succ_node == j:  # returning to j
                    B[f, q] = 1 - probability  # backward
    B /= np.sum(B, axis=0)  # normalize by number of possible steps
    return B, node, prec, succ, pair # return the transition matrix, node mapping, precursors, successors and pairs of steps


def first_order_average_steps(A, probability=1):
    """
    Mean first-passage time T[e, s] for a first-order (non-reversing) Markov walk.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix.  Must have no isolated nodes.
    probability : float, default 1
        Forward probability passed to :func:`first_order_transition_matrix`.

    Returns
    -------
    np.ndarray, shape (n, n)
        T[e, s] = expected steps from node s to node e under first-order dynamics.
        Diagonal entries are 0.

    Notes
    -----
    Written by MM. Operates on the 2-state extended graph from
    :func:`first_order_transition_matrix`.
    """
    B, node, prec, _, _ = first_order_transition_matrix(A, probability)  # get the transition matrix for the 2-state graph
    n = A.shape[0]  # number of nodes in the original graph
    T=np.zeros((n,n))
    for e in range(n): # for every end state
        enodes=[node[e,l] for l in prec[e]] # find all the 2-state nodes that are steps into the end state
        R=np.delete(np.delete(B,enodes,axis=0),enodes,axis=1) # remove their rows and columns from the transition matrix
        l=R.shape[0] # number of remaining nodes
        U=np.linalg.inv(np.identity(l)-R.T).dot(np.ones(l)) # solve
        for x in enodes:  # add the missing rows back to the solution
            U=np.insert(U,x,0)
        for s in range(n): # for every start state
            snodes=[node[s,l] for l in prec[s]] # find all the 2-state nodes that are steps into the start state
            T[e,s]=np.mean(U[snodes]) # average the capture time across all those nodes

    return np.array(T) # row index is for end nodes


def find_shortest_path(a_mat, start, end):
    """
    Find the shortest path (minimum hops) between two nodes in an unweighted graph.

    Parameters
    ----------
    a_mat : np.ndarray, shape (n, n)
        Adjacency matrix; 0 = no edge, 1 = edge (treated as undirected).
    start : int
        Index of the start node.
    end : int
        Index of the end node.

    Returns
    -------
    list of int
        Ordered node indices from ``start`` to ``end``, inclusive.
        Returns an empty list if no path exists.
    """
    dist_matrix, predecessors = shortest_path(a_mat, directed=False, unweighted=True, return_predecessors=True)

    # Reconstruct path
    path = []
    at = end
    while at != start:
        if at == -9999:
            return []  # No path
        path.append(at)
        at = predecessors[start, at]
    path.append(start)
    return path[::-1]


def find_sub_sequences(seq, s, e):
    """
    Find all contiguous sub-sequences from node s to node e in a sequence.

    Each sub-sequence starts at an occurrence of ``s`` and ends at the next
    subsequent occurrence of ``e``.

    Parameters
    ----------
    seq : list
        The full node sequence to search.
    s : any
        Start node identifier.
    e : any
        End node identifier.

    Returns
    -------
    list of np.ndarray
        List of sub-arrays, each from an occurrence of ``s`` to the next ``e``
        (inclusive).  Returns an empty list if ``s`` or ``e`` is absent.
    """
    sub_sequences = []
    if s not in seq:
        print(f"s {s} not in sequence")
        return sub_sequences
    if e not in seq:
        print(f"e {e} not in sequence")
        return sub_sequences

    seq = np.array(seq)
    lengths = []
    i = 0
    while i < len(seq):
        if seq[i] == s:
            start = i
            # Find the next B after the first A
            for j in range(i + 1, len(seq)):
                if seq[j] == e:
                    sub_sequences.append(seq[start:j+1])
                    i = j + 1  # move past the B
                    break
            else:
                break  # no B found
        else:
            i += 1
    return sub_sequences


def get_average_step_matrix(seq, node_list=None):
    """
    Compute pairwise mean step counts between all pairs of nodes from observed sequence.

    Parameters
    ----------
    seq : list
        Observed node sequence.
    node_list : list or None, default None
        Nodes to include.  Defaults to all unique nodes in ``seq``.

    Returns
    -------
    np.ndarray, shape (n_nodes, n_nodes)
        average_steps[e, s] = mean number of steps from node s to node e,
        averaged over all sub-sequences in ``seq``.  NaN where no sub-sequence
        was found.  Diagonal is 0.
    """
    if node_list is None:
        node_list = np.unique(seq)

    average_steps = np.full((len(node_list), len(node_list)), np.nan)
    for start_idx, s in enumerate(node_list):
        for end_idx, e in enumerate(node_list):
            if start_idx == end_idx:
                average_steps[end_idx, start_idx] = 0
                continue
            sub_sequences = find_sub_sequences(seq, s, e)
            if not sub_sequences:
                continue
            average_steps[end_idx, start_idx] = np.mean([len(ss) for ss in sub_sequences])
    return average_steps


def get_from_to_node_steps(T_average, start=0):
    """
    Extract from-start and to-start step columns from a step matrix.

    Parameters
    ----------
    T_average : np.ndarray, shape (n, n)
        Average step matrix where T[e, s] is mean steps from s to e.
    start : int, default 0
        Index of the reference node (e.g., home or out port).

    Returns
    -------
    from_vals : np.ndarray, shape (n-1,)
        Steps from ``start`` to all other nodes (row ``start`` of T, diagonal removed).
    to_vals : np.ndarray, shape (n-1,)
        Steps from all other nodes to ``start`` (column ``start`` of T, diagonal removed).
    """
    from_vals = T_average[:, start]
    from_vals = np.delete(from_vals, start)  # remove self
    to_vals = T_average[start, :]
    to_vals = np.delete(to_vals, start)  # remove self

    return from_vals, to_vals


def zero_order_P_occupied(A, s, e, d):
    """
    Compute node occupancy over time for a zero-order Markov walk.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix.
    s : int
        Start node index; all probability mass initialises here.
    e : int
        End (absorbing) node index; probability mass absorbed at each step.
    d : int
        Maximum number of steps to simulate.

    Returns
    -------
    np.ndarray, shape (d+1, n)
        P[t, i] = fraction of probability at node i at time t.
        P[:, e] gives the cumulative absorption probability over time.
    """
    n=A.shape[0]
    S=A/np.sum(A,axis=0) # transition matrix (zero order so no memory)
    P=[]
    Q=np.zeros(n);Q[s]=1 # initial condition
    P+=[Q.copy()]
    for _ in range(d):
        Q[e]=0
        Q=S.dot(Q)
        P+=[Q.copy()]
    return np.array(P)


def zero_order_simulation(A, s, e, d, n_agents=1, random_seed=0, order_indices=None):
    """
    Simulate zero-order Markov random walks on a graph.

    At each step each agent moves uniformly at random to a neighboring node.
    Walks stop when the end node ``e`` is reached or after ``d`` steps.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix; 1 = edge.
    s : int
        Start node index.
    e : int or None
        End node index; walk stops on arrival.  Pass None for unlimited walks.
    d : int
        Maximum number of steps per agent.
    n_agents : int, default 1
        Number of independent walks to simulate.
    random_seed : int, default 0
        Seed for :func:`numpy.random.seed`.
    order_indices : array-like or None, default None
        If provided, node indices in each path are remapped via
        :func:`map_array_based_on_ref` using this reference.

    Returns
    -------
    np.ndarray, shape (n_agents, d+1)
        Node index at each step.  Entries after the walk ends are -1.
    """
    A = np.array(A)
    num_nodes = A.shape[0]
    assert 0 <= s < num_nodes, "s index out of bounds"
    if e is not None:
        assert 0 <= e < num_nodes, "e index out of bounds"

    paths_array = -np.ones((n_agents, d + 1), dtype=int)  # +1 to include s
    np.random.seed(random_seed) # Set random seed for reproducibility
    for agent in tqdm(range(n_agents), desc="Simulating zero-order agents"):
        current_node = s
        paths_array[agent, 0] = current_node

        for step in range(1, d + 1):
            if current_node == e:
                break
            neighbors = np.where(A[current_node] > 0)[0]
            if len(neighbors) == 0:
                break
            current_node = np.random.choice(neighbors)
            paths_array[agent, step] = current_node

    # convert the corridors based on the indices
    if order_indices is not None:
        for i in range(n_agents):
            full_path = paths_array[i, :]  # get the path for the agent
            full_path = full_path[full_path > -1]  # remove -1s

            paths_array[i, :len(full_path)] = map_array_based_on_ref(ref=order_indices, original=full_path)

    return paths_array


def first_order_simulation(A, s, e, d, n_agents=1, random_seed=0, order_indices=None):
    """
    Simulate first-order (non-reversing) Markov random walks on a graph.

    Agents avoid reversing direction unless at a dead end, using the 2-state
    transition matrix from :func:`first_order_transition_matrix`.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix; 1 = edge.
    s : int
        Start node index.
    e : int or None
        End node index; walk stops on arrival.  Pass None for unlimited walks.
    d : int
        Maximum number of steps per agent.
    n_agents : int, default 1
        Number of independent walks to simulate.
    random_seed : int, default 0
        Seed for :func:`numpy.random.seed`.
    order_indices : array-like or None, default None
        If provided, node indices in each path are remapped via
        :func:`map_array_based_on_ref`.

    Returns
    -------
    np.ndarray, shape (n_agents, d+1)
        Node index at each step.  Entries after the walk ends are -1.
    """
    A = np.array(A)
    num_nodes = A.shape[0]
    assert 0 <= s < num_nodes, "s index out of bounds"
    if e is not None:
        assert 0 <= e < num_nodes, "e index out of bounds"
    # construct the first-order transition matrix
    B, node, prec, succ, pair = first_order_transition_matrix(A)
    paths_array = -np.ones((n_agents, d + 1), dtype=int)  # +1 to include s
    np.random.seed(random_seed)  # Set random seed for reproducibility
    for agent in tqdm(range(n_agents), desc="Simulating first-order agents"):
        current_node = s
        paths_array[agent, 0] = current_node
        previous_node = s
        for step in range(1, d + 1):
            if current_node == e:
                break
            neighbors = succ[current_node]
            if len(neighbors) == 0:
                break
            # Get the indices of the neighbors in the 2-state graph
            neighbor_indices = [node[neighbor, current_node] for neighbor in neighbors]
            if step == 1: # first step, no previous node
                probabilities = np.full((len(neighbors)), 1/len(neighbors))
            else:
                # Get the probabilities of transitioning to each neighbor
                probabilities = B[neighbor_indices, node[current_node, previous_node]]
            next_node = np.random.choice(neighbors, p=probabilities)
            previous_node = current_node # update previous node
            current_node = next_node
            paths_array[agent, step] = current_node

    # convert the corridors based on the indices
    if order_indices is not None:
        for i in range(n_agents):
            full_path = paths_array[i, :]  # get the path for the agent
            full_path = full_path[full_path > -1]  # remove -1s
            paths_array[i, :len(full_path)] = map_array_based_on_ref(ref=order_indices, original=full_path)

    return paths_array


def first_order_P_occupied(A, s, e, d, probability=1):
    """
    Compute node occupancy over time for a first-order (non-reversing) Markov walk.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix.
    s : int
        Start node index.
    e : int
        End (absorbing) node index.
    d : int
        Maximum number of steps to simulate.
    probability : float, default 1
        Forward probability passed to :func:`first_order_transition_matrix`.

    Returns
    -------
    np.ndarray, shape (d+1, n)
        P[t, i] = summed probability at node i at time t under first-order dynamics.
    """

    # check probability
    B, node, _, _, pair = first_order_transition_matrix(A, probability=probability)
    fm = B.shape[0]  # number of nodes in the 2-state graph
    n = A.shape[0]  # number of nodes in the original graph

    # calculate P for each step
    P=[]
    Q=np.zeros(fm) # find the initial condition
    for q in range(fm):
        if pair[q][1] == s: # if node pair start with start node:
            Q[q] = 1 # transitioning out from this
    P+=[Q.copy()]
    for _ in range(d): # because of two states, this needs to be calculated d+1 times
        for q in range(fm):
            if pair[q][1] == e: #if node pair start with end node, do not transition out
                Q[q] = 0  # absorbing
        Q=B.dot(Q) # apply transition matrix
        P+=[Q.copy()]
    first_order_P = np.array(P) # now convert back to the original nodes
    P_orig = np.zeros((d+1, n)) # keep it the same shape to include the first step
    for time_step in range(d):
        for q in range(fm):
            i, j = pair[q]
            P_orig[time_step+1, i] += first_order_P[time_step, q]  # accumulate probability
    return P_orig


def convert_markov_steps(path_array, order_by_appearance=True, order_indices=None):
    """
    Reorder path node indices by first-appearance order in each path.

    Converts absolute node indices in simulation output to a canonical ordering
    based on when each node is first visited (or based on a provided reference).

    Parameters
    ----------
    path_array : np.ndarray, shape (n_agents, d+1)
        Raw simulation paths from :func:`zero_order_simulation` or
        :func:`first_order_simulation`.  Padding value -1 is stripped.
    order_by_appearance : bool, default True
        If True, derive ``order_indices`` from the first-appearance order within
        each path.  If False, ``order_indices`` must be provided.
    order_indices : array-like or None, default None
        Reference ordering used when ``order_by_appearance`` is False.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Remapped path(s).  Returns a single array if ``path_array`` has one
        agent; otherwise returns a list of arrays.
    """
    assert order_by_appearance or order_indices is not None, "If order_by_appearance is True, order_indices must be provided."

    new_paths = []
    for full_path in path_array:
        full_path = full_path[full_path>-1]
        if order_by_appearance:
            # order the indices by their appearance in the full path
            unique_indices = np.sort(np.unique(full_path, return_index=True)[1])  # get the unique corridor indices, remove the first corridor
            order_indices = [full_path[idx] for idx in unique_indices]  # save the corridor index in the first column
        new_paths.append(map_array_based_on_ref(order_indices, full_path))  # map the full path to the order indices

    if len(new_paths) == 1:
        return new_paths[0]
    else:
        return new_paths
