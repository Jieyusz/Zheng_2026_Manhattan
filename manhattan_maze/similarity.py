"""Path/transition similarity metrics (Jaccard, adjusted Jaccard, cosine).

Split out of utils.py; see docs.
"""
import numpy as np

__all__ = ['transition_vec_similarity', 'cosine_similarity', 'jaccard_similarity', 'self_similarity_matrix', 'get_average_traverse_similarity', 'get_mat_mean_diagonal', 'select_similarity_example', 'SIMILARITY_EXAMPLE_MIN_SIDE']

# Smallest matrix side an example triplet must have to be worth drawing. Most animals
# contribute a 1x1 matrix (a single traverse pair), which renders as one meaningless cell.
SIMILARITY_EXAMPLE_MIN_SIDE = 10

def transition_vec_similarity(bmatrix_1, bmatrix_2, n_guaranteed_transitions):
    """
    Adjusted Jaccard similarity between two corridor-transition matrices (Eq. 5).

    Subtracts ``n_guaranteed_transitions`` from both intersection and union to
    remove topologically mandatory corridor transitions that appear in every
    traverse regardless of path choice.

    Parameters
    ----------
    bmatrix_1 : np.ndarray
        Boolean 22×22 corridor transition matrix for traverse 1.
    bmatrix_2 : np.ndarray
        Boolean 22×22 corridor transition matrix for traverse 2.
    n_guaranteed_transitions : int
        Number of mandatory transitions to subtract from both intersection
        and union.  Pass ``mask.n_guaranteed_transitions_for_adjusted_jaccard``.
        Use 0 for standard Jaccard (no correction); use 3 for Mask D.
        This argument is required — no default is provided because the correct
        value is mask-specific (R15).

    Returns
    -------
    float
        Adjusted Jaccard similarity in [0, 1], or 0.0 if adjusted union is 0.

    Raises
    ------
    ValueError
        If the adjusted union is negative, indicating
        ``n_guaranteed_transitions`` exceeds the actual union size.

    Notes
    -----
    The correction of 3 is Mask-D-specific: three corridor transitions are
    topologically mandatory in every Mask D traverse (left-subgraph exit,
    bottleneck crossing, right-subgraph entry).  Applying this correction to
    other masks produces incorrect similarity values.
    Mask attribute: ``MaskDSpecial.n_guaranteed_transitions_for_adjusted_jaccard = 3``;
    ``Mask.n_guaranteed_transitions_for_adjusted_jaccard = 0``.
    """
    bvec_1 = bmatrix_1.flatten()
    bvec_2 = bmatrix_2.flatten()
    intersection = np.logical_and(bvec_1, bvec_2).sum() - n_guaranteed_transitions
    union = np.logical_or(bvec_1, bvec_2).sum() - n_guaranteed_transitions
    if union < 0:
        raise ValueError(
            f"Adjusted union is negative (n_guaranteed_transitions={n_guaranteed_transitions}). "
            "Check that matrices are from valid traverses."
        )
    if union == 0:
        return 0
    return intersection / union


def cosine_similarity(bmatrix_1, bmatrix_2):
    """
    Compute cosine similarity between two binary corridor-transition matrices.

    Parameters
    ----------
    bmatrix_1 : np.ndarray
        Boolean or binary matrix (e.g., 22×22 corridor transition matrix).
    bmatrix_2 : np.ndarray
        Boolean or binary matrix of the same shape as ``bmatrix_1``.

    Returns
    -------
    float
        Cosine similarity in [0, 1] for non-negative binary inputs.
    """
    return np.dot(bmatrix_1.flatten(), bmatrix_2.flatten())/(np.linalg.norm(bmatrix_1)*np.linalg.norm(bmatrix_2))


def jaccard_similarity(bmatrix_1, bmatrix_2):
    """
    Compute standard (unadjusted) Jaccard similarity between two binary matrices.

    Parameters
    ----------
    bmatrix_1 : np.ndarray
        Boolean or binary matrix.
    bmatrix_2 : np.ndarray
        Boolean or binary matrix of the same shape as ``bmatrix_1``.

    Returns
    -------
    float
        |intersection| / |union|; returns 0.0 when both matrices are all-zero.
    """
    intersection = np.logical_and(bmatrix_1, bmatrix_2).sum()
    union = np.logical_or(bmatrix_1, bmatrix_2).sum()
    if union == 0:
        return 0
    else:
        return intersection / union


def self_similarity_matrix(bouts, similarity_function=transition_vec_similarity, **similarity_kwargs):
    """
    Compute the pairwise self-similarity matrix for a list of bouts.

    Parameters
    ----------
    bouts : list of Bout
        Bouts to compare.  Must each implement ``get_corridor_transition_matrix``.
    similarity_function : callable, default transition_vec_similarity
        Function with signature ``f(mat1, mat2, **kwargs) → float``.
    **similarity_kwargs
        Keyword arguments forwarded to ``similarity_function``.  When using
        ``transition_vec_similarity``, pass
        ``n_guaranteed_transitions=mask.n_guaranteed_transitions_for_adjusted_jaccard``.

    Returns
    -------
    np.ndarray, shape (n_bouts, n_bouts) or None
        Symmetric similarity matrix; ``None`` if ``bouts`` is empty.
    """
    if len(bouts) == 0:
        return None
    mat = np.full((len(bouts), len(bouts)), np.nan)
    for i, bout_1 in enumerate(bouts):
        for j, bout_2 in enumerate(bouts):
            mat1 = bout_1.get_corridor_transition_matrix(normalize=False)
            mat2 = bout_2.get_corridor_transition_matrix(normalize=False)
            mat[i, j] = similarity_function(mat1, mat2, **similarity_kwargs)

    return mat


def get_average_traverse_similarity(j_oo, j_hh, j_oh_prime):
    """
    Mean of each manuscript similarity matrix $J_{O,O}$, $J_{H,H}$, $J_{O,H'}$.

    Parameters
    ----------
    j_oo : np.ndarray or None
        $J_{O,O}$ — pairwise self-similarity matrix among outbound (H→O) traverses.
    j_hh : np.ndarray or None
        $J_{H,H}$ — pairwise self-similarity matrix among homebound (O→H) traverses.
    j_oh_prime : np.ndarray or None
        $J_{O,H'}$ — cross-similarity matrix between outbound and reversed-homebound
        traverses.

    Returns
    -------
    j_oo_mean : float
        Mean similarity across the lower triangle (k=-1) of ``j_oo``;
        np.nan if ``j_oo`` is None.
    j_hh_mean : float
        Mean similarity across the lower triangle (k=-1) of ``j_hh``;
        np.nan if ``j_hh`` is None.
    j_oh_prime_mean : float
        Mean similarity across the lower triangle including diagonal (k=0)
        of ``j_oh_prime``; np.nan if ``j_oh_prime`` is None.
    """
    # mean below-diagonal similarity for the self matrices; whole lower triangle
    # (incl. diagonal) for the cross matrix.
    j_oo_mean = np.mean(j_oo[np.tril_indices_from(j_oo, k=-1)]) if j_oo is not None else np.nan
    j_hh_mean = np.mean(j_hh[np.tril_indices_from(j_hh, k=-1)]) if j_hh is not None else np.nan
    j_oh_prime_mean = np.mean(j_oh_prime[np.tril_indices_from(j_oh_prime, k=0)]) if j_oh_prime is not None else np.nan #include diagonal
    # one by one to avoid non-homogenous shapes
    return j_oo_mean, j_hh_mean, j_oh_prime_mean


def get_mat_mean_diagonal(mat):
    """
    Compute the mean value of each off-diagonal band of a matrix.

    For a square matrix of size n, returns n-1 values: the mean of all elements
    at Manhattan distance 0, 1, …, n-2 from the main diagonal.

    Parameters
    ----------
    mat : np.ndarray, shape (m, n)
        Input matrix, typically a similarity matrix.  If m > n, the matrix is
        trimmed to (n, n) before processing.

    Returns
    -------
    np.ndarray, shape (n-1,)
        Mean value at each diagonal offset.  Returns ``[np.nan]`` if ``mat``
        is not 2-D or has more columns than rows.

    Notes
    -----
    Diagonal offset 0 corresponds to the main diagonal (self-similarity = 1
    for normalized similarity matrices).
    """
    # if the matrix is not 2 D, return nan
    if len(mat.shape) != 2:
        print("Matrix is not 2D")
        return [np.nan]
    if mat.shape[0] < mat.shape[1]:
        print("Matrix has more columns than rows: more homebound traverses than outbound?")
        return np.full(mat.shape[1] - 1, np.nan)
    if mat.shape[0] > mat.shape[1]:
        # trim rows to make it square:
        mat = mat[:mat.shape[1], :]
    mat_size = mat.shape[1] # there should be few
    off_diagonal = np.full(mat_size - 1, np.nan)
    for diagonal_offset in np.arange(mat_size - 2):
        diag_mask = np.abs(np.arange(mat_size)[:, None] - np.arange(mat_size)) == diagonal_offset
        values = mat[diag_mask]
        off_diagonal[diagonal_offset] = np.nanmean(values)
    return off_diagonal


def _triplet_min_side(triplet):
    """Smallest side across all three matrices of a similarity triplet; 0 if degenerate."""
    sides = []
    for m in triplet:
        if m is None:
            return 0
        arr = np.asarray(m)
        if arr.ndim < 2:      # a 0-d/1-d entry is not a similarity matrix
            return 0
        sides.append(min(arr.shape))
    return min(sides)


def select_similarity_example(similarity_list, min_side=SIMILARITY_EXAMPLE_MIN_SIDE):
    """
    Pick the example animal for a Mask-D route-similarity triplet by content, not position.

    Returns the ``(j_oo, j_hh, j_oh_prime)`` entry whose *smallest* matrix side is largest --
    the animal with the most traverses, and so the most informative example.

    Why not an index
    ----------------
    These lists are built in ``gen_ac_generalization.py`` / ``gen_wildtype_d_data.py``, whose
    cohort order comes from ``set(...)``.  Python salts string hashing per process, so the
    order is different on every regeneration run, and a positional index into the result is
    only valid until the next regen -- which is exactly how the old
    ``ACORTICAL_D_SIMILARITY_EXAMPLE_ID`` went stale and started selecting a 1x1
    (single-traverse-pair) entry.  Scoring on content is immune to the reshuffle *and* keeps
    choosing the same mouse for as long as the underlying data is unchanged.

    The list is also shorter than the session list (zero-traverse sessions are skipped at
    generation time), so its indices never lined up with anything else either.

    Parameters
    ----------
    similarity_list : sequence of (j_oo, j_hh, j_oh_prime)
        As saved under ``"<genotype> D similarity matrices"``.  Entries may legitimately be
        ``None`` or 1x1, and the three matrices of one entry may differ in shape.
    min_side : int, optional
        Reject entries whose smallest side is below this.

    Returns
    -------
    tuple of np.ndarray
        The chosen ``(j_oo, j_hh, j_oh_prime)``.

    Raises
    ------
    ValueError
        If no entry clears ``min_side``.
    """
    sides = [_triplet_min_side(t) for t in similarity_list]
    best = int(np.argmax(sides)) if sides else None
    if best is None or sides[best] < min_side:
        raise ValueError(
            f"No similarity triplet reaches min_side={min_side}: {len(similarity_list)} "
            f"entries with smallest sides {sides}. Either the cohort changed or the data "
            f"was regenerated with too few traverses per animal.")
    return similarity_list[best]
