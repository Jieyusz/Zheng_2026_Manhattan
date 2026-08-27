"""Behavioural metrics, turn-correctness, animal/session selection, and figure-data wrangling.

Split out of utils.py; see docs.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from manhattan_maze.utils import df_condense_consecutive_repeats, extract_array
from manhattan_maze.random_walk import reversal_decisions, forward_bias_mle

__all__ = ['select_sessions_by_traverse', 'create_t1_df', 'select_t1_sessions', 'create_mask_learn_seq_df', 'get_animal_df', 'create_ob_df',
           'check_overlap_percentage', 'get_top_patterns', 'get_turn_at_loc', 'get_first_at_loc', 'get_hole_correctness', 'count_error',
           'calculate_seq_error', 'calculate_seq_error_rate', 'get_first_learning_session', 'get_animal_repeated_mask_df', 'get_animal_repeated_mask_sessions',
           'get_session_df_by_day_range', 'get_session_df_by_gap_size', 'get_traverse_data_df', 'get_session_list_from_df',
           'get_memory_metric_dict_of_mask', 'data_list_to_curve_fit_df', 'get_two_day_data_df', 'extract_x_y_from_data_df',
           'get_wildtype_d_sessions', 'get_d_transition_matrices', 'select_d_transition_dict', 'count_rewards_over_day',
           'extract_timepoint_from_array_dict', 'get_mask_learning_count_df', 'get_animal_learning_masks_df', 'select_biclique_offpath_transitions',
           'renormalize_choice_among_arms', 'journey_distance_seq', 'localize_distance_seq', 'observed_n_pos', 'start_distance',
           'position_error_matrix', 'cohort_position_error_rate', 'hole_error_rate_by_direction',
           'sorties_per_journey_by_direction', 'first_journey_corridor_seq',
           'first_journey_forward_bias_curve', 'first_traverse_forward_bias']

def select_sessions_by_traverse(sessions, n_traverses=0):
    """
    Filter sessions by minimum traverse count and return qualifying animal names.

    Parameters
    ----------
    sessions : list of Session
        Session objects to evaluate.  Must be non-empty.
    n_traverses : int, default 0
        Minimum number of traverses required for a session to be selected.

    Returns
    -------
    list of str
        Animal names (``session.name``) for sessions with at least
        ``n_traverses`` traverses.

    Raises
    ------
    ValueError
        If ``sessions`` is empty.
    """
    if len(sessions)==0:
        raise ValueError("sessions is empty")

    nicknames = []
    for session in tqdm(sessions): # counting the numbers may take sometime
        n_rewards =  len(session.filter("traverse")) # count the number of traverses in the session Mask A
        if n_rewards >= n_traverses:
            nicknames.append(session.name)
    return nicknames


def create_t1_df(mdf):
    """
    Build a dataframe of first-day (t1) experiment sessions from metadata.

    Filters ``mdf`` to rows whose ``Nickname`` contains ``"t1"``, then parses
    the ``Config_label_list`` column into a wide-format dataframe where each
    column is a session slot and each row is one t1 animal.

    Parameters
    ----------
    mdf : pd.DataFrame
        Master metadata dataframe with columns ``Nickname`` (str) and
        ``Config_label_list`` (str; bracket-delimited, comma-separated mask labels).

    Returns
    -------
    pd.DataFrame
        Wide-format dataframe with one column per session slot (0, 1, …) and
        an additional ``Nickname`` column.  Cell values are mask label strings
        (e.g., ``'A'``, ``'B'``).
    """
    t1_mdf = mdf[mdf.Nickname.str.contains("t1")]
    t1_df = pd.DataFrame([m.strip('][').split(', ') for m in t1_mdf.Config_label_list.tolist()])
    t1_df["Nickname"] = t1_mdf.Nickname.tolist()
    return t1_df


def select_t1_sessions(data, t1_df, session_idx=0, mask_name="A"):
    """
    Select session objects for a given mask from the t1 session dataframe.

    Parameters
    ----------
    data : dict-like
        Data object indexed by nickname, then by session index.
    t1_df : pd.DataFrame
        Output of :func:`create_t1_df`; must have a ``Nickname`` column and
        integer session-slot columns.
    session_idx : int, default 0
        Session slot index to retrieve from each matching animal.
    mask_name : str, default 'A'
        Mask label to filter on (e.g., ``'A'``, ``'B'``).

    Returns
    -------
    list of Session
        Session objects for all animals whose session slot ``session_idx``
        equals ``mask_name``.
    """
    # get the list of nicknames
    nicknames = t1_df[t1_df[session_idx]==mask_name].Nickname.tolist()
    # select the sessions for the nicknames
    sessions = [data[nickname][session_idx] for nickname in nicknames]
    return sessions


def create_mask_learn_seq_df(mdf, animals, mask_list, generalization_only=True):
    """
    Build a dataframe of mask-learning order per animal.

    For each animal, extracts the first session for each mask in ``mask_list``
    and assigns a ``Mask_order`` index based on chronological order
    (Day then Session_idx).  Animals that encountered only one mask in the list
    are excluded when ``generalization_only=True``.

    Parameters
    ----------
    mdf : pd.DataFrame
        Master metadata dataframe.
    animals : list of str
        Animal identifiers (substrings of ``Nickname``).
    mask_list : list of str
        Mask labels to include (e.g., ``['A', 'B', 'C']``).
    generalization_only : bool, default True
        If True, skip animals that appear in fewer than two masks from
        ``mask_list``.

    Returns
    -------
    pd.DataFrame
        Concatenated per-animal dataframes with columns from
        :func:`get_animal_df` plus ``Mask_order`` (int, 0-based).
    """
    all_df_list = []
    for animal in animals:
        animal_df = get_animal_df(mdf, animal)
        # only show the first session for each mask
        mask_first_df = animal_df.groupby("Mask").first()
        # only keep the ones in A, B, C
        mask_first_df = mask_first_df[mask_first_df.index.isin(mask_list)]
        # turn this index back into Mask and reset index
        mask_first_df.reset_index(inplace=True, drop=False)
        if len(mask_first_df) == 1 and generalization_only:
            continue
        # name this first index as mask order
        mask_first_df = mask_first_df.sort_values(by=["Day", "Session_idx"],
                                                          ascending=[True, True]).reset_index(drop=True)
        mask_first_df["Mask_order"] = mask_first_df.index
        all_df_list.append(mask_first_df)
    # concatenate all the dataframes
    mask_gen_df = pd.concat(all_df_list, ignore_index=True)
    return mask_gen_df


def get_animal_df(mdf, animal_name):
    """
    Return a per-animal session dataframe sorted chronologically.

    Filters ``mdf`` to rows matching ``animal_name``, then expands each
    recording's ``Config_label_list`` into one row per session slot.

    Parameters
    ----------
    mdf : pd.DataFrame
        Master metadata dataframe with columns ``Nickname``, ``Age`` (int),
        and ``Config_label_list`` (str).
    animal_name : str
        Substring matched against ``Nickname`` to select the animal.

    Returns
    -------
    pd.DataFrame
        Columns: ``Nickname``, ``Day`` (int, days since first recording),
        ``Session_idx`` (int, slot within recording), ``Mask`` (str),
        ``Animal`` (str).  Sorted by ``Day`` then ``Session_idx``.

    Raises
    ------
    ValueError
        If no rows match ``animal_name``.
    """
    animal_df = mdf[mdf.Nickname.str.contains(animal_name)].reset_index(drop=True)
    # sort by age
    animal_df = animal_df.sort_values(by=["Age"]).reset_index(drop=True)
    if animal_df.empty:
        raise ValueError(f"Animal name {animal_name} not found in metadata")
    # create a session idx for each config
    animal_session_list = []
    day_count = animal_df["Age"].tolist()[0]
    for _, row in animal_df.iterrows():
        nickname = row.Nickname
        age = row.Age
        config_labels = row.Config_label_list.strip('][').split(', ')
        for session_config_idx, config_label in enumerate(config_labels):
            animal_session_list.append((nickname, age-day_count, session_config_idx, config_label))

    animal_session_df = pd.DataFrame(animal_session_list, columns=["Nickname", "Day", "Session_idx", "Mask"])
    animal_session_df["Animal"] = animal_name
    # sort the session df first by day and then by session_idx (from early to late)
    animal_session_df = animal_session_df.sort_values(by=["Day", "Session_idx"], ascending=[True, True]).reset_index(drop=True)

    return animal_session_df


def create_ob_df(mdf):
    """
    Build a dataframe for olfactory-bulb-ablated animals.

    Selects animals whose ``Nickname`` contains 'B' or 'C' (ablation cohorts),
    then computes days since first exposure and an experiment count per animal.

    Parameters
    ----------
    mdf : pd.DataFrame
        Master metadata dataframe with columns ``Nickname`` (str), ``Age`` (int),
        and ``Condition`` (str).

    Returns
    -------
    pd.DataFrame
        Columns: ``Nickname``, ``Animal`` (first 2 chars of Nickname),
        ``Age``, ``Condition``, ``Day_since_first_exp`` (int, 1-based),
        ``Experiment_count`` (int, last character of Nickname parsed as int).
    """
    # albation experiments (animal with B and C)
    nicknames_ablated = mdf[mdf["Nickname"].str.contains("B") | mdf["Nickname"].str.contains("C")].Nickname.tolist()
    ob_df = pd.DataFrame({"Nickname": nicknames_ablated, "Animal": [nickname[:2] for nickname in nicknames_ablated],
                          "Age": [mdf[mdf.Nickname == nickname].Age.values[0] for nickname in nicknames_ablated],
                          "Condition": [mdf[mdf.Nickname == nickname].Condition.values[0] for nickname in nicknames_ablated],})
    # calculate the day of exposure, grouped by animal
    ob_df["Day_since_first_exp"] = ob_df.groupby("Animal").Age.transform(lambda x: x - np.min(x) + 1)  # add 1 for first day, since first experiment
    ob_df["Experiment_count"] = ob_df["Nickname"].transform(lambda x:int(x[-1])) # get the experiment number
    return ob_df


def check_overlap_percentage(list_1, list_2):
    """
    Compute the fraction of unique elements in list_2 that also appear in list_1.

    Parameters
    ----------
    list_1 : list
        Reference collection (e.g., previously visited corridors).
    list_2 : list
        Query collection whose coverage is measured.

    Returns
    -------
    float
        |unique(list_2) ∩ unique(list_1)| / |unique(list_2)|;
        returns 0.0 when ``list_2`` is empty.
    """
    pre_set = set(list_1)
    post_set = set(list_2)
    overlap = len(pre_set.intersection(post_set))
    return overlap / len(post_set) if len(post_set) > 0 else 0


def get_top_patterns(sequence, pattern_length=3, top_n=10):
    """
    Find the top-N most frequent n-grams (sub-sequences) in a sequence.

    Parameters
    ----------
    sequence : list
        Ordered list of elements (e.g., turns, locations, corridor indices).
    pattern_length : int, default 3
        Length of each n-gram.
    top_n : int, default 10
        Maximum number of most-common patterns to return.

    Returns
    -------
    list of tuple[tuple, int]
        Up to ``top_n`` entries of the form (pattern_tuple, count), sorted
        from most to least frequent.  Returns an empty list if
        ``pattern_length > len(sequence)``.
    """
    if pattern_length > len(sequence):
        return []

    # Generate all n-grams
    patterns = [tuple(sequence[i:i+pattern_length])
                for i in range(len(sequence) - pattern_length + 1)]

    # Count and return top N
    counter = Counter(patterns)
    return counter.most_common(top_n)


def get_turn_at_loc(seq):
    """
    Aggregate all turn decisions at each location into a dictionary.

    Parameters
    ----------
    seq : list of tuple[any, any]
        Sequence of (location, decision) pairs.

    Returns
    -------
    dict of {any: list}
        Keys are unique locations; values are lists of all decisions recorded
        at that location, in order.
    """
    # get the unique keys of holes in the sequence
    unique_locs = list(set([loc for loc, dec in seq]))  # get unique locations
    turns_at_locs = {loc: [] for loc in unique_locs}  # initialize the dictionary with empty lists
    for loc, dec in seq:
        turns_at_locs[loc].append(dec)  # append the turn to the list
    return turns_at_locs


def get_first_at_loc(turns_at_locs):
    """
    Keep only the first turn decision at each location.

    Parameters
    ----------
    turns_at_locs : dict of {any: list}
        Output of :func:`get_turn_at_loc`; keys are locations, values are
        lists of decisions.

    Returns
    -------
    dict of {any: list}
        Same keys; each value is a length-1 list containing only the first
        decision, or ``[np.nan]`` if the decision list is empty.
    """
    first_at_locs = {loc: [turns[0]] if turns else [np.nan] for loc, turns in turns_at_locs.items()}
    return first_at_locs


def get_hole_correctness(seq, correct_dict, include="all"):
    """
    Compute per-hole turn correctness for a corridor sequence.

    Parameters
    ----------
    seq : list
        Corridor sequence (output of get_allocentric_turns or similar).
    correct_dict : dict
        Mapping of hole (x, y) → correct allocentric direction ('N','S','E','W').
        Use Mask.get_correct_turns() to obtain this dict.
    include : {'all', 'first'}
        'all': average correctness over all (given) visits to each hole.
        'first': use only the first visit to each hole.
        Note: this aggregates whatever crossings it is handed. The
        approach-conditioned turn-error metric filters to correct-corridor
        crossings *before* calling this (see ``Bout.count_error``); passing the
        raw crossing sequence reproduces the deprecated, inflated measure.

    Returns
    -------
    np.ndarray, shape (n_holes,)
        Mean correctness per hole in [0, 1]; np.nan if a hole was not visited.
    """
    turns_at_locs = get_turn_at_loc(seq)
    if include == "first":
        turns_at_locs = get_first_at_loc(turns_at_locs)

    correct_vec = np.full(len(correct_dict.keys()), np.nan)
    for hole_idx, (hole, dec) in enumerate(correct_dict.items()):
        if hole not in turns_at_locs:
            correctness = [np.nan]
        else:
            decisions = turns_at_locs[hole]
            correctness = [1 if dec == correct_dict[hole] else 0 for dec in decisions]
        correct_vec[hole_idx] = np.mean(correctness)
    return correct_vec


def count_error(correctness_vec, error_type="rate"):
    """
    Compute turn error count or rate from a correctness vector.

    Parameters
    ----------
    correctness_vec : array-like
        Per-hole (or per-crossing) correctness values in {0, 1, np.nan}. NaN
        marks an unscored hole (e.g. never visited on the approach corridor) and
        is dropped from both numerator and denominator, so the canonical
        first-decision-per-hole vector (which is NaN for unvisited holes) reduces
        correctly. A plain 0/1 sequence contains no NaN and is unaffected.
    error_type : {'rate', 'count'}
        'rate': errors / n_scored (turn error rate, as in Eq. 2), where n_scored
        counts only non-NaN entries.
        'count': raw number of errors over scored entries.

    Returns
    -------
    float
        Error rate in [0, 1] or raw error count; np.nan if no entry is scored.
    """
    vec = np.asarray(correctness_vec, dtype=float)
    n_scored = int(np.count_nonzero(~np.isnan(vec)))  # scored holes/crossings only
    errors = n_scored - np.nansum(vec)
    if error_type == "rate":
        return errors / n_scored if n_scored > 0 else np.nan
    elif error_type == "count":
        return errors
    else:
        raise ValueError(f"Unknown error_type {error_type}, must be 'rate' or 'count'")


def calculate_seq_error(distance_seq):
    """
    Count forward steps (distance-increasing moves) in a distance-to-goal sequence.

    Parameters
    ----------
    distance_seq : array-like
        Sequence of maze distances (in tile steps) to the goal at each position.

    Returns
    -------
    int
        Number of steps where the distance to goal increases relative to the
        previous position (i.e., the animal moved away from the goal).
    """
    step_seq = np.diff(distance_seq, append=distance_seq[-1])
    error_seq = [1 if x > 0 else 0 for x in
                 step_seq]  # the errors are when the distance to goal increases from the previous tile
    return np.sum(error_seq)


def calculate_seq_error_rate(distance_seq):
    """
    Per-step non-progress rate for a distance-to-goal sequence.

    Fraction of steps that fail to decrease the distance to the goal, i.e. the
    per-step version of :func:`calculate_seq_error`. Uses the non-decreasing
    (``>= 0``) rule of :func:`localize_distance_seq` — a step counts as an error
    when the distance to goal does not strictly decrease, which credits the
    zero-distance start/port re-entry as non-progress. Pooled over a whole
    traverse this is identical to the ``error_propagation`` corridor error rate
    (``counts / opps`` summed over positions), so corridor/tile error share one
    definition with the supplementary figure.

    Parameters
    ----------
    distance_seq : array-like
        Sequence of maze distances (in graph steps) to the goal at each position.

    Returns
    -------
    float
        Error steps / total steps, in ``[0, 1]``. ``np.nan`` for a sequence with
        fewer than two nodes (no step to score). Chance is ~0.5 (on the linear
        track a memoryless walker departs toward/away 50/50 at interior
        positions; on high-degree graphs 0.5 is an approximate reference).
    """
    seq = np.asarray(distance_seq, dtype=float)
    n_steps = len(seq) - 1
    if n_steps <= 0:
        return np.nan
    return float(np.count_nonzero(np.diff(seq) >= 0)) / n_steps


def get_first_learning_session(data, sub_mdf, mask_name, strict_first=True, e_trained="All"):
    """
    Select first-exposure sessions for a mask with optional prior-training filter.

    Parameters
    ----------
    data : dict-like
        Data object indexed by nickname and session index.
    sub_mdf : pd.DataFrame
        Metadata dataframe subset with an ``Animal`` column.
    mask_name : str
        Mask label to look for (must be in ``data.mask_names``).
    strict_first : bool, default True
        If True, only include animals for whom ``mask_name`` is their first
        mask exposure (excluding 'O' and 'E' from the prior list).
    e_trained : {'All', 'Yes', 'No'}, default 'All'
        Filter on Mask E prior training:
        'All' = include all animals; 'Yes' = must have seen E first;
        'No' = must not have seen E before.

    Returns
    -------
    list of Session
        First-exposure session objects for qualifying animals.

    Raises
    ------
    AssertionError
        If ``mask_name`` is not in ``data.mask_names`` or ``e_trained`` is invalid.
    ValueError
        If ``e_trained='Yes'`` and ``mask_name='E'`` (contradictory).
    """
    assert mask_name in data.mask_names, f"Mask {mask_name} not in masks {data.mask_names}"
    assert e_trained in ["Yes", "No", "All"], f"e_trained must be 'Yes', 'No' or 'All', but got {e_trained}"

    if e_trained=="Yes" and mask_name == "E":
        raise ValueError("E trained animals cannot have E as their first learning session.")

    animals = sub_mdf["Animal"].unique().tolist()
    selected_sessions = []
    for a in animals:
        sub_df = get_animal_df(sub_mdf, a)
        # get a list of all configurations:
        mask_seq = sub_df.Mask.to_list()
        if mask_name not in mask_seq:
            continue
        # find the index and check if there is any config before that
        mask_index = mask_seq.index(mask_name)
        if e_trained == "All":
            prior_list = [m for m in mask_seq[:mask_index] if m !="O" and m !="E"] # include all cases
        else:
            prior_list = [m for m in mask_seq[:mask_index] if m !="O"] # check E

        if e_trained == "No" and "E" in mask_seq[:mask_index]:
            continue # not e trained
        elif e_trained == "Yes" and "E" not in mask_seq[:mask_index]:
            continue # e trained but no e before
        if strict_first and prior_list:
            continue
        nickname, session_idx = sub_df["Nickname"].iloc[mask_index], sub_df["Session_idx"].iloc[mask_index]
        selected_sessions.append(data[nickname][session_idx])
    return selected_sessions


def get_animal_repeated_mask_df(sub_df, animal, mask_name, first_in_day=True, self_day_reference=False):
    """
    Get a per-animal mask-session dataframe for repeated exposures.

    Parameters
    ----------
    sub_df : pd.DataFrame
        Master metadata dataframe subset.
    animal : str
        Animal identifier.
    mask_name : str
        Mask label to filter on.
    first_in_day : bool, default True
        If True, keep only the first session per day (de-duplicate consecutive
        repeats by Day).
    self_day_reference : bool, default False
        If True, shift Day values so that the first day of this mask = 0.

    Returns
    -------
    pd.DataFrame
        Subset of the animal's session dataframe filtered to ``mask_name``,
        with optional day de-duplication and re-referencing.
    """
    animal_df = get_animal_df(sub_df, animal)
    mask_df = animal_df[animal_df.Mask == mask_name]
    if first_in_day:
        mask_df, _ = df_condense_consecutive_repeats(mask_df, column_name="Day")

    if self_day_reference:
        mask_df["Day"] = mask_df["Day"] - mask_df["Day"].min() # make the first day of this mask as day 0
    return mask_df


def get_animal_repeated_mask_sessions(data, sub_mdf, animal, mask_name, **kwargs):
    """
    Return session objects and day indices for repeated mask exposures.

    Parameters
    ----------
    data : dict-like
        Data object indexed by nickname and session index.
    sub_mdf : pd.DataFrame
        Master metadata dataframe subset.
    animal : str
        Animal identifier.
    mask_name : str
        Mask label (must be in ``data.mask_names``).
    **kwargs
        Forwarded to :func:`get_animal_repeated_mask_df`
        (e.g., ``first_in_day``, ``self_day_reference``).

    Returns
    -------
    sessions : list of Session
        Session objects in chronological order.
    days : np.ndarray
        Unique day values corresponding to the selected sessions.
    """
    assert mask_name in data.mask_names, f"Mask {mask_name} not in masks {data.mask_names}"
    mask_df = get_animal_repeated_mask_df(sub_mdf, animal, mask_name, **kwargs)
    sessions = get_session_list_from_df(data, mask_df)
    days = mask_df.Day.unique()
    return sessions, days


def get_session_df_by_day_range(sub_mdf, mask_name, day_range=(0, 1), first_only=True,
                                self_day_reference=True):
    """
    Retrieve sessions within a relative day range for a given mask.

    Parameters
    ----------
    sub_mdf : pd.DataFrame
        Master metadata dataframe subset with an ``Animal`` column.
    mask_name : str
        Mask label to filter on.
    day_range : tuple[int, int], default (0, 1)
        (start, stop) in relative days (start inclusive, stop exclusive).
        Day 0 is the first day the animal was exposed to ``mask_name`` when
        ``self_day_reference=True``.
    first_only : bool, default True
        If True, keep only one session per animal (first occurrence).
    self_day_reference : bool, default True
        If True, rebase Day values relative to each animal's first session
        with ``mask_name``.

    Returns
    -------
    pd.DataFrame
        Concatenated session rows matching the day range.  Index is reset.
    """
    animals = sub_mdf["Animal"].unique().tolist()
    session_df_list = []
    for animal in animals:
        animal_df = get_animal_df(sub_mdf, animal).reset_index()
        animal_df = animal_df[animal_df.Mask == mask_name]
        if self_day_reference: # substract the first day of this mask for each animal, so that the day range is relative to the first day of this mask
            animal_df["Day"] = animal_df["Day"] - animal_df["Day"].iloc[0]
        for day in range(day_range[0], day_range[1]):
            day_df = animal_df[animal_df.Day == day]
            session_df_list.append(day_df)

    session_df = pd.concat(session_df_list, ignore_index=True)
    if first_only: # only one session per animal
        session_df, _ = df_condense_consecutive_repeats(session_df, column_name="Animal")
    return session_df


def get_session_df_by_gap_size(sub_mdf, mask_name, gap_size=(1, 2), self_day_reference=True):
    """
    Identify session pairs that bracket inter-session gaps of a given size.

    Parameters
    ----------
    sub_mdf : pd.DataFrame
        Master metadata dataframe subset with an ``Animal`` column.
    mask_name : str
        Mask label to filter on.
    gap_size : tuple[int, int], default (1, 2)
        (min_gap, max_gap) in days (min inclusive, max exclusive).
        Gaps are computed on the first-session-per-day timeline.
    self_day_reference : bool, default True
        If True, rebase Day values relative to the first session with
        ``mask_name``.

    Returns
    -------
    prev_session_df : pd.DataFrame
        Session rows immediately before each qualifying gap.
    post_session_df : pd.DataFrame
        Session rows immediately after each qualifying gap.
    """
    ## check if gap size is valid
    assert gap_size[0]<gap_size[1], f"gap ranges must be valid, but got {gap_size}"
    assert gap_size[0]>=0, f"gap ranges must be positive, but got {gap_size}"

    animals = sub_mdf["Animal"].unique().tolist()
    prev_session_df_list = []
    post_session_df_list = []
    for animal in animals:
        repeated_mask_df = get_animal_repeated_mask_df(sub_df=sub_mdf, animal=animal, mask_name=mask_name, self_day_reference=self_day_reference,
                                                       first_in_day=True)
        # find the sessions with gaps within the range of gap size range:
        repeated_mask_df["Day_gap"] = repeated_mask_df["Day"].diff().fillna(0) # calculate the gap size between sessions, fill the first one with 1
        # change type to integers
        repeated_mask_df.Day_gap = repeated_mask_df.Day_gap.astype(int)
        for day in range(gap_size[0], gap_size[1]):
            post_session_indices = repeated_mask_df[repeated_mask_df.Day_gap == day].index
            prev_session_indices = post_session_indices-1
            prev_session_df_list.append(repeated_mask_df.loc[prev_session_indices])
            post_session_df_list.append(repeated_mask_df.loc[post_session_indices])
    prev_session_df = pd.concat(prev_session_df_list, ignore_index=True)
    post_session_df = pd.concat(post_session_df_list, ignore_index=True)
    return prev_session_df, post_session_df


def get_traverse_data_df(sessions, data_type):
    """
    Extract per-traverse metric values from a list of sessions into a tidy dataframe.

    Parameters
    ----------
    sessions : list of Session
        Sessions to process.
    data_type : str
        Metric name (e.g., ``'duration'``, ``'turn error rate'``); passed to
        ``session.filter("traverse").get_bout_stats(unit=data_type)``.

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with columns ``b`` (int, 1-based traverse index),
        ``Value`` (float), and ``Animal`` (str).
    """
    data_df_list = []
    for s in sessions:
        animal = s.name.split("_")[0]
        sub_df = data_list_to_curve_fit_df(s.filter("traverse").get_bout_stats(unit=data_type), animal=animal)
        data_df_list.append(sub_df)
    data_df = pd.concat(data_df_list, ignore_index=True)
    return data_df


def get_session_list_from_df(data, session_df):
    """
    Retrieve session objects from a dataframe of (Nickname, Session_idx) pairs.

    Parameters
    ----------
    data : dict-like
        Data object indexed first by ``Nickname``, then by ``Session_idx``.
    session_df : pd.DataFrame
        Must contain columns ``Nickname`` (str) and ``Session_idx`` (int).

    Returns
    -------
    list of Session
        Session objects in the order of ``session_df`` rows.
    """
    session_list = []
    for _, row in session_df.iterrows():
        nickname, session_idx = row["Nickname"], row["Session_idx"]
        session_list.append(data[nickname][session_idx])
    return session_list


def get_memory_metric_dict_of_mask(data, sub_mdf, mask_name, day_ranges, size=10,
                                   metric_list=None, traverse=True, first_only=True, slice=False):
    """
    Extract metric values across day-ranges for a mask into a nested dictionary.

    Parameters
    ----------
    data : dict-like
        Data object indexed by nickname and session index.
    sub_mdf : pd.DataFrame
        Master metadata dataframe subset.
    mask_name : str
        Mask label to analyse.
    day_ranges : list of tuple[int, int]
        List of (start, stop) day ranges passed to
        :func:`get_session_df_by_day_range`.
    size : int, default 10
        Number of traverses per session to retain (passed to
        :func:`extract_array`).
    metric_list : list of str or None, default None
        Metrics to extract.  Defaults to ``['duration', 'turn error rate']``.
    traverse : bool, default True
        If True, filter sessions to traverses only before extracting metrics.
    first_only : bool, default True
        If True, keep only one session per animal per day range.
    slice : bool, default False
        If True, use ``get_slice_stats`` instead of ``get_bout_stats``.

    Returns
    -------
    dict of {str: list of tuple[tuple, np.ndarray]}
        Keys are metric names; values are lists of (day_range, array) tuples
        where array has shape (n_animals, size).
    """
    if metric_list is None:
        metric_list = ["duration", "turn error rate"]
    # create a nested list to store each metric
    metric_values_dict = {metric: [] for metric in metric_list}
    for day_range in day_ranges:
        session_df = get_session_df_by_day_range(sub_mdf, mask_name, day_range, first_only=first_only)
        sessions = get_session_list_from_df(data, session_df)
        if traverse:
            sessions = [s.filter("traverse") for s in sessions]
        for metric in metric_list:
            if slice:
                data_list = [s.get_slice_stats(metric) for s in sessions]
            else:
                data_list = [s.get_bout_stats(metric) for s in sessions]
            array = extract_array(data_list, size=size)
            metric_values_dict[metric].append((day_range, array))
    return metric_values_dict


def data_list_to_curve_fit_df(data_list, animal=None):
    """
    Convert a per-traverse value list to a curve-fit dataframe with 1-based index.

    Parameters
    ----------
    data_list : list of float
        Per-traverse metric values for a single session.
    animal : str or None, default None
        Animal identifier added as the ``Animal`` column when provided.

    Returns
    -------
    pd.DataFrame
        Columns: ``b`` (int, traverse index 1-based), ``Value`` (float),
        and optionally ``Animal`` (str).
    """
    data_df_list = []
    for bout_idx, val in enumerate(data_list):
        data_df_list.append({"b":bout_idx+1, "Value":val})

    data_df = pd.DataFrame(data_df_list)
    if animal is not None and not data_df.empty:
        data_df["Animal"] = animal
    return data_df


def get_two_day_data_df(t1_df, data, mdf, data_type="duration", traverse=True, day1_only=False, **kwargs):
    """
    Build a two-day learning dataframe for curve fitting.

    Concatenates Day 1 (Mask A, session index 1) and Day 2 bout-level metric
    values for each animal in ``t1_df``.  Adds categorical dummy variables for
    mask, day, session group, and outbound/homebound classification.

    Parameters
    ----------
    t1_df : pd.DataFrame
        Output of :func:`create_t1_df`; one row per t1 animal.
    data : dict-like
        Data object indexed by nickname and session index.
    mdf : pd.DataFrame
        Master metadata dataframe for sex and age lookup.
    data_type : str, default 'duration'
        Metric to extract per bout.
    traverse : bool, default True
        If True, filter sessions to traverses before extracting metrics.
    day1_only : bool, default False
        If True, include only Day 1 data (skip Day 2 sessions).
    **kwargs
        Forwarded to ``session.get_bout_stats``.

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with columns: ``Animal``, ``Session``, ``Mask``,
        ``Sex``, ``Age``, ``b`` (int, 1-based), ``Value``, ``Day`` (1 or 2),
        ``sg1``, ``sg2`` (session group dummies), ``outbound`` (0/1),
        ``animal_idx``, ``mask_idx``, ``maskb``, ``maskc``.
    """
    data_df_list = []

    for i, row in t1_df.iterrows():
        nn = row["Nickname"]
        animal = nn.split("_")[0]
        day1_session = data[f"{animal}_a1"][1]
        sex = mdf[mdf.Nickname==nn].Sex.values[0]
        age = mdf[mdf.Nickname==f"{animal}_a1"].Age.values[0]
        if traverse:
            day1_se = day1_session.filter("traverse")
        else:
            day1_se = day1_session
        day1_values = day1_se.get_bout_stats(data_type)
        for bout_idx in range(len(day1_values)):
            data_df_list.append({"Animal": animal, "Session": 1, "Mask": "A", "Sex":sex, "Age":age,
                                 "b":bout_idx+1, "Value": day1_values[bout_idx]})
        day2_traj = data[nn]
        day2_age = mdf[mdf.Nickname==nn].Age.values[0]
        if day1_only:
            continue
        for s_idx, s in enumerate(day2_traj):
            if traverse:
                s = s.filter("traverse")
            values = s.get_bout_stats(data_type, **kwargs)
            mask_name = row[s_idx]
            for bout_idx in range(len(values)):
                data_df_list.append({"Animal": animal, "Session": s_idx+2, "Mask": mask_name, "Sex":sex, "Age":day2_age,
                                     "Value": values[bout_idx], "b":bout_idx+1})

    data_df = pd.DataFrame(data_df_list)
    data_df = data_df[
        data_df["Mask"] != "A_flipped"]  # remove the flipped A session since it's not consistent across animals

    # Below is to categorize different sessions
    data_df["Day"] = (data_df['Session'] > 1).astype(int)
    data_df["Day"] = data_df["Day"] + 1  # for Day 1 and Day 2 convention
    # create session groups based on session indice
    data_df["sg1"] = (data_df["Session"]==2).astype(int) # day2-1
    data_df["sg2"] = (data_df["Session"]>2).astype(int) # day2-2 and later
    # separate outbound and homebound b%2
    data_df["outbound"] = (data_df["b"]%2==1).astype(int)

    # create indices for categorical variables (animals and mask)
    data_df["animal_idx"], _ = pd.factorize(data_df["Animal"])
    data_df["mask_idx"], _ = pd.factorize(data_df["Mask"])
    # create dummy varialbe for mask
    data_df["maskb"] = (data_df["Mask"]=="B").astype(int)
    data_df["maskc"] = (data_df["Mask"]=="C").astype(int)
    return data_df


def extract_x_y_from_data_df(data_df, y_name, x_columns=None,):
    """
    Extract design matrix X and response vector y from a dataframe.

    Parameters
    ----------
    data_df : pd.DataFrame
        Tidy dataframe (e.g., from :func:`get_two_day_data_df`).
    y_name : str
        Column name for the dependent variable.
    x_columns : list of str or None, default None
        Column names for the design matrix.  Defaults to
        ``['b', 'Day', 'mask_idx', 'animal_idx']``.

    Returns
    -------
    X : np.ndarray, shape (n_features, n_observations)
        Design matrix (transposed from dataframe convention).
    y : np.ndarray, shape (n_observations,)
        Response vector.
    """
    if x_columns is None:
        x_columns = ["b", "Day", "mask_idx", "animal_idx"]
    x = data_df[x_columns].to_numpy()
    y = data_df[y_name].to_numpy()
    return x.T, y


def get_wildtype_d_sessions(data, mdf, session_idx=1):
    """
    Select the Mask-D sessions for all BL6J "a1" animals (shared by the D producers).

    Parameters
    ----------
    data : dict-like
        Data object indexed by nickname then session index.
    mdf : pd.DataFrame
        Master metadata dataframe.
    session_idx : int, default 1
        Session index to pull per animal (Day-1 Mask D).

    Returns
    -------
    list of Session
        One session per matching BL6J animal, in metadata order.
    """
    nicknames = mdf[(mdf["Config_label_list"].str.contains("D"))
                    & (mdf["Nickname"].str.contains("a1"))
                    & (mdf["Genotype"] == "BL6J")].Nickname.tolist()
    return [data[nickname][session_idx] for nickname in nicknames]


def sorties_per_journey_by_direction(session):
    """Mean sorties per journey for one session, split by starting port.

    A sortie is a reward-free excursion that leaves a port and returns to it, so its
    start and end port coincide: ``H-H`` sorties leave and return to the home port
    (``is_outbound``), ``O-O`` sorties leave and return to the out/cage port
    (``is_homebound``). Counts are normalised by the number of journeys — i.e. the
    number of terminating traverses/rewards. Trailing sorties after the last reward
    form a journey with no terminating traverse and are excluded from the denominator.

    Parameters
    ----------
    session : Session
        A single-animal session.

    Returns
    -------
    (float, float)
        ``(H-H, O-O)`` mean sorties per journey; ``(nan, nan)`` if the session has no
        completed journey (no traverse).
    """
    n_journeys = len(session.filter("traverse"))  # journeys terminate on a traverse/reward
    if n_journeys == 0:
        return np.nan, np.nan
    return (len(session.filter("H-H")) / n_journeys,
            len(session.filter("O-O")) / n_journeys)


def get_d_transition_matrices(sessions, size, corridor_order):
    """
    Per-session, per-slice Mask-D corridor transition matrices in the reduced display order.

    For each session, computes the normalized corridor transition matrix per reward
    slice, then keeps only the corridors in ``corridor_order`` and reorders rows/cols
    to that order — the same outskirt-removed display order the endotaxis schematic
    uses — so transition data and schematic share one corridor index space.  Sessions
    with no traverses are skipped; slices are NaN-padded/truncated to ``size``.

    Parameters
    ----------
    sessions : list of Session
    size : int
        Number of reward slices to retain per session.
    corridor_order : array-like of int
        Display-ordered raw corridor indices (e.g. ``MaskDSpec.plot_corridor_order``);
        defines the ``n = len(corridor_order)`` reduced transition space.

    Returns
    -------
    np.ndarray, shape (n_sessions, size, n, n)
        ``arr[s, slice, end, start]`` = P(start corridor → end corridor) at that slice
        (rows = end, cols = start), NaN where a session has fewer than ``size`` slices.
    """
    corridor_order = np.asarray(corridor_order)
    n = len(corridor_order)
    session_mats = []
    for se in sessions:
        if len(se.filter("traverse")) == 0:  # skip sessions with no traverses
            continue
        trans_mats = se.get_slice_stats("corridor transition matrix", normalize=True)
        reduced = np.full((size, n, n), np.nan)
        for k, t_mat in enumerate(trans_mats[:size]):
            reduced[k] = t_mat[np.ix_(corridor_order, corridor_order)]
        session_mats.append(reduced)
    return np.array(session_mats)


def select_d_transition_dict(matrices, start, ends):
    """
    Build a ``{end: P(start → end)}`` choice dict from :func:`get_d_transition_matrices` output.

    Parameters
    ----------
    matrices : ndarray, shape (n_sessions, size, n, n)
        Output of :func:`get_d_transition_matrices`; ``arr[s, slice, end, start]`` =
        P(start → end).  Indices are display positions in the reduced corridor order.
    start : int
        Display-position index of the start corridor.
    ends : iterable of int
        Display-position indices of the end corridors (goal + controls).

    Returns
    -------
    dict of {int: ndarray (n_sessions, size)}
        One per-session/per-slice transition-probability array per end corridor,
        suitable for ``plot_aggregated_choice_ratios``.
    """
    return {end: matrices[:, :, end, start] for end in ends}


def renormalize_choice_among_arms(transition_dict, group_by=None):
    """
    Renormalize transition-probability arms to sum to 1 within each group.

    Turns raw ``P(start -> end)`` arms into the conditional choice ratio *given* one
    of the grouped arms was taken (chance = ``1/len(group)``). Shared by the
    bottleneck (E/F) and off-path biclique (G/H) panels so both use one convention.

    Parameters
    ----------
    transition_dict : dict of {key: ndarray (n_sessions, size)}
        Per-arm transition probabilities.
    group_by : callable or None
        Maps a dict key to a group id. ``None`` puts every arm in one group (a
        single start node, e.g. the E/F goal+controls). For the biclique dict keyed
        by ``(start, end)`` pass ``group_by=lambda k: k[0]`` to renormalize each
        start node's arms separately.

    Returns
    -------
    dict of {key: ndarray (n_sessions, size)}
        Same keys; each arm divided by its group's per-slice sum (NaN where the group
        has no traffic).
    """
    groups = {}
    for k in transition_dict:
        groups.setdefault(None if group_by is None else group_by(k), []).append(k)
    out = {}
    for ks in groups.values():
        total = np.nansum([transition_dict[k] for k in ks], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in ks:
                out[k] = np.where(total > 0, transition_dict[k] / total, np.nan)
    return out


def select_biclique_offpath_transitions(matrices, biclique_groups,
                                        shortest_path_corridors, corridor_order, normalize=False):
    """
    Extract all off-shortest-path transitions within one biclique, in both directions.

    A biclique has two partite groups of corridors; the corridors lying on the shortest
    path are dropped, leaving the off-path corridors in each group.  Every transition
    between the two off-path sets is collected in both directions — with the standard
    :class:`MaskDSpec` defaults each group has 3 off-path corridors, giving 3 × 3 × 2 = 18
    transitions.  Raw corridor ids are converted to display positions internally (via
    ``corridor_order.index``) and the per-transition probabilities are pulled with
    :func:`select_d_transition_dict`.

    Parameters
    ----------
    matrices : ndarray, shape (n_sessions, size, n, n)
        Output of :func:`get_d_transition_matrices`; ``arr[s, slice, end, start]`` =
        P(start → end), indexed by display position in the reduced corridor order.
    biclique_groups : sequence of two sequences of int
        The two partite groups of **raw** corridor ids for one biclique, e.g.
        ``MaskDSpec().biclique_1_groups`` = ``[[5, 3, 7, 9], [13, 15, 17, 19]]``.
    shortest_path_corridors : iterable of int
        Raw corridor ids on the shortest path
        (``MaskDSpec().shortest_path_corridor_indices``); corridors in this set are excluded.
    corridor_order : sequence of int
        Display-ordered raw corridor ids (``list(MaskDSpec().plot_corridor_order)``);
        maps a raw corridor id to its display position via ``.index``.
    normalize : bool, default False
        If True, return the choice ratio for each off-path transition instead of the raw
        probability: each value is divided by the sum over **all** corridors in the start
        node's opposite group (the 4 possible choices, including the shortest-path arm).
        Slices with no traffic across that group (zero or NaN denominator) yield NaN.

    Returns
    -------
    dict of {(int, int): ndarray (n_sessions, size)}
        Keyed by **raw** ``(start_corridor, end_corridor)`` tuples; both directions are
        distinct keys (e.g. ``(3, 13)`` and ``(13, 3)``).  Only the off-path transitions
        are returned; when ``normalize`` is True the shortest-path arm contributes to the
        denominator but is not a key.
    """
    group_a, group_b = biclique_groups
    sp = set(shortest_path_corridors)
    a_off = [c for c in group_a if c not in sp]
    b_off = [c for c in group_b if c not in sp]

    result = {}
    for srcs, dsts, full_dsts in [(a_off, b_off, group_b), (b_off, a_off, group_a)]:
        end_pos = [corridor_order.index(e) for e in dsts]
        full_pos = [corridor_order.index(e) for e in full_dsts]
        for start in srcs:
            start_pos = corridor_order.index(start)
            choice_dict = select_d_transition_dict(matrices, start_pos, end_pos)
            if normalize:
                total = sum(matrices[:, :, ep, start_pos] for ep in full_pos)
            for end_raw, ep in zip(dsts, end_pos):
                arr = choice_dict[ep]
                if normalize:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        arr = np.where(total > 0, arr / total, np.nan)
                result[(start, end_raw)] = arr
    return result

def count_rewards_over_day(data, sub_mdf, animal_name, mask_name, start_time=0, end_time=60, first_only=False):
    """
    Count rewards received in a time window across mask sessions for one animal.

    Parameters
    ----------
    data : dict-like
        Data object indexed by nickname and session index.
    sub_mdf : pd.DataFrame
        Master metadata dataframe subset.
    animal_name : str
        Animal identifier.
    mask_name : str
        Mask label to filter on.
    start_time : float, default 0
        Window start [minutes] relative to the session start.
    end_time : float, default 60
        Window end [minutes] relative to the session start.
    first_only : bool, default False
        If True, keep only the first session per day.

    Returns
    -------
    rewards_per_session : list of int
        Number of rewards within the window for each session.
    experiment_days : list of int
        Relative day index for each session.
    """
    animal_df = get_animal_df(sub_mdf, animal_name)
    mask_df = animal_df[animal_df.Mask == mask_name]
    if first_only:
        mask_df, _ = df_condense_consecutive_repeats(mask_df, column_name="Day")

    rewards_per_session = []
    experiment_days = []
    for _, row in mask_df.iterrows():
        session = data[row.Nickname][row.Session_idx]
        reward_intervals = session.get_slice_stats("reward intervals") # seconds
        reward_times = np.cumsum(reward_intervals) # get the cumulative time of rewards
        rewards_in_window = (reward_times >= start_time*60) & (reward_times <= end_time*60) # check which rewards are in the time window
        rewards_per_session.append(np.sum(rewards_in_window))
        experiment_days.append(row.Day)
    return rewards_per_session, experiment_days


def extract_timepoint_from_array_dict(data_dict, t):
    """
    Slice the same time index from all arrays in a dictionary.

    Parameters
    ----------
    data_dict : dict of {any: np.ndarray}
        Values are 2-D arrays of shape (n_animals, n_time).
    t : int
        Time index to extract.  Must be < the minimum n_time across all arrays.

    Returns
    -------
    dict of {any: np.ndarray}
        Same keys; each value is a 1-D array of shape (n_animals,) at time t.

    Raises
    ------
    AssertionError
        If ``t`` is out of range for any array in ``data_dict``.
    """
    # check if t is within the range of the array
    array_shape = [value.shape[1] for value in data_dict.values()]
    assert t < min(array_shape) and t >=0, f"t must be smaller than the length of the array shape {array_shape}"
    return {key:value[:, t] for key, value in data_dict.items()}


def get_mask_learning_count_df(data, sub_mdf, mask_names=None, high_reward_number=19.9):
    """
    Count animals that attempted and learned each mask.

    Parameters
    ----------
    data : dict-like
        Data object indexed by nickname and session index.
    sub_mdf : pd.DataFrame
        Master metadata dataframe subset with an ``Animal`` column.
    mask_names : list of str or None, default None
        Masks to evaluate.  Defaults to ``['O', 'E', 'F', 'A', 'B', 'C', 'D']``.
    high_reward_number : float, default 19.9
        Reward count threshold above which an animal is considered to have
        learned a mask (≥ ``high_reward_number`` rewards in at least one session).

    Returns
    -------
    pd.DataFrame
        Columns: ``Mask``, ``Total_sessions``, ``Animal_list``,
        ``Num_animals``, ``Rewarded_animal_list``, ``Num_rewarded_animals``,
        ``Long_term_animal_list``, ``Num_long_term_animals``,
        ``Highly_rewarded_animal_list``, ``Num_highly_rewarded_animals``.
    """
    if mask_names is None:
        mask_names = ["O", "E", "F", "A", "B", "C", "D"]
    animals = sub_mdf["Animal"].unique().tolist()
    session_df = pd.concat([get_animal_df(sub_mdf, animal) for animal in animals],
                                     ignore_index=True)
    # filter out the masks that are not in the mask_names list to speed up
    session_df = session_df[session_df.Mask.isin(mask_names)]
    rwd_counts = []
    for _, row in session_df.iterrows():
        nn, sidx = row["Nickname"], row["Session_idx"]
        session = data[nn][sidx]
        rwd_counts.append(session._n_rewards)
    session_df["Reward_count"] = rwd_counts

    count_list = []
    for mask in mask_names:
        mask_sessions = session_df[session_df['Mask'] == mask]
        mask_animals = mask_sessions['Animal'].unique()
        animal_str = ", ".join(mask_animals)
        # number of rewarded naimals  at least one pair of traverses
        rewarded_animals = mask_sessions[mask_sessions['Reward_count'] > 2]['Animal']
        rewarded_str = ", ".join(rewarded_animals.unique())
        # count the ones with more than 1 session

        # count the ones with more than 10 rewards in at least one session
        highly_rewarded_animals = mask_sessions[mask_sessions['Reward_count'] >= high_reward_number]['Animal']
        highly_rewarded_str = ", ".join(highly_rewarded_animals.unique())
        animal_session_counts = highly_rewarded_animals.value_counts()

        # of these animals that learned, count the ones with long-term memory test
        long_term_animals = animal_session_counts[animal_session_counts > 1].index.tolist()
        long_term_str = ", ".join(long_term_animals)
        count_list.append((mask, len(mask_sessions), animal_str, len(mask_animals),
                           rewarded_str, len(rewarded_animals.unique()),
                           long_term_str, len(long_term_animals),
                           highly_rewarded_str, len(highly_rewarded_animals.unique())))

    count_df = pd.DataFrame(count_list, columns=['Mask', 'Total_sessions', 'Animal_list',
                                                 'Num_animals', 'Rewarded_animal_list', 'Num_rewarded_animals',
                                                 'Long_term_animal_list', 'Num_long_term_animals',
                                                 'Highly_rewarded_animal_list', 'Num_highly_rewarded_animals'])
    return count_df


def get_animal_learning_masks_df(count_df):
    """
    Summarise learned, failed, and unseen masks per animal.

    Parameters
    ----------
    count_df : pd.DataFrame
        Output of :func:`get_mask_learning_count_df`.

    Returns
    -------
    pd.DataFrame
        Columns: ``Animal``, ``Learned_masks`` (comma-separated str),
        ``Failed_masks`` (attempted but not learned), ``Unseen_masks``
        (never attempted).
    """
    animal_mask_summary = []
    animals = count_df["Animal_list"].str.split(", ").explode().unique()
    for animal in animals:
        # get learned masks for this animal:
        learned_masks = count_df[count_df["Highly_rewarded_animal_list"].str.contains(animal)]['Mask'].tolist()
        unlearned_masks = count_df[~count_df["Highly_rewarded_animal_list"].str.contains(animal)]['Mask'].tolist()
        # also check if these masks were ever attempted:
        all_mask_seen = count_df[count_df["Animal_list"].str.contains(animal)]['Mask'].tolist()
        failed_masks = [mask for mask in unlearned_masks if mask in all_mask_seen]
        unseen_masks = [mask for mask in unlearned_masks if mask not in all_mask_seen]
        animal_mask_summary.append((animal, ", ".join(learned_masks),
                                    ", ".join(failed_masks), ", ".join(unseen_masks)))

    animal_mask_df = pd.DataFrame(animal_mask_summary,
                                  columns=['Animal', 'Learned_masks', 'Failed_masks', 'Unseen_masks'])

    return animal_mask_df


# ---------------------------------------------------------------------------
# Error localization along the path graph — position x journey/traverse
# ---------------------------------------------------------------------------
# These score *where along a journey* corridor (or tile) errors happen and how
# that changes over learning, for the path-graph masks (A/B/C), separately per
# traverse direction (outbound "H-O", homebound "O-H"). A corridor error is a step
# that fails to bring the animal closer to the reward corridor; errors are
# referenced to the reward corridor of the whole slice (its terminating-traverse
# destination) via public Session/Bout methods. See scripts/gen_error_propagation.py.

def journey_distance_seq(journey, unit="corridor"):
    """
    Distance-to-reward over a journey's concatenated bouts.

    The reward corridor (tile) is the journey's terminating-traverse destination
    — the last element of the concatenated sequence — and every bout is scored
    against it. Reuses ``Session.concat_corridors_df`` for the sequence and
    ``Bout.get_corridor_distance_seq`` for the per-corridor distance (tile
    analogues for ``unit="tile"``). Port repeats at bout seams are preserved.

    Parameters
    ----------
    journey : Session slice
        A journey (leading sorties + terminating traverse), e.g. from
        ``Session.slice_to_journeys``.
    unit : {"corridor", "tile"}, default "corridor"
        Graph unit whose shortest-path distance is measured.

    Returns
    -------
    np.ndarray
        Distance-to-reward at each step; empty if the journey has no corridors.
    """
    if unit == "corridor":
        seq = journey.concat_corridors_df()["corridor"].to_numpy()
        dseq_method = "get_corridor_distance_seq"
    else:
        seq = journey.concat_tiles_df()["tile"].to_numpy()
        dseq_method = "get_tile_distance_seq"
    if len(seq) == 0:
        return np.array([])
    reward = int(seq[-1])  # goal = journey's final (destination) corridor/tile
    parts = [np.asarray(getattr(b, dseq_method)(reward)) for b in journey.bouts]
    return np.concatenate(parts) if parts else np.array([])


def localize_distance_seq(dist, n_pos):
    """
    Per-position error/opportunity counts for one distance-to-reward sequence.

    A step is an error if the distance-to-reward does **not** decrease
    (``dist[i+1] - dist[i] >= 0``), attributed to the departing position's
    distance. The non-decreasing rule (rather than strictly increasing) credits
    the start corridor: when the animal leaves at the port and re-enters, the
    start corridor repeats (a zero-distance step) and that failure to progress is
    counted. Distances 1..n_pos-2 are identical under either rule.

    Parameters
    ----------
    dist : array-like
        Distance-to-reward at each step of a slice.
    n_pos : int
        Number of distance bins (0..n_pos-1).

    Returns
    -------
    counts, opps : np.ndarray, shape (n_pos,)
        Indexed by the distance-to-reward the step departs from; ``opps[d]``
        counts all departures from distance ``d`` (the rate denominator).
    """
    counts = np.zeros(n_pos)
    opps = np.zeros(n_pos)
    dist = np.asarray(dist, dtype=float)
    for i in range(len(dist) - 1):
        p = int(dist[i])                       # distance-to-reward BEFORE the step
        if 0 <= p < n_pos:
            opps[p] += 1
            if dist[i + 1] - dist[i] >= 0:     # did not get closer => error
                counts[p] += 1
    return counts, opps


def observed_n_pos(sessions, unit="corridor"):
    """
    Number of distance-to-reward bins = max distance actually reached + 1.

    Sized from the data, not the graph diameter: the corridor graph spans the
    whole physical maze but path-graph animals only use the path (Mask A: max
    distance 9 -> n_pos 10). This admits off-path corridors on masks B/C that sit
    farther than the home->out geodesic without empty trailing rows or clipping.
    """
    max_d = 0
    for s in sessions:
        for jr in s.slice_to_journeys():
            if jr.bouts and jr.bouts[-1].satisfy("traverse"):
                d = journey_distance_seq(jr, unit)
                if len(d):
                    max_d = max(max_d, int(np.nanmax(d)))
    return max_d + 1


def start_distance(mask, unit="corridor"):
    """
    Distance-to-reward of the start corridor/tile (the home<->out geodesic).

    Used as the 'far' end for colour normalization so the start corridor maps to
    the far end on every mask; off-path corridors farther than this clamp to it.
    """
    if unit == "corridor":
        return int(mask.corridors_shortest_distance[mask.home_corridor, mask.out_corridor])
    return int(mask.tiles_shortest_distances[mask.home_tile, mask.out_tile])


def position_error_matrix(session, n_pos, count_by="journey", unit="corridor", direction=None):
    """
    Per-session ``(counts, opps)`` of shape ``(n_pos, n_slices)``.

    A column is one journey (``count_by="journey"``: leading sorties + terminating
    traverse, via ``Session.slice_to_journeys``; sorties-only trailing slices are
    dropped) or one completed traverse bout (``count_by="traverse"``). Errors are
    referenced to the reward corridor of that slice. ``direction`` (``"H-O"`` /
    ``"O-H"``) keeps only slices whose terminating traverse matches; ``None`` keeps
    both.
    """
    if count_by == "journey":
        slices = [jr for jr in session.slice_to_journeys()
                  if jr.bouts and jr.bouts[-1].satisfy("traverse")
                  and (direction is None or jr.bouts[-1].bout_type == direction)]
        dseqs = [journey_distance_seq(jr, unit) for jr in slices]
    else:  # traverse — each completed traverse is a single-bout slice (goal = its end)
        method = "get_corridor_distance_seq" if unit == "corridor" else "get_tile_distance_seq"
        bouts = session.filter(direction) if direction else session.filter("traverse")
        dseqs = [np.asarray(getattr(b, method)(-1)) for b in bouts]

    counts = np.zeros((n_pos, len(dseqs)))
    opps = np.zeros((n_pos, len(dseqs)))
    for j, dist in enumerate(dseqs):
        c, o = localize_distance_seq(dist, n_pos)
        counts[:, j] = c
        opps[:, j] = o
    return counts, opps


def _cohort_position_mean(matrices, width):
    """
    Mean of per-session ``(n_pos x n_slices)`` rate matrices across animals.

    Each session's matrix is NaN-padded to ``width`` columns (head-aligned, the
    repo convention via ``extract_array``) then averaged position-by-position with
    ``np.nanmean`` so animals with fewer slices do not bias later columns.
    """
    import warnings
    n_pos = matrices[0].shape[0]
    out = np.full((n_pos, width), np.nan)
    for p in range(n_pos):
        rows = [m[p, :] for m in matrices]
        stacked = extract_array(rows, size=width)  # (n_animals x width)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[p, :] = np.nanmean(stacked, axis=0)
    return out


def cohort_position_error_rate(sessions, n_pos, count_by="traverse", unit="corridor", direction=None):
    """
    Population-mean error-rate matrix ``(n_pos x width)`` and animal count.

    For each session, ``position_error_matrix`` gives per-slice error counts and
    opportunities; the per-column error **rate** (``counts / opps``) is averaged
    across animals with NaN-padding. Row ``d`` is distance-to-reward ``d``; column
    ``j`` is the ``j``-th slice (journey or traverse) of ``count_by``.

    Returns
    -------
    (np.ndarray, int)
        ``(cohort_matrix, n_animals)``; ``cohort_matrix`` is all-NaN width-0 if no
        session has a matching slice.
    """
    per = []
    for s in sessions:
        c, o = position_error_matrix(s, n_pos, count_by, unit, direction)
        if c.shape[1] > 0:
            rate = np.divide(c, o, out=np.full_like(c, np.nan), where=o > 0)
            per.append(rate)
    width = max((m.shape[1] for m in per), default=0)
    cohort = _cohort_position_mean(per, width) if per else np.full((n_pos, 0), np.nan)
    return cohort, len(per)


def hole_error_rate_by_direction(sessions, direction, size, include="first"):
    """
    Per-hole turn-error-rate arrays for one traverse direction, ordered close->far.

    Mirrors the ``"error rate by hole"`` map but selectable scoring: ``include=
    "first"`` is the canonical first-decision-per-hole approach-conditioned rate
    (chance 0.5); ``include="all"`` reproduces the all-visits map. For each animal,
    filters to ``direction`` traverses and stacks
    ``1 - Bout.get_hole_correctness_vec(include, condition="approach")`` into a
    ``(n_holes, n_bouts)`` array; then, per hole, NaN-pads across animals to
    ``size`` columns with ``extract_array``.

    ``Mask.get_correct_turns`` orders holes start -> reward for both directions, so
    the returned list is reversed to run **close -> far from reward** (index 0 =
    reward side), matching the distance-to-reward colour convention used for the
    corridor panels.

    Returns
    -------
    list of np.ndarray
        One ``(n_animals, size)`` array per hole, index 0 = closest to reward.
    """
    per_animal = []
    for s in sessions:
        group = s.filter("traverse").filter(direction)
        vecs = [1 - np.asarray(b.get_hole_correctness_vec(include=include, condition="approach"))
                for b in group]
        if vecs:
            per_animal.append(np.array(vecs).T)  # (n_holes, n_bouts)
    per_animal = [a for a in per_animal if a.shape[1] > 0]
    if not per_animal:
        return []
    n_holes = per_animal[0].shape[0]
    hole_list = []
    for k in range(n_holes):
        rows = [a[k, :] for a in per_animal]
        hole_list.append(extract_array(rows, size=size))  # (n_animals, size)
    return hole_list[::-1]  # reverse: index 0 = reward side, matches corridor panels


def first_journey_corridor_seq(session):
    """
    Run-length-collapsed corridor sequence of a session's first journey.

    The "first journey" is everything the animal did **before its first reward**, plus the
    traverse that earned it: the pre-reward outbound sorties (excursions that leave the home
    port and return to it) followed by the first traverse. This is the window in which any
    structure the animal has picked up must have been learned *latently*, without reward.

    Parameters
    ----------
    session : Session
        Session to read.

    Returns
    -------
    np.ndarray of int
        Corridor indices in order of traversal [corridor index 0-21], with consecutive
        repeats collapsed so that a "reversal" is well defined as
        ``seq[t+1] == seq[t-1]``. Empty if the session contains no traverse.

    Notes
    -----
    ``slice_by_traverse_idx(None, 0)`` is exactly the first journey — every bout up to and
    including traverse 0 — so the sorties need no separate selection; the same
    ``concat_corridors_df`` idiom is used by :func:`journey_distance_seq`. Verified
    byte-identical to the previous hand-rolled bout loop (which filtered
    ``is_sortie() and is_outbound()`` and appended the traverse) on all 79 available
    sessions across Masks A, D and O.

    No longer strictly read-only: ``concat_corridors_df`` tags each visited bout's
    ``corridors_df`` with a ``bout_idx`` column, as it does for every other caller.
    """
    if not sorted(sum(session.get_traverse_indices(), [])):
        return np.array([])

    journey = session.slice_by_traverse_idx(None, 0)
    return _collapse_repeats(journey.concat_corridors_df()["corridor"].to_numpy())


def first_traverse_forward_bias(session):
    """
    Reversal-based forward bias :math:`\\hat{\\beta}` over a session's **first traverse**.

    The readout behind the ``Wildtype mice`` row of manuscript ``tab:walker``, which
    compares the walker and the animals on the first outbound traverse. Scored on exactly
    the bout that produces column 0 of the ``Wildtype {A,D} Corridor error array``, so
    :math:`\\mathcal{E}`, :math:`\\rho` and :math:`\\hat{\\beta}` in that table all describe
    one and the same traverse.

    Parameters
    ----------
    session : Session
        Session to read.

    Returns
    -------
    float
        :math:`\\hat{\\beta}` in [0, 1], or NaN if the session contains no traverse or the
        traverse holds no scorable interior decision.

    Notes
    -----
    Deliberately **not** the same window as :func:`first_journey_corridor_seq`, which spans
    the pre-reward sorties *plus* the first traverse and feeds
    :func:`first_journey_forward_bias_curve`. That journey window carries the latent-learning
    claim; this one carries the ``tab:walker`` comparison, and the two differ substantially
    (Mask A 0.62 vs 0.54; Mask D 0.62 vs 0.44), so they must not be interchanged.

    Degrees come from the corridor adjacency matrix; nodes of degree < 2 are dropped by
    :func:`~manhattan_maze.random_walk.reversal_decisions`, since a reversal at a dead end is
    forced rather than chosen.
    """
    traverses = session.filter("traverse")
    if len(traverses) == 0:
        return np.nan
    seq = _collapse_repeats(traverses[0].corridors_df["corridor"].to_numpy())
    adj = np.asarray(session.mask.corridors_adj_mat)
    degrees = {i: int(adj[i].sum()) for i in range(adj.shape[0])}
    _positions, is_reversal, scored_degrees = reversal_decisions(seq, degrees)
    return forward_bias_mle(is_reversal, scored_degrees)


def _collapse_repeats(node_seq):
    """
    Drop consecutive duplicates from an integer sequence.

    Parameters
    ----------
    node_seq : array_like of int
        Node sequence, possibly with a node repeated on adjacent steps.

    Returns
    -------
    np.ndarray of int
        The sequence with runs collapsed to a single element.
    """
    node_seq = np.asarray(node_seq, int)
    if node_seq.size == 0:
        return node_seq
    return node_seq[np.concatenate([[True], node_seq[1:] != node_seq[:-1]])]


def first_journey_forward_bias_curve(sessions, win=0.20, n_points=18, min_animals=2,
                                     min_decisions=2, min_length=6, mode="valid"):
    """
    Cohort forward-bias curve over the course of the pre-reward first journey.

    For each position on a grid spanning each animal's own first journey
    (:func:`first_journey_corridor_seq`), pools the reversal decisions lying within
    ``win / 2`` of that position and fits :math:`\\hat{\\beta}` once over them with
    :func:`~manhattan_maze.random_walk.forward_bias_mle`, then averages across animals.
    Positions are *fractional*, so animals with journeys of very different lengths are
    comparable. A curve that rises above 0.5 before any reward is the latent-learning
    readout: the animal is navigating with directional persistence that no reward has yet
    taught it.

    The sliding window is the **only** smoothing applied — there is deliberately no
    post-hoc moving average on top of it (see Notes).

    Parameters
    ----------
    sessions : sequence of Session
        Cohort sessions, all on the **same mask** — the corridor-degree map is taken from
        ``sessions[0].mask``.
    win : float, default 0.20
        Window width as a fraction of each animal's journey, i.e. each point pools the
        decisions within ``+/- win / 2`` of it.
    n_points : int, default 18
        Number of evaluation points, on a **closed** grid: ``np.linspace(0, 1, n_points)``,
        so the extreme points sit exactly at the start and end of the journey.
    min_animals : int, default 2
        Points with fewer contributing animals are returned as NaN, since a "cohort mean"
        of one animal has no meaningful SE.
    min_decisions : int, default 2
        An animal contributes to a point only if its window there holds at least this many
        scorable decisions.
    min_length : int, default 6
        Animals whose collapsed first journey is shorter than this are skipped entirely.
    mode : {"valid", "same"}, default "valid"
        Edge handling, with the same meaning as in
        :func:`~manhattan_maze.utils.moving_average`. ``"valid"`` returns only the positions
        whose whole window fits inside the journey (``win / 2 <= x <= 1 - win / 2``), leaving
        the rest NaN; ``"same"`` also evaluates the positions whose window is truncated by an
        end of the journey, so the curve spans a full 0 to 1.

    Returns
    -------
    np.ndarray, shape (3, n_points)
        Row 0 = fractional position along the journey, ``linspace(0, 1, n_points)``; row 1 =
        cohort mean :math:`\\hat{\\beta}`; row 2 = SE across animals. Dimensionless. NaN
        where fewer than ``min_animals`` contributed, and (under ``mode="valid"``) at
        positions whose window would overhang an end of the journey.

    Notes
    -----
    **One smoothing stage.** Each point is a single fit over its own decisions, so the
    bandwidth of the curve is exactly ``win`` and nothing else. Averaging :math:`\\hat{\\beta}`
    *estimates* — rather than pooling their decisions — would additionally bias the result,
    since :math:`\\hat{\\beta} = 1/(1+\\varphi)` is nonlinear in the reversal count.

    **Why ``"valid"`` is the default.** A window overhanging an end of the journey still
    yields an unbiased estimate — :func:`~manhattan_maze.random_walk.forward_bias_mle` sums
    over exactly the decisions supplied, and on a graph whose scored nodes all have degree 2
    it reduces to the exactly unbiased ``1 - R/N`` — but it rests on about half as many
    decisions, so those end points swing enough to read as a spurious hook. Plotting only
    fully-supported positions is the treatment
    :func:`~manhattan_maze.plot_behavior.plot_individual_memory` already applies to the
    smoothed solid lines of ``fig:ac_mem_gen`` A (``moving_average(..., mode="valid")`` drawn
    against ``xs[2:-2]``). At the defaults it drops two points per side, so the curve runs
    x = 0.118 to 0.882 inside an unchanged 0-to-1 axis.

    **Adjacent points overlap** (by ``1 - 1 / (win * (n_points - 1))``, ~70% at the defaults)
    and are therefore correlated. They may be read as a continuous estimate but must not be
    treated as independent samples.

    Windows are fitted **within** an animal before averaging across animals, so a mouse with
    a long journey does not dominate the cohort mean.
    """
    if mode not in ("valid", "same"):
        raise ValueError(f"mode must be 'valid' or 'same', got {mode!r}")
    grid = np.linspace(0, 1, n_points)
    # "valid" keeps only positions whose whole window fits inside the journey
    in_support = np.ones(n_points, dtype=bool) if mode == "same" else \
        (grid >= win / 2) & (grid <= 1 - win / 2)

    adjacency = np.asarray(sessions[0].mask.corridors_adj_mat)
    active = np.where(adjacency.sum(0) > 0)[0]
    degrees = {int(i): int((adjacency[i, active] > 0).sum()) for i in active}

    per_animal = []
    for session in sessions:
        corridor_seq = first_journey_corridor_seq(session)
        if corridor_seq.size < min_length:
            continue
        positions, is_reversal, scored_degrees = reversal_decisions(corridor_seq, degrees)
        if positions.size == 0:
            continue
        fractions = positions / (corridor_seq.size - 1)
        row = np.full(n_points, np.nan)
        for col in np.flatnonzero(in_support):
            selected = np.abs(fractions - grid[col]) <= win / 2
            if int(selected.sum()) >= min_decisions:
                row[col] = forward_bias_mle(is_reversal[selected], scored_degrees[selected])
        per_animal.append(row)

    # reshape keeps the empty-cohort case two-dimensional, so `supported` stays a mask
    per_animal = np.asarray(per_animal, dtype=float).reshape(-1, n_points)
    n_per_point = np.sum(~np.isnan(per_animal), 0)
    supported = n_per_point >= min_animals
    # only the supported columns are reduced, so nanmean/nanstd never see an empty slice
    cohort_mean, cohort_se = np.full(n_points, np.nan), np.full(n_points, np.nan)
    cohort_mean[supported] = np.nanmean(per_animal[:, supported], 0)
    cohort_se[supported] = (np.nanstd(per_animal[:, supported], 0)
                            / np.sqrt(n_per_point[supported]))
    return np.vstack([grid, cohort_mean, cohort_se])
