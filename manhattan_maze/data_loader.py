import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json
from glob import glob
import os
from copy import deepcopy
import sys
from manhattan_maze import utils
from manhattan_maze.mask import Mask, MaskDSpecial
from manhattan_maze.trajectory import Trajectory

# Grid dimension of one floor; all tile/corridor indices scale with this. Default
# maze topology assumed by the loader (override via the mask_size argument below).
MAZE_SIZE = 11


class DataLoader:
    """
    A DataLoader is a class that loads data from the manhattan maze dataset.
    The only required argument is the data_dir, which is the path to the data directory.
    You can access the trajectory of an experiment by calling DataLoader[nn], which returns a Trajectory object.
    """

    def __init__(self, data_dir="./data", # data directory
                 metadata_filename="manhattan_metadata_published.csv", # metadata file name (default path in data_dir)
                 metadata_indices=None, # indices of the metadata to use, if None, use all
                 mask_names=('O', 'A', 'AO', 'A_flipped', 'B', 'C', 'C_flipped', 'D', 'D_flipped', 'E', 'DT', 'F'), # names of the masks to use, the default sets of all
                 mask_filename_template="masks/holes_{name}.npy", # template for the mask filenames
                 raw_traj_filename_template="trajectories/raw/*{nn}*cpf_df.pickle", # template for the raw trajectory filenames
                 processed_traj_filename_template="trajectories/processed/{nn}.pickle", # template for the processed trajectory filenames
                 room_status_filename_template="verified_room_status/{nn}*.csv", # template for the room status filenames
                 reward_filename_template="rewards/{nn}*.csv", # template for the reward filenames
                 manual_fixes_path="manual_fixes/manual_fixes.json", # path to the manual fixes JSON file, if None, no manual fixes are applied
                 mask_size=MAZE_SIZE, # size of the mask, default is 11x11x2
                 FPS = 30, # frames per second, default is 30
                 min_frames_per_cell=1, # minimum number of frames per cell to consider it a valid cell, default is 1
                 force_reprocess=False, # if True, force reprocessing of the trajectories, default is False
                 check_trajectories=True, # if True, check the trajectories for errors, default is True
                 automated_fixes=True, # if True, apply automated fixes to the trajectories, default is True
                 manual_fixes=None, # if True, apply manual fixes to the trajectories, default is None
                 use_cache=True, # if True, use the cache to load trajectories (processed files in the local directory), default is True
                 exclude_HtM=True, # if True, exclude the HtM cells (home to maze) from the trajectory processing, default is False (fixing the last tile problem)
                 version=None, # DEPRECATED/IGNORED: only one processing method remains (was "v1"). Accepted for back-compat; remove from callers in Step 6.
                 check_out=None, # file to write the checks to, if None, do not write to file
                 manual_out=None, # file to write the manual fixes to, if None, do not write to file
                 home_coordinates = (0, 5, 0),  # home cell for the trajectory segmentation
                 out_coordinates = (5, 9, 1), # out cell for the trajectory segmentation
                 error_directory = "traj_errors", # directory to save the errors for manual fixes, default is "traj_errors"
                 verbose=False, # if False, only output big steps (major fixes), not each individual fix, default is False
                 ):
        """
        Configure the data loader and load all per-experiment index files.

        Only ``data_dir`` is required; every other argument has a production
        default. Per-argument meaning is given by the inline comments on the
        signature above; the scientifically load-bearing ones are:

        Parameters
        ----------
        data_dir : str
            Root data directory. The loader expects ``masks/``, ``trajectories/``,
            ``rewards/``, ``verified_room_status/`` and the metadata CSV; only
            ``masks/`` and the metadata CSV are distributed with this repository
            (see ``docs/data_contracts.md``).
        FPS : int, default 30
            Frames per second; sets the frame→**seconds** conversion for all
            durations and reward intervals.
        min_frames_per_cell : int, default 1
            Minimum frames to count a cell occupancy; >1 drops brief visits and
            changes bout structure (scientific invariant).
        home_coordinates, out_coordinates : tuple[int, int, int]
            Home ``(0,5,0)`` and out ``(5,9,1)`` ports; define which bouts are
            traverses vs sorties (scientific invariant).
        exclude_HtM : bool, default True
            Remove Home-to-Maze cells before segmentation (standard processing).
        mask_size : int, default MAZE_SIZE (11)
            Maze grid dimension (11); drives all tile/corridor indices.
        version : optional
            DEPRECATED and ignored — a single processing method remains.

        Notes
        -----
        Construction eagerly reads the metadata, room-status, mask-label,
        session-timestamp and reward index files for every nickname, and loads
        all masks. Trajectories themselves are loaded lazily via ``self[nn]``.
        See ``docs/data_contracts.md`` for the schema of each input.
        """
        self.data_dir = data_dir
        self.metadata_filename = metadata_filename
        self.metadata_indices = metadata_indices
        self.metadata = self._get_metadata(metadata_indices)
        self.FPS = FPS
        self.mask_size = mask_size

        self.cache = {} if use_cache else None
        self.check_trajectories = check_trajectories
        self.automated_fixes = automated_fixes
        self.verbose = verbose  # Add verbose flag to control printing
        self.manual_fixes = manual_fixes
        if manual_fixes_path is not None:
            manual_fixes_path = os.path.join(data_dir, manual_fixes_path)
            with open(manual_fixes_path, 'rb') as f:
                self.manual_fixes = json.load(f)

        # load files
        self.raw_traj_filename_template = raw_traj_filename_template
        self.room_status_filename_template = room_status_filename_template
        self.reward_filename_template = reward_filename_template
        self.nicknames = self.metadata["Nickname"].tolist() # list of experiment names
        self.raw_trajectory_files = {nn: self._get_raw_trajectory_filename(nn)
                                     for nn in self.nicknames}
        self._room_status = {nn: self._get_room_status(nn)
                             for nn in self.nicknames}
        self._mask_labels = {nn: self._get_mask_labels(nn)
                             for nn in self.nicknames}
        self._session_timestamps = {nn: self._get_session_timestamps(nn)
                                    for nn in self.nicknames}
        self._reward_dfs = {nn: self._get_reward_df(nn) for nn in self.nicknames}

        self.processed_traj_filename_template = processed_traj_filename_template

        self.min_frames_per_cell = min_frames_per_cell

        self.force_reprocess = force_reprocess

        # home and out coordinates, based on the default
        self.home_coordinates = home_coordinates
        self.home_pos = (home_coordinates[0], home_coordinates[1]) # pos for segmentation
        self.home_tile = utils.xyz_to_ti(home_coordinates, self.mask_size)
        self.home_corridor = utils.xyz_to_ci(home_coordinates, self.mask_size)[0]
        self.home_str = f"{home_coordinates[0]}-{home_coordinates[1]}" # for traj segmentation

        self.out_coordinates = out_coordinates
        self.out_pos = (out_coordinates[0], out_coordinates[1])
        self.out_tile = utils.xyz_to_ti(out_coordinates, self.mask_size)  # second floor
        self.out_corridor = utils.xyz_to_ci(out_coordinates, self.mask_size)[0] # integer
        self.out_str = f"{out_coordinates[0]}-{out_coordinates[1]+1}" # for segmentation

        # load masks
        self.mask_filename_template = mask_filename_template
        self.mask_names = mask_names
        self.masks = {mask_name: self._load_mask(mask_name) for mask_name in mask_names}

        # traject fixing
        self.exclude_HtM = exclude_HtM
        self.manual_out = open(manual_out, 'w') if manual_out is not None else sys.stdout
        self.check_out = open(check_out, 'w') if check_out is not None else sys.stdout

        # error outputs
        self.nn_error_outputs_dict = {}
        self.error_directory = error_directory

    def get_nicknames(self):
        """
        Return the experiment nicknames available in the loaded metadata.

        Returns
        -------
        list of str
            Experiment identifiers (e.g. ``"Z9_t1"``), in metadata row order.
            Each is a valid key for ``DataLoader[nn]``.
        """
        return self.nicknames

    def __getitem__(self, nn):
        """
        Load (and cache) the :class:`~manhattan_maze.trajectory.Trajectory` for one experiment.

        Parameters
        ----------
        nn : str
            Experiment nickname (e.g. ``"Z9_t1"``); one of :meth:`get_nicknames`.

        Returns
        -------
        Trajectory
            The processed, QC-fixed trajectory for ``nn``. When ``use_cache`` is
            enabled the result is memoised, so repeated access returns the same
            in-memory object.

        Notes
        -----
        On a cache/disk miss the raw trajectory is segmented and quality-
        controlled (or loaded from the processed-pickle cache); see
        :meth:`_load_trajectory`.
        """
        if self.cache is not None:
            if nn in self.cache:
                return self.cache[nn]
            else:
                self.cache[nn] = self._load_trajectory(nn)
                return self.cache[nn]
        else:
            return self._load_trajectory(nn)

    def __len__(self):
        """Return the number of experiments (nicknames) available to load."""
        return len(self.nicknames)

    # noinspection PyUnboundLocalVariable
    def _load_trajectory(self, nn):
        """
        Return the processed :class:`Trajectory` for one experiment.

        Loads the cached processed-bout pickle if present, otherwise calls
        :meth:`_generate_processed_trajectory` to segment and QC the raw data
        and writes the result to the processed-pickle cache. The processed
        bouts are then wrapped in a :class:`~manhattan_maze.trajectory.Trajectory`
        together with the experiment's mask order, masks, per-session frame
        segments and reward DataFrame.

        Parameters
        ----------
        nn : str
            Experiment nickname (e.g. ``"Z9_t1"``).

        Returns
        -------
        Trajectory
            Processed trajectory object for ``nn``.

        Notes
        -----
        Reprocessing is triggered when ``force_reprocess`` is set or the
        processed pickle is missing. When ``force_reprocess`` is set and QC
        errors were recorded for ``nn``, per-bout error plots and an
        ``errors_for_manual_fixes.csv`` are written under
        ``{data_dir}/{error_directory}/{nn}/``. Frame segments are the
        ``(first_frame, last_frame)`` pairs from the session timestamps; frames
        are absolute video frames (FPS=30).
        """
        processed_traj_filepath = os.path.join(self.data_dir,
                                               self.processed_traj_filename_template.format(nn=nn))

        # force process
        if self.force_reprocess or (not os.path.exists(processed_traj_filepath)):
            processed_traj = self._generate_processed_trajectory(nn)

            with open(processed_traj_filepath, 'wb') as traj_file:
                pickle.dump(processed_traj, traj_file)
        else:
            with open(processed_traj_filepath, 'rb') as traj_file:
                processed_traj = pickle.load(traj_file)

        # frames for segmentation
        frame_segments = [(first_frame, last_frame)
                          for _, (first_frame, last_frame)
                          in self._session_timestamps[nn]]

        # generate trajectory object
        traj = Trajectory(processed_traj,
                          mask_order=self._mask_labels[nn],
                          masks=self.masks,
                          name=nn,
                          frame_segments=frame_segments,
                          FPS=self.FPS,
                          rwd_df=self._reward_dfs[nn])

        # save error outputs
        if self.force_reprocess and self.nn_error_outputs_dict and nn in self.nn_error_outputs_dict.keys():
            all_errors = []
            os.makedirs(f"{self.data_dir}/{self.error_directory}/{nn}", exist_ok=True)
            for error in self.nn_error_outputs_dict[nn]:
                error['nn'] = nn
                all_errors.append(error)
                title = f"session={error['session']}_bout={error['bout']}_error={error['error_type']}"
                # print(title)
                traj[error['session']][error['bout']].plot()
                plt.title(title)
                plt.savefig(f"{self.data_dir}/{self.error_directory}/{nn}/{title}.png")
                plt.close()
            all_errors = pd.DataFrame(all_errors)
            if not all_errors.empty:
                all_errors.to_csv(f"{self.data_dir}/{self.error_directory}/{nn}/errors_for_manual_fixes.csv", index=False)

        return traj

    def _get_metadata(self, metadata_indices):
        """
        Read the metadata CSV and add the derived ``Animal`` column.

        Parameters
        ----------
        metadata_indices : array-like of int or None
            Row positions (0-based) to keep via ``.iloc``. If ``None``, all
            rows are used.

        Returns
        -------
        pandas.DataFrame
            Metadata table (schema in data contract §7) with the artefact
            ``Unnamed: 0`` column dropped and an added ``Animal`` column equal
            to ``Nickname.split("_")[0]``.
        """
        metadata_filepath = os.path.join(self.data_dir, self.metadata_filename)
        m_df = utils.drop_unnamed_column(pd.read_csv(metadata_filepath))
        if metadata_indices is not None:
            m_df = m_df.iloc[metadata_indices]
        # Add a column for animal names
        m_df["Animal"] = m_df["Nickname"].apply(lambda x: x.split("_")[0])
        return m_df

    def _load_raw_trajectory(self, nn):
        """
        Load the raw (per-frame) trajectory pickle for one experiment.

        Parameters
        ----------
        nn : str
            Experiment nickname (e.g. ``"Z9_t1"``).

        Returns
        -------
        pandas.DataFrame
            The ``raw_trajectory_df`` with at
            least the ``cell`` and ``frame`` columns; ``frame`` is the absolute
            video frame index. The pickle is read without schema validation.
        """
        filepath = self.raw_trajectory_files[nn]
        return pd.read_pickle(filepath)

    def _generate_processed_trajectory(self, nn, maze_water_port=None, cage_entrance_cell=None):
        """
        Segment and quality-control one experiment's raw trajectory into bouts.

        Parameters
        ----------
        nn : str
            Experiment nickname (e.g. ``"Z9_t1"``).
        maze_water_port : str or None
            Out/reward port cell label in ``"col-row"`` format. Defaults to the
            loader's ``out_str``.
        cage_entrance_cell : str or None
            Home/cage entrance cell label in ``"col-row"`` format. Defaults to
            the loader's ``home_str``.

        Returns
        -------
        list of list of pandas.DataFrame
            ``processed_bout_df_list``: one inner list per session, each holding
            the session's bout DataFrames (schema as in data contract §3).
        """
        if maze_water_port is None:
            #
            maze_water_port = self.out_str # a bit different from the out_str
        if cage_entrance_cell is None:
            cage_entrance_cell = self.home_str

        raw_trajectory = self._load_raw_trajectory(nn)
        sessions_in_exp = []
        for (maze_label, (start_frame, end_frame)) in self._session_timestamps[nn]:
            # use session timestamps to extract the trajectory for the session
            session_trajectory = raw_trajectory[raw_trajectory["frame"].between(start_frame, end_frame)]

            # drop all HtM
            if self.exclude_HtM:
                session_trajectory = session_trajectory[session_trajectory["cell"] != "HtM"]

            # generate cell sequence dataframe (condensed)
            reduced_session_trajectory = utils.generate_cell_sequence_df(session_trajectory, self.min_frames_per_cell)

            # For each configuration segment the total_cs_df into self, whenever the animal went out of the maze or solved the maze
            if self.exclude_HtM: # exclude HtM cells
                segment_indices = reduced_session_trajectory[(reduced_session_trajectory.cell == cage_entrance_cell) |
                                                             (reduced_session_trajectory.cell == maze_water_port)].index
            else: # use HtM for segmentation (no longer used)
                raise ValueError("exclude_HtM must be True")

            bouts_in_session = []
            # counter = 0 # debug usage
            # bout segmentation
            for start_idx, end_idx in zip(segment_indices[:-1], segment_indices[1:]):
                # extract the tiles in the bout
                # does not include the cell before the bout but included the result cell for reward
                cells_in_bout = reduced_session_trajectory.iloc[start_idx:end_idx]
                # remove the irrelevant cells, if Home cell is in it
                cells_in_bout = cells_in_bout[~(cells_in_bout.cell == "H")]
                cells_in_bout = cells_in_bout[~(cells_in_bout.cell == maze_water_port)]

                if len(cells_in_bout) < 3:
                    # skip; the mouse only poked inside the maze
                    continue

                if cage_entrance_cell != self.home_str: # if cage_entrance_cell is not the segmentation cells
                    cells_in_bout = cells_in_bout[~(cells_in_bout.cell == maze_water_port) & ~(cells_in_bout.cell == cage_entrance_cell)]
                # This would still have the last cell for video segmentation, so that the reward activity can be tracked
                bouts_in_session.append(cells_in_bout)
            sessions_in_exp.append(bouts_in_session)

        # save bout_df as raw data
        for session_idx, session in enumerate(sessions_in_exp):
            for bout_idx, bout_df in enumerate(session):
                sessions_in_exp[session_idx][bout_idx] = bout_df[bout_df["cell"] != "HtM"]

        # convert the sessions to tiles format
        bout_df_list = utils.format_sessions_with_tiles(sessions_in_exp)
        # check for errors
        if self.check_trajectories:
            # 1. Initialize the dictionary key with an empty list
            self.nn_error_outputs_dict[nn] = []

            # 2. Create a reference pointer to that specific list
            error_list = self.nn_error_outputs_dict[nn]

            # 3. Pass that reference. Any .append() inside _apply_fixes
            # modifies the list inside self.nn_error_outputs_dict[nn].
            bout_df_list = self._apply_fixes(nn, bout_df_list, error_list=error_list)

        return bout_df_list

    def _apply_fixes(self, nn, bout_df_list, error_list=None):
        """
        Run the full automated/manual QC pipeline over one experiment's bouts.

        Parameters
        ----------
        nn : str
            Experiment nickname.
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df`` (schema in data contract §3).
        error_list : list, optional
            Mutable list collecting unfixable errors (mutated in place). If
            ``None``, a fresh list is used.

        Returns
        -------
        list of list of pandas.DataFrame
            Bouts after all automated and manual fixes have been applied.

        Notes
        -----
        The step order is LOAD-BEARING and must run in exactly this sequence:
        empty -> missed segmentation -> manual bout fixes -> endpoint/first-and-
        last coords -> coords -> interpolation -> jumps -> repeated -> edge tiles
        -> manual tile fixes. Earlier steps establish invariants the later ones
        rely on (e.g. empty/short bouts removed before alternation and duration
        checks). Manual bout fixes run before manual tile fixes, and manual
        coord fixes are applied last so their indices reference positions after
        all automated fixes. See ``docs/data_contracts.md`` §3.
        """

        print(f"Checking experiment {nn}", file=self.check_out)
        if not self.verbose:
            print("Step by step check is hidden. Set verbose to True to inspect every automated check")

        # Apply checks in order: first remove empty/short bouts, then check trajectory logic
        print("Checking empty bouts...")
        bout_df_list = self._check_no_empty_bouts(bout_df_list)
        print("Checking missed segmentations...")
        bout_df_list = self._check_missed_segmentation(bout_df_list)
        bout_df_list = self._check_no_empty_bouts(bout_df_list) # check again to eliminate empty bouts added
        # Apply manual fixes for bouts first
        if not self.automated_fixes:
            print("Warning: manual fix files were compiled based on automated fixes.", file=self.check_out)
        bout_df_list = self._apply_manual_bout_fixes(nn, bout_df_list)
        # Check trajectory consistency
        print("Checking starting and ending coordinates...")
        bout_df_list = self._check_first_and_last_coords(bout_df_list)
        bout_df_list = self._validate_bouts_alternate(bout_df_list, error_list)
        print("Validating coordinates...")
        bout_df_list = self._check_coords_valid(nn, bout_df_list) # call nn when masks matter
        bout_df_list = self._check_skipped_coords(bout_df_list)
        bout_df_list = self._check_jumps_and_turns(nn, bout_df_list)
        bout_df_list = self._check_repeated_coords(bout_df_list)
        print("Validating durations...")
        bout_df_list = self._check_long_last_coords(bout_df_list)
        bout_df_list = self._check_long_first_coords(bout_df_list)
        bout_df_list = self._apply_manual_coord_fixes(nn, bout_df_list) # this goes after all the fixing is completed
        self._validate_and_record_errors(nn, bout_df_list, error_list)

        if self.verbose:
            print("--------------------", file=self.check_out, flush=True)
        else:
            print(f"Experiment {nn} checks completed", file=self.check_out, flush=True)

        return bout_df_list

    @staticmethod
    def _all_turns_correct(bout, holes_list):
        """
        Test whether every turn in a bout is geometrically valid.

        A turn is valid only if it is a straight run (same corridor) or a 90-
        degree turn that occurs at a hole; diagonal moves are always invalid.
        Both ``prev->curr`` and ``curr->post`` transitions are inspected to
        enforce corridor continuity.

        Parameters
        ----------
        bout : pandas.DataFrame
            A single ``bout_df`` with a ``discrete_loc`` column of ``(col, row)``
            tuples.
        holes_list : list of tuple of int
            Valid hole coordinates ``(col, row)`` where turns are allowed.

        Returns
        -------
        bool
            ``True`` if all turns are valid, ``False`` on the first diagonal
            move or turn not at a hole.
        """
        for ix in range(1, len(bout) - 1):
            coords_seq = bout["discrete_loc"]
            prev, curr, post = (
                coords_seq[ix - 1],
                coords_seq[ix],
                coords_seq[ix + 1],
            )

            # Turn error: Turn not at a hole (only if both transitions are valid)
            if utils.same_corridor([prev, curr, post]):
                continue
            elif utils.is_diagonal(prev, curr): #only compare with pre to avoid double records
                return False
            elif utils.is_turn(prev, curr, post) and curr not in holes_list:
                return False
        
        return True

    @staticmethod
    def _get_incorrect_turns(bout, holes_list):
        """
        List every invalid turn in a bout with its coordinates and error type.

        Parameters
        ----------
        bout : pandas.DataFrame
            A single ``bout_df`` with a ``discrete_loc`` column and ``in_frame``/
            ``out_frame`` columns (absolute video frames).
        holes_list : list of tuple of int
            Valid hole coordinates ``(col, row)`` where turns can occur.

        Returns
        -------
        list of dict
            One dict per offending tile, each with keys:
            ``"coords"`` (0-based tile index within the bout), ``"prev"``,
            ``"curr"``, ``"post"`` (the three ``(col, row)`` coordinates),
            ``"error_type"`` (``"diagonal_turn"`` or ``"turn_not_in_hole"``),
            and ``"in_frame"``/``"out_frame"`` (the problematic tile's frames).
        """
        incorrect_turns = []
        coords_seq = bout["discrete_loc"]
        
        for ix in range(1, len(bout) - 1):
            prev, curr, post = (
                coords_seq[ix - 1],
                coords_seq[ix],
                coords_seq[ix + 1],
            )
            if utils.is_diagonal(prev, curr):
                incorrect_turns.append({
                    "coords": ix,
                    'prev': prev,
                    'curr': curr,
                    'post': post,
                    'error_type': 'diagonal_turn',
                    'in_frame': bout['in_frame'].iloc[ix],
                    'out_frame': bout['out_frame'].iloc[ix],
                })
            elif utils.is_turn(prev, curr, post) and curr not in holes_list:
                incorrect_turns.append({
                    "coords": ix,
                    'prev': prev,
                    'curr': curr,
                    'post': post,
                    'error_type': 'turn_not_in_hole',
                    'in_frame': bout['in_frame'].iloc[ix],
                    'out_frame': bout['out_frame'].iloc[ix],
                })
        return incorrect_turns

    def _check_missed_segmentation(self, bout_df_list):
        """
        Split bouts that pass straight through a port mid-bout.

        Detects interior tiles equal to ``out_pos`` (when its neighbours share
        the column) or ``home_pos`` (when its neighbours share the row), which
        indicate a missed segmentation, and splits the bout after each such
        point into separate bouts.

        Parameters
        ----------
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df`` (schema in data contract §3).

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with offending bouts split. Returned unchanged when
            ``automated_fixes`` is ``False``.
        """
        new_tiles = []

        for session_idx, session in enumerate(bout_df_list):
            reconstructed_session = []
            for bout_idx, bout in enumerate(session):
                discrete_locs = bout['discrete_loc']
                # Find all indices that trigger a split within this specific bout
                split_at = []
                for k in range(1, len(discrete_locs) - 1):
                    loc = discrete_locs[k]
                    prev_loc = discrete_locs[k - 1]
                    next_loc = discrete_locs[k + 1]

                    # Logic for Out Pos (Column) or Home Pos (Row)
                    # Check Out Position logic (Column match)
                    if loc == self.out_pos and utils.same_col([prev_loc, loc, next_loc]):
                        is_split = True
                    elif loc == self.home_pos and utils.same_row([prev_loc, loc, next_loc]):
                        is_split = True
                    else:
                        is_split = False

                    if is_split:
                        if self.verbose:
                            print(f"Split found: Session {session_idx}, Bout {bout_idx} at index {k}", file=self.check_out)
                        split_at.append(k + 1)  # Split after the identified point

                if not split_at:
                    # No splits needed, keep the bout as is
                    reconstructed_session.append(deepcopy(bout))
                else:
                    # We have one or more split points.
                    # We slice the original bout data into multiple new bouts.
                    start_idx = 0
                    # Ensure split points are sorted and include the total length of the DataFrame
                    total_len = len(bout)
                    all_split_points = sorted(split_at) + [total_len]

                    for end_idx in all_split_points:
                        # Use .iloc to slice the DataFrame rows [start:end]
                        # This automatically keeps all columns (time, velocity, etc.) synchronized
                        sub_bout_df = bout.iloc[start_idx:end_idx].copy().reset_index(drop=True)

                        # Only add the bout if it's not empty (prevents issues with consecutive split points)
                        if not sub_bout_df.empty:
                            reconstructed_session.append(sub_bout_df)

                        # The next segment starts where this one ended
                        start_idx = end_idx
            new_tiles.append(reconstructed_session)
        return new_tiles if self.automated_fixes else bout_df_list

    def _handle_jumped_coordinate(self, bout_copy, ix):
        """
        Try to correct a single jumped (diagonal) coordinate in place.

        A jumped coordinate changes in both ``col`` and ``row`` when only one
        should change. This handles two patterns: a ping-pong jump
        (``prev == post``, replaced by ``prev``) and an off-track jump where
        ``prev`` and ``post`` lie in the same corridor and that line is
        supported by the surrounding context (replaced by the midpoint of
        ``prev`` and ``post``).

        Parameters
        ----------
        bout_copy : pandas.DataFrame
            Working copy of a single ``bout_df``; modified in place when a
            correction is found.
        ix : int
            0-based index of the coordinate under inspection.

        Returns
        -------
        tuple of (pandas.DataFrame, bool)
            The (possibly modified) bout and a flag that is ``True`` when a
            correction was applied (and the caller should advance), ``False``
            when no fix was possible (e.g. ``post`` is missing).
        """
        # TRY 1: Intelligently correct the jumped coordinate using context
        # STRATEGY 1: Look backward at pre_pre->prev pattern
        discrete_locs = bout_copy['discrete_loc']
        curr = discrete_locs[ix]
        prev = discrete_locs[ix-1]
        post = discrete_locs[ix+1] if ix < len(discrete_locs) - 1 else None
        corrected_coord = None
        # Simplified and more robust jumping detection
        if post is None:
            # Cannot fix when there is not enough info for comparison
            return bout_copy, False
        if prev == post:
            # Classic "Point-A -> Point-B -> Point-A" jump
            corrected_coord = prev
            if self.verbose:
                print(f"Corrected ping-pong jump: {curr} -> {corrected_coord}")
        elif utils.same_corridor([prev, post]):
            # Gather extended context
            pre_pre = discrete_locs[ix - 2] if ix >= 2 else None
            post_post = discrete_locs[ix + 2] if ix < len(discrete_locs) - 2 else None

            # Strict check: Is the straight line [prev -> post] supported by
            # the coordinate immediately before OR immediately after?
            is_supported_pre = pre_pre is not None and utils.same_corridor([pre_pre, prev, post])
            is_supported_post = post_post is not None and utils.same_corridor([prev, post, post_post])
            if is_supported_pre or is_supported_post:
                # The jump is confirmed as off-track because the animal was
                # already in this corridor or continued in it.
                corrected_coord = ((prev[0] + post[0]) // 2, (prev[1] + post[1]) // 2)
                if self.verbose:
                    print(f"Strict corrected off-track jump: {curr} -> {corrected_coord} "
                          f"(Supported by: {'pre' if is_supported_pre else ''} "
                          f"{'post' if is_supported_post else ''})")

        # Update the DataFrame if a correction was found
        if corrected_coord is not None:
            bout_copy.at[ix, 'discrete_loc'] = corrected_coord
            return bout_copy, True
        else:
            if self.verbose:
                print(f"Jumped coordinate at {ix} cannot be fixed. Now apply turn fix.", file=self.check_out)
            return bout_copy, False  # Cannot be fixed with jumping coordinates

    def _attempt_turn_fix(self, bout_copy, ix, prev, curr, post, holes_list):
        """
        Decide which side of a turn is wrong and delegate the tile insertion.

        Identifies whether the diagonal occurs on the ``prev->curr`` transition
        (wrong index ``ix-1``) or the ``curr->post`` transition (wrong index
        ``ix``), then calls :meth:`_apply_turn_fixing_to_bout` to insert a
        bridging hole tile.

        Parameters
        ----------
        bout_copy : pandas.DataFrame
            Working copy of a single ``bout_df``.
        ix : int
            0-based index of the current coordinate.
        prev, curr, post : tuple of int
            The three consecutive ``(col, row)`` coordinates around ``ix``.
        holes_list : list of tuple of int
            Valid hole coordinates where a turn may be inserted.

        Returns
        -------
        pandas.DataFrame or None
            The bout with an intermediate tile inserted, or ``None`` if neither
            transition is diagonal or no valid bridging tile could be found.
        """
        # IDENTIFY: Which coordinate has the problematic turn?
        # Option A: prev -> curr is diagonal (turn at prev position)
        if prev[0] != curr[0] and prev[1] != curr[1]:
            if self.verbose:
                print("Apply turn fixing to pre", file=self.check_out)
            wrong_ix = ix - 1 # put the before turn at the first
        # Option B: curr -> post is diagonal (turn at curr position)
        elif curr[0] != post[0] and curr[1] != post[1]:
            if self.verbose:
                print("Apply turn fixing to post", file=self.check_out)
            wrong_ix = ix
        # Option C: Neither transition is diagonal - no turn to fix
        else:
            if self.verbose:
                print("No turn fixing applied")
            return None
        
        # ATTEMPT: Call the turn fixing method to add intermediate tile
        fixed_bout = self._apply_turn_fixing_to_bout(bout_copy, wrong_ix, holes_list)
        return fixed_bout

    def _fix_bout_turns(self, bout, holes_list):
        """
        Iteratively repair all turn errors in a single bout.

        Walks the bout with a dynamic index (the bout length can change as
        tiles are added or corrected), and at each step that is neither a
        straight run nor a valid in-hole turn first attempts a jumped-coordinate
        fix (:meth:`_handle_jumped_coordinate`) and then a turn fix
        (:meth:`_attempt_turn_fix`). Unfixable turns are left in place for later
        validation/manual review.

        Parameters
        ----------
        bout : pandas.DataFrame
            A single ``bout_df`` (not modified; a copy is returned).
        holes_list : list of tuple of int
            Valid hole coordinates ``(col, row)`` for this session's mask.

        Returns
        -------
        pandas.DataFrame
            A new bout with turn errors corrected where possible.
        """
        # 1. Work on a copy to avoid side effects
        bout_copy = bout.copy()

        # Use a while loop because len(bout_copy) changes when we add/remove tiles
        ix = 1

        while ix < len(bout_copy) - 1:
            # Re-extract coordinates at each step because the DataFrame may have changed
            coords_seq = bout_copy['discrete_loc'].values
            prev, curr, post = coords_seq[ix - 1], coords_seq[ix], coords_seq[ix + 1]

            # Check if the current configuration is valid
            is_straight = utils.same_corridor([prev, curr, post])
            is_valid_turn = utils.is_turn(prev, curr, post) and curr in holes_list

            if is_straight or is_valid_turn:
                ix += 1
                continue
            # Attempt Fix 1: Handle jumped coordinates (usually re-assigning a value)
            bout_copy, fixed = self._handle_jumped_coordinate(bout_copy, ix+1) # center the post

            if fixed:
                ix +=1
                continue

            # We pass the index and context to the fixer
            fixed_bout = self._attempt_turn_fix(bout_copy, ix, prev, curr, post, holes_list)

            if fixed_bout is not None:
                bout_copy = fixed_bout
                # We want to re-check the current index against the new neighbors
                # created by the interpolation to ensure the new transition is also valid.
                continue
            else:
                if self.verbose:
                    print(f"Could not fix turn at index {ix} with coords {curr}. Marking for manual review.", file=self.check_out)
                ix +=1

        return bout_copy

    def _validate_and_record_errors(self, nn, bout_df_list, error_list=None):
        """
        Record any turn errors that survived the automated fixes.

        Iterates every bout, recomputes per-session holes, and uses
        :meth:`_get_incorrect_turns` to find remaining diagonal/illegal turns,
        appending one entry per error to ``error_list`` and printing a summary
        to ``check_out``. This is a validation-only pass; bouts are not modified.

        Parameters
        ----------
        nn : str
            Experiment nickname (used to look up the per-session mask sequence).
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        error_list : list, optional
            Mutable list to append error dicts to (mutated in place). If
            ``None``, a fresh list is created.

        Returns
        -------
        list of dict
            The ``error_list``, each entry carrying ``"session"`` and ``"bout"``
            (0-based) plus the fields from :meth:`_get_incorrect_turns`.
        """
        if error_list is None:
            error_list = []

        # 1. Pre-calculate holes to avoid repeated lookups
        mask_names = self._mask_labels[nn]
        mask_holes_per_session = [self.masks[name].holes_list for name in mask_names]

        for session_idx, session in enumerate(bout_df_list):
            # Get the specific holes list for this session
            holes_list = mask_holes_per_session[session_idx]

            for bout_idx, bout in enumerate(session):

                # 2. Identify remaining incorrect transitions
                incorrect_turns = self._get_incorrect_turns(bout, holes_list)

                # 3. Only act if there are actually errors
                if incorrect_turns:
                    print(
                        f"Validation Error: {nn}, session {session_idx}, bout {bout_idx}. "
                        f"Found {len(incorrect_turns)} invalid turn(s). Manual fix required.",
                        file=self.check_out
                    )

                    # 4. Record details for each error
                    for error_info in incorrect_turns:
                        # Merge coordinate info with session/bout context
                        entry = {
                            'session': session_idx,
                            'bout': bout_idx,
                            **error_info  # Includes 'index', 'coord', etc., from _get_incorrect_turns
                        }
                        error_list.append(entry)

        # Note: Function modifies error_list in-place, so no return is strictly necessary,
        # but returning it is often helpful for chaining.
        return error_list


    def _apply_turn_fixing_to_bout(self, bout_df, ix, hole_list):
        """
        Insert one bridging hole tile to repair a turn between two tiles.

        Searches for a valid hole that connects ``coords_seq[ix]`` and
        ``coords_seq[ix+1]`` (falling back to extended backward/forward context
        if needed), then inserts it as a new row between them, splitting the
        surrounding ``in_frame``/``out_frame`` boundaries at their midpoints so
        frame coverage stays contiguous.

        Parameters
        ----------
        bout_df : pandas.DataFrame
            A single ``bout_df`` with ``discrete_loc``, ``in_frame`` and
            ``out_frame`` columns (frames are absolute video frames).
        ix : int
            0-based index of the tile before the problematic turn.
        hole_list : list of tuple of int
            Valid hole coordinates ``(col, row)`` where turns can occur.

        Returns
        -------
        pandas.DataFrame or None
            A new bout with the intermediate hole tile inserted, or ``None`` if
            no valid bridging tile (distinct from both endpoints) could be
            found.
        """
        # EXTRACT: Get coordinate and time data
        coords_seq = bout_df['discrete_loc'].values
        in_frames = bout_df['in_frame'].values
        out_frames = bout_df['out_frame'].values

        pre_turn = coords_seq[ix]
        post_turn = coords_seq[ix+1]
        # CALCULATE: Frame boundaries for the new intermediate tile by splitting the pre_turn and post_turn
        in_frame = (in_frames[ix] + out_frames[ix]) // 2
        out_frame = (out_frames[ix+1] + in_frames[ix+1]) // 2
        
        # SEARCH 1: Try direct connection between pre_turn and post_turn
        add_tile = utils.add_turn_at_hole(pre_turn, post_turn, hole_list)
        
        # SEARCH 2: If failed, use additional backward context
        if add_tile is None and ix - 1 >= 0:
            pre_pre = coords_seq[ix - 1]
            add_tile = utils.add_turn_at_hole(pre_pre, post_turn, hole_list)
        
        # SEARCH 3: If failed, use additional forward context
        if add_tile is None and ix + 2 <= len(coords_seq) - 1:
            post_post = coords_seq[ix + 2]
            add_tile = utils.add_turn_at_hole(pre_turn, post_post, hole_list)
        
        # VALIDATE: Check if we found a valid intermediate tile
        if add_tile is None or add_tile == pre_turn or add_tile == post_turn:
            return None  # Couldn't find valid intermediate tile

        # With correct add_tile, now add to the bout_df
        # BUILD: New bout with the intermediate tile inserted
        new_bout_data = {col: bout_df[col].to_list() for col in bout_df.columns}
        
        # UPDATE: Adjust frame boundaries at insertion points
        new_bout_data['out_frame'][ix] = in_frame
        new_bout_data['in_frame'][ix + 1] = out_frame
        # add the new row with add_tile
        new_bout_data["discrete_loc"].insert(ix+1, add_tile)
        new_bout_data["in_frame"].insert(ix+1, in_frame)
        new_bout_data["out_frame"].insert(ix+1, out_frame)
        if self.verbose:
            print(f"Added intermediate tile at {ix}: {add_tile}", file=self.check_out)
        # CREATE: Return the fixed bout
        modified_bout = pd.DataFrame(new_bout_data)
        return modified_bout

    def _check_jumps_and_turns(self, nn, bout_df_list):
        """
        Fix turn errors across every bout using this experiment's masks.

        Looks up the per-session mask sequence to obtain each session's hole
        list, then applies :meth:`_fix_bout_turns` to every bout (or leaves it
        unchanged in validation-only mode).

        Parameters
        ----------
        nn : str
            Experiment nickname (used to resolve per-session masks).
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with turn errors corrected. Bouts are passed through
            unchanged when ``automated_fixes`` is ``False``.
        """
        print("Checking jumps and turns...")
        # 1. Pre-fetch mask data to avoid repeated lookups in the loop
        mask_names = self._mask_labels[nn]
        # Ensure we handle sessions that might share or have different masks
        mask_holes_per_session = [self.masks[name].holes_list for name in mask_names]

        new_bout_list = []

        for session_idx, session in enumerate(bout_df_list):
            new_session = []
            holes_list = mask_holes_per_session[session_idx]

            for bout_idx, bout in enumerate(session):
                if self.automated_fixes:
                    # IMPORTANT: Pass only the specific bout to the helper
                    # to prevent the helper from accidentally modifying the wrong data
                    fixed_bout = self._fix_bout_turns(bout, holes_list)
                    new_session.append(fixed_bout)
                else:
                    # Just append the original (validation only mode)
                    new_session.append(bout)
            new_bout_list.append(new_session)

        return new_bout_list


    def _check_long_last_coords(self, bout_df_list, threshold=2):
        """
        Trim an over-long final tile to the previous tile's duration.

        When the last tile's frame span exceeds ``threshold`` times the previous
        tile's span, its ``in_frame``/``out_frame`` are rewritten so its
        duration matches the previous tile (starting one frame after the
        previous tile's ``out_frame``).

        Parameters
        ----------
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        threshold : int, default 2
            Duration-ratio threshold above which the final tile is trimmed.

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with over-long final tiles trimmed. Returned unchanged
            when ``automated_fixes`` is ``False``.

        Raises
        ------
        AssertionError
            If any bout has length <= 1 (short bouts must already be removed).
        """
        new_bout_list = deepcopy(bout_df_list)
        for session_idx, session in enumerate(bout_df_list):
            for bout_idx, bout in enumerate(session):
                assert len(bout) > 1, "Short bouts should have been removed. Check errors!"
                last_tile = bout.iloc[-1]
                prev_tile = bout.iloc[-2]
                last_duration = last_tile['out_frame'] - last_tile['in_frame']
                prev_duration = prev_tile['out_frame'] - prev_tile['in_frame']
                if last_duration > threshold * prev_duration:
                    if self.verbose:
                        print(f"Last tile in session {session_idx} bout {bout_idx} is too long (duration {last_duration} vs {prev_duration})",
                              file=self.check_out)
                    if self.automated_fixes:
                        new_bout_list = self._edit_coords(new_bout_list, session_idx, bout_idx, len(bout) - 1,
                                                      new_vals={"in_frame": prev_tile['out_frame']+1,
                                                              "out_frame": prev_tile['out_frame'] + prev_duration})
        return new_bout_list if self.automated_fixes else bout_df_list

    def _check_long_first_coords(self, bout_df_list, threshold=2):
        """
        Trim an over-long first tile to the second tile's duration.

        When the first tile's frame span exceeds ``threshold`` times the second
        tile's span, its ``in_frame``/``out_frame`` are rewritten so its
        duration matches the second tile (ending one frame before the second
        tile's ``in_frame``).

        Parameters
        ----------
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        threshold : int, default 2
            Duration-ratio threshold above which the first tile is trimmed.

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with over-long first tiles trimmed. Returned unchanged when
            ``automated_fixes`` is ``False``.

        Raises
        ------
        AssertionError
            If any bout has length <= 1 (short bouts must already be removed).
        """
        new_bout_list = deepcopy(bout_df_list)
        for session_idx, session in enumerate(bout_df_list):
            for bout_idx, bout in enumerate(session):
                assert len(bout) > 1, "Short bouts should have been removed. Check errors!"
                first_tile = bout.iloc[0]
                second_tile = bout.iloc[1]
                first_duration = first_tile['out_frame'] - first_tile['in_frame']
                second_duration = second_tile['out_frame'] - second_tile['in_frame']
                if first_duration > threshold * second_duration:
                    if self.verbose:
                        print(f"First tile in session {session_idx} bout {bout_idx} is too long (duration {first_duration} vs {second_duration})",
                              file=self.check_out)
                    if self.automated_fixes:
                        new_bout_list = self._edit_coords(new_bout_list, session_idx, bout_idx, 0,
                                                      new_vals={"in_frame": second_tile['in_frame']-second_duration,
                                                              "out_frame": second_tile['in_frame']-1})
        return new_bout_list if self.automated_fixes else bout_df_list


    def _check_no_empty_bouts(self, bout_df_list, min_length=3):
        """
        Drop bouts shorter than the minimum tile count.

        Parameters
        ----------
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        min_length : int, default 3
            Minimum number of rows (tiles) for a bout to be kept; a valid bout
            has >= 3 rows (data contract §3).

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with short bouts removed. Returned unchanged when
            ``automated_fixes`` is ``False``.
        """
        # If automated fixes are off, return the original list immediately
        new_bout_list = []

        for s_idx, session in enumerate(bout_df_list):
            # Use a list comprehension to filter the session
            # This is significantly faster and avoids index-shifting bugs
            filtered_session = [
                bout for b_idx, bout in enumerate(session)
                if self._is_valid_bout(bout, s_idx, b_idx, min_length)
            ]

            new_bout_list.append(filtered_session)

        return new_bout_list if self.automated_fixes else bout_df_list

    def _is_valid_bout(self, bout, s_idx, b_idx, min_length):
        """
        Return whether a bout meets the minimum length, logging removals.

        Parameters
        ----------
        bout : pandas.DataFrame
            A single ``bout_df``.
        s_idx, b_idx : int
            0-based session and bout indices (used only for log messages).
        min_length : int
            Minimum number of rows (tiles) required.

        Returns
        -------
        bool
            ``True`` if ``len(bout) >= min_length``, else ``False``.
        """
        if len(bout) < min_length:
            if self.verbose:
                print(f"Session {s_idx}, Bout {b_idx} removed: "
                      f"length {len(bout)} < {min_length}", file=self.check_out)
            return False
        return True

    def _check_first_and_last_coords(self, bout_df_list, error_list=None):
        """
        Extend each bout so its first and last tiles are ports.

        Post-QC, row 0 and row -1 of every bout must be ``home_pos = (0, 5)`` or
        ``out_pos = (5, 9)`` (data contract §3). When an endpoint is some other
        tile, :meth:`_get_endpoint_extension` synthesises the missing tiles
        along the home row or out column and prepends/appends them; failures to
        fix the end are logged as ``"endpoint"`` errors.

        Parameters
        ----------
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        error_list : list, optional
            Mutable list to append endpoint-fix failures to. If ``None``, a
            fresh list is used.

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with endpoints extended to ports. Returned unchanged when
            ``automated_fixes`` is ``False``.
        """
        if error_list is None:
            error_list = []

        new_bout_list = []

        for s_idx, session in enumerate(bout_df_list):
            reconstructed_session = []

            for b_idx, bout in enumerate(session):
                if bout.empty:
                    reconstructed_session.append(bout)
                    continue

                # 1. Extract boundary information
                first_row = bout.iloc[0].to_dict()
                last_row = bout.iloc[-1].to_dict()
                first_tile = first_row['discrete_loc']
                last_tile = last_row['discrete_loc']

                # Prepare the list of rows for the new bout
                # We will collect dicts and convert to DataFrame once at the end
                bout_rows = bout.to_dict('records')

                # 2. Check and fix the START of the bout
                if first_tile not in [self.home_pos, self.out_pos]:
                    start_fixes = self._get_endpoint_extension(
                        first_row, target_home=self.home_pos, target_out=self.out_pos, mode='start'
                    )
                    if start_fixes:
                        # Insert fixes at the beginning of the list
                        bout_rows = start_fixes + bout_rows
                    elif self.verbose:
                        print(f"S{s_idx} B{b_idx} start fix failed at {first_tile}", file=self.check_out)

                # 3. Check and fix the END of the bout
                if last_tile not in [self.home_pos, self.out_pos]:
                    end_fixes = self._get_endpoint_extension(
                        last_row, target_home=self.home_pos, target_out=self.out_pos, mode='end'
                    )
                    if end_fixes:
                        # Append fixes to the end of the list
                        bout_rows = bout_rows + end_fixes
                    else:
                        if self.verbose:
                            print(f"S{s_idx} B{b_idx} end fix failed at {last_tile}", file=self.check_out)
                        error_list.append({'session': s_idx, 'bout': b_idx, 'error_type': 'endpoint',
                                           'coords':-1, })

                reconstructed_session.append(pd.DataFrame(bout_rows))

            new_bout_list.append(reconstructed_session)

        return new_bout_list if self.automated_fixes else bout_df_list

    @staticmethod
    def _get_endpoint_extension(reference_row, target_home, target_out, mode='start'):
        """
        Build the tile rows linking a reference tile to its nearest port.

        Generates one synthetic tile per step along the home row (if the
        reference shares the home row) or the out column (if it shares the out
        column), with ``in_frame``/``out_frame`` offset by integer multiples of
        the reference tile's duration (negative offsets for ``mode='start'``,
        positive for ``mode='end'``). Frames are absolute video frames.

        Parameters
        ----------
        reference_row : dict
            The boundary tile, with ``discrete_loc`` ``(col, row)`` and
            ``in_frame``/``out_frame``.
        target_home, target_out : tuple of int
            Home ``(0, 5)`` and out ``(5, 9)`` port coordinates.
        mode : {'start', 'end'}, default 'start'
            Whether the gap is before the bout's first tile or after its last.

        Returns
        -------
        list of dict or None
            New tile rows ordered to lead into (start) or out of (end) the bout,
            or ``None`` if the reference tile shares neither the home row nor the
            out column (cannot be fixed automatically).
        """
        col, row = reference_row['discrete_loc']
        in_f, out_f = reference_row['in_frame'], reference_row['out_frame']
        duration = out_f - in_f

        new_rows = []

        # Logic for Home Position corridor (Row match)
        if row == target_home[1]:
            # Determine range (exclusive of the reference tile itself)
            step = -1 if mode == 'start' else 1
            path = range(col + step, target_home[0] + step, step)

        # Logic for Out Position corridor (Column match)
        elif col == target_out[0]:
            step = 1 if target_out[1] > row else -1
            path = range(row + step, target_out[1] + step, step)
            # Handle the coordinate inside the coordinate tuple correctly below
        else:
            return None  # Cannot fix automatically

        # Build the list of dictionaries
        for i, val in enumerate(path):
            new_tile = reference_row.copy()
            # Determine if 'val' is a new column or new row
            loc = (val, row) if row == target_home[1] else (col, val)

            # Calculate shifted frames based on mode
            # Start mode: subtract time to go backwards; End mode: add time to go forwards
            multiplier = (i + 1)
            offset = -multiplier * duration if mode == 'start' else multiplier * duration

            new_tile.update({
                'discrete_loc': loc,
                'in_frame': in_f + offset,
                'out_frame': out_f + offset
            })
            new_rows.append(new_tile)

        # If fixing the start, the path was generated moving away from the bout,
        # so we reverse it to ensure it leads INTO the first tile.
        return new_rows[::-1] if mode == 'start' else new_rows

    def _log_skip(self, session_idx, bout_idx, tile_idx, prev_tile, curr_tile, axis_name):
        """
        Print a one-line note about a detected coordinate gap.

        Parameters
        ----------
        session_idx, bout_idx, tile_idx : int
            0-based session, bout and tile indices of the gap (the gap is
            between ``tile_idx - 1`` and ``tile_idx``).
        prev_tile, curr_tile : tuple of int
            The ``(col, row)`` coordinates on either side of the gap.
        axis_name : str
            Human-readable name of the axis that skipped (``"row"`` or
            ``"column"``).
        """
        print(f"skipped {axis_name} in session {session_idx}, bout {bout_idx}, "
              f"between tiles {tile_idx - 1} {prev_tile} and {tile_idx} {curr_tile}",
              file=self.check_out)

    def _check_skipped_coords(self, bout_df_list, min_length=3):
        """
        Interpolate straight-line gaps (teleports) between consecutive tiles.

        Where two consecutive tiles differ by more than one step along a single
        axis (the other axis unchanged), the missing tiles are filled in via
        ``utils.interpolate_coords_with_frames`` and frame coverage is made
        contiguous. Simultaneous changes in both axes are flagged as turn-fixing
        cases and left for the turn logic rather than interpolated here.

        Parameters
        ----------
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        min_length : int, default 3
            Bouts shorter than this are dropped (a valid bout has >= 3 rows).

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with single-axis gaps interpolated. Returned unchanged when
            ``automated_fixes`` is ``False``.
        """
        new_bout_list = []  # Rebuilding the session list from scratch for speed

        for session_idx, session in enumerate(bout_df_list):
            reconstructed_session = []

            for bout_idx, bout in enumerate(session):
                if len(bout) < min_length:
                    # removing invalid tiles might lead to shorter bouts
                    continue
                # Convert to numpy for vectorized distance calculations
                locs = np.stack(bout['discrete_loc'].values)
                cols, rows = locs[:, 0], locs[:, 1]

                # Calculate differences between consecutive tiles
                d_cols = np.abs(np.diff(cols))
                d_rows = np.abs(np.diff(rows))

                # A gap exists if movement > 1 in one axis while the other is 0
                # Or if both axes changed (requires manual fix/turn)
                has_gap = (d_cols > 1) | (d_rows > 1) | ((d_cols > 0) & (d_rows > 0))

                if not np.any(has_gap):
                    reconstructed_session.append(bout.copy())
                    continue

                # We will rebuild this specific bout row-by-row
                new_bout_rows = []

                # Iterate through the original bout's rows
                for i in range(len(bout)):
                    current_row = bout.iloc[i].to_dict()

                    if i > 0:
                        pc, pr = cols[i - 1], rows[i - 1]  # Previous coords
                        c, r = cols[i], rows[i]  # Current coords
                        dc, dr = d_cols[i - 1], d_rows[i - 1]

                        # Check for straight-line gaps
                        if (dc == 0 and dr > 1) or (dr == 0 and dc > 1):
                            axis = "row" if dr > 1 else "col"
                            if self.verbose:
                                self._log_skip(session_idx, bout_idx, i, (pc, pr), (c, r), "column" if dr > 1 else "row")

                            # This now returns [Edited_Prev_Tile, New_Mid_Tile_1, New_Mid_Tile_2, ...]
                            interpolated_sequence = utils.interpolate_coords_with_frames(bout.iloc[i - 1], bout.iloc[i],
                                                                                         axis)

                            if interpolated_sequence:
                                # Remove the 'old' previous tile that was added in the last iteration
                                new_bout_rows.pop()
                                # Add the corrected previous tile + all missing tiles
                                new_bout_rows.extend(interpolated_sequence)

                                # Update the current tile's in_frame to perfectly follow the last interpolated tile
                                current_row['in_frame'] = interpolated_sequence[-1]['out_frame'] + 1

                        elif dc > 0 and dr > 0:
                            if self.verbose:
                                print(f"Turn fixing required: Session {session_idx}, Bout {bout_idx} at indices {i - 1}-{i}",
                                      file=self.check_out)

                    new_bout_rows.append(current_row)
                # Convert the list of dicts back into a DataFrame
                reconstructed_session.append(pd.DataFrame(new_bout_rows))

            new_bout_list.append(reconstructed_session)

        return new_bout_list if self.automated_fixes else bout_df_list

    def _check_repeated_coords(self, tiles):
        """
        Collapse consecutive duplicate tiles within each bout.

        Scanning each bout from the end, a tile equal to its predecessor is
        removed and the predecessor's ``out_frame`` is extended to the removed
        tile's ``out_frame`` so no frame coverage is lost.

        Parameters
        ----------
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df`` (with ``discrete_loc`` and
            ``out_frame`` columns; frames are absolute video frames).

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with consecutive repeated tiles merged. Returned unchanged
            when ``automated_fixes`` is ``False``.
        """
        new_tiles = deepcopy(tiles)
        for session_idx, session in enumerate(tiles):
            for bout_idx, bout in enumerate(session):
                if len(bout) > 1:
                    for tile_idx in range(len(bout) - 1, 1, -1):
                        tile = bout['discrete_loc'].iloc[tile_idx]
                        out_frame = bout["out_frame"].iloc[tile_idx]
                        prev_tile = bout['discrete_loc'].iloc[tile_idx - 1]
                        if tile == prev_tile:
                            if self.verbose:
                                print(f"Repeated tile in session {session_idx}, bout {bout_idx}, tile {tile_idx}",
                                      file=self.check_out)
                            if self.automated_fixes:
                                new_tiles = self._remove_tile(new_tiles, session_idx, bout_idx, tile_idx)
                                # update the out_frame of the previous tile
                                new_tiles = self._edit_coords(new_tiles, session_idx, bout_idx, tile_idx - 1,
                                                              new_vals={"out_frame": out_frame})
        return new_tiles if self.automated_fixes else tiles

    def _check_coords_valid(self, nn, bout_df_list):
        """
        Remove tiles whose position is impossible in the session's mask.

        For each session's mask, the valid columns and rows are taken from the
        mask holes; a tile whose column is not a valid column AND whose row is
        not a valid row is out of bounds and is dropped (scanning from the end
        to keep indices stable).

        Parameters
        ----------
        nn : str
            Experiment nickname (used to resolve the per-session mask).
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.

        Returns
        -------
        list of list of pandas.DataFrame
            Sessions with out-of-bounds tiles removed. Returned unchanged when
            ``automated_fixes`` is ``False``.
        """
        new_bout_list = deepcopy(bout_df_list)
        for session_idx, session in enumerate(bout_df_list):
            valid_rows, valid_cols = [], []
            mask_name = self._mask_labels[nn][session_idx]
            mask = self.masks[mask_name]
            for hole in mask.get_holes():
                col, row = hole
                valid_rows.append(row)
                valid_cols.append(col)
            for bout_idx, bout in enumerate(session):
                for tile_idx in range(len(bout) - 1, -1, -1):
                    tile = bout['discrete_loc'].iloc[tile_idx]
                    col, row = tile
                    if (col not in valid_cols) and (row not in valid_rows):
                        if self.verbose:
                            print(f"tile {tile_idx} in session {session_idx} bout {bout_idx}"
                                  f" is at position {tile}, which is out of bounds for Mask {mask.name}",
                                  file=self.check_out)
                        if self.automated_fixes:
                            new_bout_list = self._remove_tile(new_bout_list, session_idx, bout_idx, tile_idx)
        return new_bout_list if self.automated_fixes else bout_df_list

    def _validate_bouts_alternate(self, bout_df_list, error_list=None):
        """
        Validate that consecutive bouts share matching port endpoints.

        A bout starting at ``home_pos`` must follow a bout that ended at
        ``home_pos`` (and likewise for ``out_pos``); any mismatch is recorded in
        ``error_list`` for both the current and the previous bout. No bouts are
        modified.

        Parameters
        ----------
        bout_df_list : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        error_list : list, optional
            Mutable list to append endpoint-mismatch errors to. If ``None``, a
            fresh list is used.

        Returns
        -------
        list of list of pandas.DataFrame
            The input ``bout_df_list`` unchanged.

        Raises
        ------
        AssertionError
            If any bout is empty (empty bouts must have been removed earlier).
        """
        if error_list is None:
            error_list = []

        for s_idx, session in enumerate(bout_df_list):
            # Track the last non-empty bout to avoid nested while loops
            last_non_empty_bout = None
            last_bout_idx = -1

            for b_idx, bout in enumerate(session):
                # Skip empty bouts
                assert not bout.empty, "Empty bouts should have been removed in the previous check. Please check the order of checks or set min_length accordingly."
                # If we have a previous bout to compare against
                if last_non_empty_bout is not None:
                    first_tile = bout['discrete_loc'].iloc[0]
                    prev_last_tile = last_non_empty_bout['discrete_loc'].iloc[-1]

                    # Logical check: A bout should start where the previous one ended
                    # (Both at home OR both at out_pos)
                    is_mismatch = (
                            (first_tile == self.home_pos and prev_last_tile != self.home_pos) or
                            (first_tile == self.out_pos and prev_last_tile != self.out_pos)
                    )

                    if is_mismatch:
                        if self.verbose:
                            print(f"Endpoint Mismatch: Session {s_idx}, Bout {b_idx} starts at {first_tile}, "
                                  f"but previous Bout {last_bout_idx} ended at {prev_last_tile}",
                                  file=self.check_out)
                        # Record the errors
                        error_list.append({'session': s_idx, 'bout': b_idx, 'error_type': 'endpoints', 'coords':-1})
                        error_list.append(
                            {'session': s_idx, 'bout': last_bout_idx, 'error_type': 'prev bout (inspect)', 'coords':-1})

                # Update the tracker for the next iteration
                last_non_empty_bout = bout
                last_bout_idx = b_idx

        # Note: Since this is a validation function (not modifying data),
        # we simply return the original tiles.
        return bout_df_list

    def _apply_manual_bout_fixes(self, nn, tiles):
        """
        Apply the bout-level manual fixes for one experiment.

        Reads this experiment's fix list from ``manual_fixes`` and applies the
        bout-level operations: ``remove_bout`` first (sorted by
        ``(session, bout)`` DESCENDING so earlier removals do not shift later
        indices), then ``add_bout``/``merge_bouts``/``split_bouts`` in JSON list
        order.

        Parameters
        ----------
        nn : str
            Experiment nickname; if absent from ``manual_fixes`` the tiles are
            returned unchanged.
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.

        Returns
        -------
        list of list of pandas.DataFrame
            Tiles after bout-level manual fixes.

        Notes
        -----
        Bout indices in ``manual_fixes.json`` reference positions AFTER all
        prior fixes in the same list have been applied.
        """
        # manual_fixes is None when no fix file was supplied (see the constructor's
        # manual_fixes_path docstring) -- nothing to apply.
        if not self.manual_fixes or nn not in self.manual_fixes:
            return tiles

        exp_fixes = self.manual_fixes[nn]

        # Separate operations
        removals = [f for f in exp_fixes if f.get('type') == "remove_bout"]
        structural = [f for f in exp_fixes if f.get('type') in ["add_bout", "merge_bouts", "split_bouts"]]

        # 1. Remove bouts in REVERSE order
        for fix in sorted(removals, key=lambda x: (x['session'], x['bout']), reverse=True):
            print(f"Bout removal: session {fix['session']}, bout {fix['bout']}", file=self.manual_out)
            tiles = self._remove_bout(tiles, fix['session'], fix['bout'])

        # 2. Structural changes (Add, Merge, Split)
        # Note: These are applied in the order they appear in JSON, or you can sort them
        for fix in structural:
            f_type = fix['type']
            s, b = fix['session'], fix.get('bout')

            if f_type == "add_bout":
                tiles = self._add_bout(tiles, s, b)
            elif f_type == "merge_bouts":
                tiles = self._merge_bouts(tiles, s, fix['start_bout'], fix['end_bout'])
            elif f_type == "split_bouts":
                tiles = self._split_bouts(tiles, s, b, fix['index'])

        return tiles

    def _apply_manual_coord_fixes(self, nn, tiles):
        """
        Apply the tile-level manual fixes for one experiment.

        Reads this experiment's fix list from ``manual_fixes`` and applies the
        tile-level operations: ``remove_tile`` first (sorted by
        ``(session, bout, index)`` DESCENDING for index stability), then
        ``edit_tile`` (order-neutral), then ``add_tile``.

        Parameters
        ----------
        nn : str
            Experiment nickname; if absent from ``manual_fixes`` the tiles are
            returned unchanged.
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.

        Returns
        -------
        list of list of pandas.DataFrame
            Tiles after tile-level manual fixes.

        Notes
        -----
        Tile indices in ``manual_fixes.json`` reference positions AFTER all
        prior fixes in the same list have been applied; this method is invoked
        last in :meth:`_apply_fixes` so indices are relative to the fully
        automated-fixed bouts.
        """
        # manual_fixes is None when no fix file was supplied (see the constructor's
        # manual_fixes_path docstring) -- nothing to apply.
        if not self.manual_fixes or nn not in self.manual_fixes:
            return tiles

        exp_fixes = self.manual_fixes[nn]

        # Organize by type
        removals = [f for f in exp_fixes if f.get('type') == "remove_tile"]
        edits = [f for f in exp_fixes if f.get('type') == "edit_tile"]
        additions = [f for f in exp_fixes if f.get('type') == "add_tile"]

        # 1. Remove tiles in REVERSE order (crucial for index stability)
        for fix in sorted(removals, key=lambda x: (x['session'], x['bout'], x['index']), reverse=True):
            tiles = self._remove_tile(tiles, fix['session'], fix['bout'], fix['index'])

        # 2. Edits (Order neutral)
        for fix in edits:
            tiles = self._edit_coords(tiles, fix['session'], fix['bout'], fix['index'], fix['new_vals'])

        # 3. Additions
        for fix in additions:
            tiles = self._add_coords(tiles, fix['session'], fix['bout'], fix['index'], fix['new_vals'])

        return tiles

    def _remove_bout(self, tiles, session, bout):
        """
        Return a copy of ``tiles`` with one bout removed.

        Parameters
        ----------
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        session : int
            0-based session index.
        bout : int
            0-based bout index within the session to remove.

        Returns
        -------
        list of list of pandas.DataFrame
            Deep copy with the bout removed.
        """
        new_tiles = deepcopy(tiles)
        new_tiles[session].pop(bout)
        if self.verbose:
            print(f"bout {bout} in session {session} has been removed", file=self.manual_out)
        return new_tiles

    def _add_bout(self, tiles, session, bout):
        """
        Insert a new empty bout at a given session/position.

        Parameters
        ----------
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        session : int
            0-based session index.
        bout : int
            0-based position at which to insert the new bout.

        Returns
        -------
        list of list of pandas.DataFrame
            Deep copy with an empty ``bout_df`` (columns ``in_frame``,
            ``out_frame``, ``discrete_loc``) inserted at ``bout``.
        """
        new_tiles = deepcopy(tiles)
        new_tiles[session].insert(bout, pd.DataFrame(columns=["in_frame", "out_frame", "discrete_loc"]))
        if self.verbose:
            print(f"bout added in session {session} at position {bout}", file=self.manual_out)
        return new_tiles

    def _split_bouts(self, tiles, session, bout, index):
        """
        Split one bout into two at a row index.

        Rows ``[:index]`` become the first new bout and rows ``[index:]`` the
        second; the original bout is removed and the two parts inserted in its
        place.

        Parameters
        ----------
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        session : int
            0-based session index.
        bout : int
            0-based index of the bout to split.
        index : int
            0-based row index at which to split (first part is ``[:index]``).

        Returns
        -------
        list of list of pandas.DataFrame
            Deep copy with the bout replaced by its two halves.
        """
        new_tiles = deepcopy(tiles)
        bout_to_split = tiles[session][bout]
        new_bout1 = bout_to_split.iloc[:index].reset_index(drop=True)
        new_bout2 = bout_to_split.iloc[index:].reset_index(drop=True)
        # Remove the old bout
        new_tiles[session].pop(bout)
        if self.verbose:
            print(f"bout {bout} in session {session} has been removed for splitting into two bouts", file=self.manual_out)
        # Add the two new bouts
        new_tiles[session].insert(bout, new_bout2)
        new_tiles[session].insert(bout, new_bout1)
        return new_tiles


    def _merge_bouts(self, tiles, session, start_bout, end_bout):
        """
        Concatenate a contiguous range of bouts into a single bout.

        Bouts ``start_bout`` through ``end_bout`` (inclusive) are concatenated
        in order; the originals are removed (in reverse to avoid index shifts)
        and the merged bout inserted at ``start_bout``.

        Parameters
        ----------
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        session : int
            0-based session index.
        start_bout, end_bout : int
            0-based inclusive range of bout indices to merge
            (``end_bout > start_bout``).

        Returns
        -------
        list of list of pandas.DataFrame
            Deep copy with the range merged into one bout.
        """
        new_tiles = deepcopy(tiles)
        bout_indices = np.arange(start_bout, end_bout + 1)
        new_bout = pd.concat([tiles[session][bout_idx] for bout_idx in bout_indices])
        new_bout = new_bout.reset_index(drop=True)
        if self.verbose:
            print(f"bout {bout_indices} in session {session} has been merged", file=self.manual_out)
            print(new_bout, file=self.manual_out)
        # Remove all old bouts in reverse order to avoid index shifting
        for bout in reversed(list(bout_indices)):
            new_tiles[session].pop(bout)
            if self.verbose:
                print(f"bout {bout} in session {session} has been removed", file=self.manual_out)
        # Add new bout
        new_tiles[session].insert(bout_indices[0], new_bout)
        if self.verbose:
            print(f"New bout added in session {session} at position {bout_indices[0]}", file=self.manual_out)
        return new_tiles

    def _edit_coords(self, tiles, session, bout, index, new_vals):
        """
        Overwrite selected columns of one tile in a bout.

        Parameters
        ----------
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        session, bout, index : int
            0-based session, bout and tile (row) indices of the tile to edit.
        new_vals : dict
            Column-name to value mapping (e.g. ``in_frame``, ``out_frame``,
            ``discrete_loc``). A ``discrete_loc`` value is coerced to a
            ``(col, row)`` tuple.

        Returns
        -------
        list of list of pandas.DataFrame
            Deep copy with the tile's columns updated.
        """
        new_tiles = deepcopy(tiles)
        for col, val in new_vals.items():
            if col == 'discrete_loc':
                val = tuple(val)
            new_bout = new_tiles[session][bout][col].to_list()
            new_bout[index] = val
            new_tiles[session][bout][col] = pd.Series(new_bout)
        if self.verbose:
            print(f"session {session} bout {bout} tile {index} has been edited to {new_vals}", file=self.manual_out)
        return new_tiles

    def _remove_tile(self, tiles, session, bout, index):
        """
        Drop one tile (row) from a bout and reindex.

        Parameters
        ----------
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        session, bout, index : int
            0-based session, bout and tile (row) indices of the tile to remove.

        Returns
        -------
        list of list of pandas.DataFrame
            Deep copy with the tile removed and the bout index reset.
        """
        new_tiles = deepcopy(tiles)
        new_bout = new_tiles[session][bout].drop(index).reset_index(drop=True)
        new_tiles[session][bout] = new_bout
        if self.verbose:
            print(f"session {session} bout {bout} tile {index} has been removed", file=self.manual_out)
        return new_tiles

    def _add_coords(self, tiles, session, bout, index, new_vals):
        """
        Insert a new tile (row) into a bout at a given index.

        Each column of the new row is taken from ``new_vals`` when present and
        set to ``None`` otherwise; a ``discrete_loc`` value is coerced to a
        ``(col, row)`` tuple.

        Parameters
        ----------
        tiles : list of list of pandas.DataFrame
            Sessions, each a list of ``bout_df``.
        session, bout, index : int
            0-based session, bout and row index at which to insert.
        new_vals : dict
            Column-name to value mapping for the new row. Columns absent from
            ``new_vals`` are filled with ``None`` (which can introduce NaN
            frames; see data contract §3 failure modes).

        Returns
        -------
        list of list of pandas.DataFrame
            Deep copy with the new tile inserted.
        """
        new_tiles = deepcopy(tiles)
        new_bout_data = {col: new_tiles[session][bout][col].to_list()
                         for col in new_tiles[session][bout].columns}
        for col in new_bout_data.keys():
            val = None
            if col in new_vals:
                val = new_vals[col]
                if col == 'discrete_loc':
                    val = tuple(val)
            new_bout_data[col].insert(index, val)
        new_tiles[session][bout] = pd.DataFrame(new_bout_data)
        if self.verbose:
            print(f"tile added in bout {bout} in session {session} at position {index} with vals {new_vals}",
                  file=self.manual_out)
        return new_tiles

    def _get_reward_df(self, nn):
        """
        Load the reward DataFrame for one experiment.

        Globs ``rewards/{nn}*.csv`` and reads the alphabetically last match
        (latest by the date-sorted naming convention), dropping the artefact
        ``Unnamed: 0`` column.

        Parameters
        ----------
        nn : str
            Experiment nickname.

        Returns
        -------
        pandas.DataFrame
            ``reward_df`` with columns ``rwd_1``
            (int frame of the trigger LED) and ``rwd_2`` (float frame of the
            consumption LED, or NaN); frames are absolute video frames. An empty
            DataFrame is returned if no reward file is found.
        """
        reward_filepath = os.path.join(self.data_dir, self.reward_filename_template.format(
            nn=nn))
        file_names = glob(reward_filepath)
        if not file_names:
            print(f"Reward file for {nn} not found; please process raw LED info again.")
            return pd.DataFrame()
        else:
            # sort to get the latest
            file_names = sorted(file_names)
            reward_df = utils.drop_unnamed_column(pd.read_csv(file_names[-1]))
            return reward_df

    def _get_room_status(self, nn):
        """
        Load the verified room-status DataFrame for one experiment.

        Globs ``verified_room_status/{nn}*.csv`` and reads the alphabetically
        last match (latest by the date-sorted naming convention), dropping the
        artefact ``Unnamed: 0`` column.

        Parameters
        ----------
        nn : str
            Experiment nickname.

        Returns
        -------
        pandas.DataFrame
            ``room_status_df`` with ``Status``,
            ``Start_frame`` and ``End_frame`` columns; frames are absolute video
            frames.
        """
        room_status_filepath = os.path.join(self.data_dir, self.room_status_filename_template.format(
            nn=nn))
        file_names = glob(room_status_filepath)
        room_status = utils.drop_unnamed_column(pd.read_csv(file_names[-1]))
        return room_status

    def _get_mask_labels(self, nn):
        """
        Parse the ordered mask configuration sequence for one experiment.

        Parameters
        ----------
        nn : str
            Experiment nickname.

        Returns
        -------
        list of str
            Mask names in configuration order (one per session), e.g.
            ``['O', 'A']``.

        Notes
        -----
        The metadata ``Config_label_list`` field is parsed as
        ``field.strip('][').split(', ')`` — the space after the comma is
        significant; a missing space silently produces wrong mask names (data
        contract §7).
        """
        mask_labels = self.metadata.loc[self.metadata["Nickname"] == nn, "Config_label_list"].item()
        mask_labels = mask_labels.strip('][').split(', ')
        return mask_labels

    def _validate_rwd_df(self, nn):
        """
        Validate the reward DataFrame's structure and ordering.

        Checks that ``len(rwd_1) - len(rwd_2) <= 1`` (at most one trigger
        without a matching consumption, and never more consumptions than
        triggers) and that for every paired reward ``rwd_2[k] > rwd_1[k]``
        (consumption follows trigger). Frames are absolute video frames.

        Parameters
        ----------
        nn : str
            Experiment nickname.

        Returns
        -------
        str or None
            The nickname ``nn`` if the reward data is empty or fails a check
            (i.e. it is flagged as problematic), otherwise ``None``.
        """
        rwd_df = self._get_reward_df(nn)
        if rwd_df.empty:
            print(f"No reward files for {nn}")
            return nn
        # find the ones with column length difference > 1
        n_rwd1 = len(rwd_df.rwd_1)
        n_rwd2 = len(rwd_df.rwd_2)
        rwd1_time = rwd_df.rwd_1.to_numpy()
        rwd2_time = rwd_df.rwd_2.to_numpy()
        if n_rwd1 - n_rwd2 > 1 or n_rwd2 > n_rwd1:
            print(f"Reward numbers for {nn} are not correct: reward 1 {n_rwd1}, reward 2 {n_rwd2}")
            return nn
        # check if any reward 1 is earlier than reward 2:
        n_rwds = min(n_rwd1, n_rwd2)
        rwd_differences = rwd2_time[:n_rwds] - rwd1_time[:n_rwds]
        if np.any(rwd_differences < 0):
            print(f"{nn} reward time differences have problems")
            return nn
        else:
            return None

    def _get_session_timestamps(self, nn):
        """
        Build the per-session (mask label, frame range) list for segmentation.

        Matches each ``"Configuration"`` row in the room-status file positionally
        to the experiment's mask labels and pairs it with its
        ``(Start_frame, End_frame)`` range. Frames are absolute video frames.

        Parameters
        ----------
        nn : str
            Experiment nickname.

        Returns
        -------
        list of tuple
            One ``(mask_label, (start_frame, end_frame))`` per configuration, in
            session order.

        Raises
        ------
        ValueError
            If the number of mask labels does not equal the number of
            ``"Configuration"`` rows in the room-status file.
        """
        mask_labels = self._mask_labels[nn]
        room_status_df = self._room_status[nn]

        n_changes = len(mask_labels)
        config_df = room_status_df[room_status_df["Status"].str.contains("Configuration")].reset_index()
        if n_changes != len(config_df):
            print("Warning: check room status file. Do your configuration changes match with room status?")
            raise ValueError(f'{nn} # config label {n_changes} does not match room status # {len(config_df)}')

        config_frame_tuple_list = []  # use list instead
        for i, row in config_df.iterrows():
            config_frame_tuple_list.append((mask_labels[i], (row["Start_frame"], row["End_frame"])))
        return config_frame_tuple_list

    def _get_raw_trajectory_filename(self, nn):
        """
        Resolve the raw trajectory file path for one experiment.

        Globs the raw-trajectory template for ``nn`` and returns the
        alphabetically last match (newest by the date-sorted naming
        convention).

        Parameters
        ----------
        nn : str
            Experiment nickname.

        Returns
        -------
        str or None
            Path to the latest raw trajectory pickle, or ``None`` if no file
            matches.
        """
        trajectories = glob(os.path.join(self.data_dir, self.raw_traj_filename_template.format(nn=nn)))
        if not trajectories:
            print(f"No trajectory file found for {nn}")
            return None
        else:
            latest_trajectory_path = sorted(trajectories)[-1] # use the newest date
            return latest_trajectory_path

    def _load_mask(self, mask_name):
        """
        Construct the :class:`Mask` object for one mask configuration.

        Loads ``masks/holes_{name}.npy`` and builds the maze geometry/graph,
        propagating the loader's ``mask_size``, ``home_coordinates`` and
        ``out_coordinates``. Masks ``"D"`` and ``"D_flipped"`` use the
        :class:`MaskDSpecial` subclass to attach Mask-D special annotations;
        all others use :class:`Mask`.

        Parameters
        ----------
        mask_name : str
            Mask configuration name (e.g. ``"O"``, ``"A"``, ``"D"``).

        Returns
        -------
        Mask
            The mask object (schema in data contract §6).
        """
        mask_filename = self.mask_filename_template.format(name=mask_name)
        mask_filepath = os.path.join(self.data_dir, mask_filename)
        cls = MaskDSpecial if mask_name in ("D", "D_flipped") else Mask
        return cls(mask_filepath, self.mask_size, mask_name, self.home_coordinates, self.out_coordinates)
