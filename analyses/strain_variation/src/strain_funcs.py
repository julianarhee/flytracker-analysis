#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strain_funcs.py

Helper functions for the `strain_variation` analysis (Dmel and Dyak 2x2
multichamber courtship assays, Caitlin RA data).

These mirror the preprocessing style of `analyses/multichamber/src` but are
collected here and extended to handle the *multi-fly* (8-fly) FlyTracker
`-actions.mat` manual annotations. The arena-ID lookup, sex assignment, and
copulation-frame extraction are adapted from
`analyses/multichamber/src/multichamber.py`; the JAABA loaders are adapted from
`analyses/multichamber/src/multichamber_strains.py`. If more analyses end up
needing the multi-fly actions loader, promote it to `libs/utils.py`.

Layout assumed for one acquisition `<acq>` under a species directory:

    <acq>/
        calibration.mat
        movie.avi
        scores_chasing_*.mat              (JAABA, top level)
        scores_unilateral_extension*.mat  (JAABA, top level)
        <acq>/                            (nested FlyTracker output)
            <acq>-track.mat
            <acq>-feat.mat
            <acq>-actions.mat             (manual annotations)
"""
import os
import glob

import numpy as np
import pandas as pd
import scipy.io

import libs.utils as util


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
# Default root for the Caitlin RA data (update for your mount point).
ROOTDIR = '/Volumes/Juliana/Caitlin_RA_data'

# Species -> source subdirectory containing acquisition folders + metadata csv.
SPECIES_DIRS = {
    'Dmel': 'Caitlin_2x2_mel_strains',
    'Dyak': 'Caitlin_2x2_yak_strains',
}

# Manual-annotation behaviors of interest (as they appear in -actions.mat,
# before whitespace normalization). 'copulation ' has a trailing space in the
# FlyTracker files, and 'unilateral extension' contains a space.
COURTSHIP_BEHAVIORS = ['courtship', 'chasing', 'copulation', 'unilateral extension']

# JAABA score files to merge: behavior column name -> glob pattern (the
# chasing classifier is named per-species, so we glob loosely).
JAABA_SCORE_PATTERNS = {
    'chasing': 'scores_chasing*.mat',
    'unilateral_extension': 'scores_unilateral_extension*.mat',
}


def normalize_behavior_name(name):
    """Normalize a FlyTracker behavior name to a tidy column name.

    'copulation '          -> 'copulation'
    'unilateral extension' -> 'unilateral_extension'
    """
    return name.strip().replace(' ', '_').lower()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def load_strain_metadata(species_dir):
    """Load the per-species metadata csv (Google Sheets export).

    Args:
        species_dir (str): Path to the species source directory (contains the
            acquisition folders and a single `courtship_free_behavior*.csv`).

    Returns:
        pd.DataFrame: metadata with an `acquisition` column (== `file_name`).
    """
    meta_fpaths = glob.glob(os.path.join(species_dir, '*.csv'))
    assert len(meta_fpaths) == 1, \
        "Expected exactly 1 metadata csv in {}, found {}".format(species_dir, len(meta_fpaths))
    meta = pd.read_csv(meta_fpaths[0])
    # Drop fully empty rows (trailing blank rows are common in these exports).
    meta = meta.dropna(how='all').copy()
    if 'acquisition' not in meta.columns:
        meta['acquisition'] = meta['file_name']
    # fly_num is the arena pair index (1-indexed); coerce to int where present.
    meta = meta[meta['fly_num'].notna()].copy()
    meta['fly_num'] = meta['fly_num'].astype(int)
    return meta


# ---------------------------------------------------------------------------
# Arena / fly-ID assignment (2x2 multichamber)
# ---------------------------------------------------------------------------
def meta_flynum_to_ft_id(array_size='2x2'):
    """Map metadata `fly_num` (arena pair, counted top-bottom/left-right) to the
    FlyTracker *male* id (counted left-right/top-bottom).

    The female of each pair is the returned id + 1. Adapted from
    `analyses/multichamber/src/multichamber.py`.
    """
    if array_size == '2x2':
        return {1: 0, 2: 4, 3: 2, 4: 6}
    elif array_size == '3x3':
        return {1: 0, 2: 6, 3: 12, 4: 2, 5: 8, 6: 14, 7: 4, 8: 10, 9: 16}
    raise ValueError("array_size not recognized: {}".format(array_size))


def assign_sex(df):
    """Even FlyTracker ids are male ('m'), odd are female ('f')."""
    df['sex'] = 'f'
    df.loc[df['id'] % 2 == 0, 'sex'] = 'm'
    df['sex'] = df['sex'].astype('category')
    return df


def assign_frame_number(df):
    """Assign a 0-indexed frame number per fly id."""
    for i, df_ in df.groupby('id'):
        df.loc[df['id'] == i, 'frame'] = np.arange(len(df_))
    return df


def assign_strain_to_multichamber(df, meta, array_size='2x2'):
    """Assign `fly_pair` and `strain` (and male/female strain) by arena.

    Assumes even id = male, odd id = female. Adapted from
    `analyses/multichamber/src/multichamber.py:assign_strain_to_multichamber`.
    """
    id_lut = meta_flynum_to_ft_id(array_size=array_size)
    for i in meta['fly_num'].unique():
        male_id = id_lut[i]
        df.loc[df['id'] == male_id, 'fly_pair'] = i        # male
        df.loc[df['id'] == male_id + 1, 'fly_pair'] = i    # female
        row = meta[meta['fly_num'] == i].iloc[0]
        df.loc[df['fly_pair'] == i, 'strain'] = row['strain_male']
        if 'strain_male' in meta.columns:
            df.loc[df['fly_pair'] == i, 'strain_male'] = row['strain_male']
        if 'strain_female' in meta.columns:
            df.loc[df['fly_pair'] == i, 'strain_female'] = row['strain_female']
        if 'species_male' in meta.columns:
            df.loc[df['fly_pair'] == i, 'species'] = row['species_male']
    return df


# ---------------------------------------------------------------------------
# Manual annotations (-actions.mat), multi-fly aware
# ---------------------------------------------------------------------------
def find_actions_mat(acqdir):
    """Locate the nested `<acq>-actions.mat` for an acquisition directory."""
    matches = glob.glob(os.path.join(acqdir, '*', '*-actions.mat'))
    return matches[0] if matches else None


def ft_actions_to_bout_df(action_fpath, behaviors=None):
    """Parse a FlyTracker `-actions.mat` into a tidy bout DataFrame for ALL flies.

    Unlike `libs.utils.ft_actions_to_bout_df` (which only reads fly id 0), this
    handles the multichamber case where `mat['bouts']` is (n_flies, n_behaviors)
    and manual annotations are assigned to each *male* fly id (0, 2, 4, 6 ...).

    Args:
        action_fpath (str): full path to `<acq>-actions.mat`.
        behaviors (list or None): raw behavior names to keep (default:
            `COURTSHIP_BEHAVIORS`). Pass [] / 'all' to keep every behavior.

    Returns:
        pd.DataFrame with columns:
            id        - FlyTracker fly id the bout is annotated on
            action    - normalized behavior name
            start,end - bout start/end frame (inclusive)
            likelihood- annotation likelihood flag from FlyTracker
            boutnum   - unique bout index within the file
    """
    if behaviors is None:
        behaviors = COURTSHIP_BEHAVIORS
    keep_all = behaviors == 'all' or behaviors == []
    keep_norm = None if keep_all else {normalize_behavior_name(b) for b in behaviors}

    mat = scipy.io.loadmat(action_fpath)
    beh_names = [v[0][0] for v in mat['behs']]
    bouts = mat['bouts']  # (n_flies, n_behaviors), each cell is (n_bouts, 3)
    n_flies = bouts.shape[0]

    rows = []
    for fly_id in range(n_flies):
        for j, beh in enumerate(beh_names):
            norm = normalize_behavior_name(beh)
            if keep_norm is not None and norm not in keep_norm:
                continue
            cell = np.asarray(bouts[fly_id, j])
            if cell.size == 0 or cell.ndim != 2 or cell.shape[1] != 3:
                continue
            b_df = pd.DataFrame(cell, columns=['start', 'end', 'likelihood'])
            b_df['action'] = norm
            b_df['id'] = fly_id
            rows.append(b_df)

    if not rows:
        return pd.DataFrame(columns=['id', 'action', 'start', 'end', 'likelihood', 'boutnum'])

    boutdf = pd.concat(rows, ignore_index=True)
    boutdf[['start', 'end']] = boutdf[['start', 'end']].astype(int)
    boutdf['boutnum'] = np.arange(len(boutdf))
    return boutdf


def assign_action_frames_to_df(df, actions):
    """Mark annotated behavior frames on the tracking DataFrame, per fly id.

    Each behavior becomes a binary 0/1 column (1 within a bout) plus a
    `<behavior>_boutnum` column. Annotations are matched on BOTH `id` and
    `frame`, so a bout annotated on the male id only flags the male's rows.

    Args:
        df (pd.DataFrame): tracking df with `id` and `frame` columns.
        actions (pd.DataFrame): output of `ft_actions_to_bout_df`.

    Returns:
        pd.DataFrame: df with added behavior columns.
    """
    for action_name in actions['action'].unique():
        if action_name not in df.columns:
            df[action_name] = 0
            df['{}_boutnum'.format(action_name)] = np.nan

    for _, bout in actions.iterrows():
        fly_id = bout['id']
        frame_range = np.arange(bout['start'], bout['end'] + 1)
        mask = (df['id'] == fly_id) & (df['frame'].isin(frame_range))
        df.loc[mask, bout['action']] = 1
        df.loc[mask, '{}_boutnum'.format(bout['action'])] = bout['boutnum']
    return df


def get_copulation_frames(action_fpath):
    """Return {fly_id: copulation_onset_frame} (-1 if none), with the female of
    each pair inheriting the male's copulation frame.

    Adapted from `analyses/multichamber/src/multichamber.py:get_copulation_frames`.
    """
    mat = scipy.io.loadmat(action_fpath)
    beh_names = [v[0][0] for v in mat['behs']]
    cop_ix = beh_names.index('copulation ') if 'copulation ' in beh_names \
        else beh_names.index('copulation')

    cop_starts = []
    for v in mat['bouts']:
        cell = np.asarray(v[cop_ix])
        cop_starts.append(int(cell[0][0]) if cell.size > 0 else -1)
    cop_dict = {k: v for k, v in enumerate(cop_starts)}

    # Propagate copulation frame to the paired fly (male<->female).
    for k in [k for k, v in cop_dict.items() if v != -1]:
        partner = k - 1 if k % 2 > 0 else k + 1
        cop_dict[partner] = cop_dict[k]
    return cop_dict


# ---------------------------------------------------------------------------
# JAABA automated scores
# ---------------------------------------------------------------------------
def _stack_ragged_scores(per_fly, tstart):
    """Stack per-fly JAABA score arrays into a (frames, n_flies) array.

    Per-fly arrays can have *different* lengths: a fly whose track ends early
    (tracking lost) has a shorter score vector. Each fly's scores cover frames
    `[tStart_i - 1, tStart_i - 1 + len_i)` (tStart is 1-indexed); we place them
    at that offset in a NaN-padded common-length array. When all flies share the
    same length and tStart==1 this reduces to the old `np.vstack(...).T`.
    """
    arrs = [np.asarray(f, dtype=float).ravel() for f in per_fly]
    tstart = np.asarray(tstart).ravel().astype(int)
    starts = [int(tstart[i]) - 1 if i < len(tstart) else 0
              for i in range(len(arrs))]
    max_len = max((s + a.shape[0] for s, a in zip(starts, arrs)), default=0)
    out = np.full((len(arrs), max_len), np.nan)
    for i, (s, a) in enumerate(zip(starts, arrs)):
        out[i, s:s + a.shape[0]] = a
    return out.T


def load_jaaba_from_mat(mat_fpath, return_dict=False):
    """Load a JAABA `scores_*.mat` file.

    Adapted from `analyses/multichamber/src/multichamber_strains.py`, but
    tolerant of *ragged* per-fly score arrays (flies whose tracks end early have
    shorter score vectors) — these are NaN-padded to a common length aligned by
    `tStart`, rather than failing the old equal-length `np.vstack`.

    Returns:
        pd.DataFrame (frames x fly_ids) of raw scores, or the full allScores
        dict if `return_dict=True`.
    """
    mat = scipy.io.loadmat(mat_fpath)
    allscores_data = mat['allScores'][0][0]
    fields = ['scores', 'tStart', 'tEnd', 'postprocessed', 'postprocessparams',
              't0s', 't1s', 'scoreNorm']
    raw = dict(zip(fields, allscores_data))
    tstart = raw['tStart']  # 1-indexed per-fly start frame; raveled in helper
    allscores = {}
    for k in fields:
        v = raw[k]
        if k in ('scores', 'postprocessed'):
            allscores[k] = _stack_ragged_scores(v[0], tstart)
        else:
            allscores[k] = v[0]
    if return_dict:
        return allscores
    return pd.DataFrame(allscores['scores'],
                        columns=range(allscores['scores'].shape[1]))


def stack_jaaba_scores(scores_df):
    """Wide (frames x fly_ids) -> long (frame, id, score)."""
    stacked = scores_df.stack().reset_index()
    stacked.columns = ['frame', 'id', 'score']
    return stacked


def find_jaaba_score_file(acqdir, beh_type):
    """Locate the JAABA score .mat for a behavior at the acquisition top level."""
    pattern = JAABA_SCORE_PATTERNS.get(beh_type, 'scores_{}*.mat'.format(beh_type))
    matches = sorted(glob.glob(os.path.join(acqdir, pattern)))
    return matches[-1] if matches else None


def add_jaaba_scores(df, acqdir, beh_types=('chasing', 'unilateral_extension'),
                     thresholds=None):
    """Merge JAABA scores into the tracking df as `jaaba_<beh>` columns.

    Adds `jaaba_<beh>` (raw score) and `jaaba_<beh>_binary` columns, merged on
    (frame, id). Behaviors with no score file are skipped.

    Args:
        df (pd.DataFrame): tracking df with `frame`, `id`.
        acqdir (str): acquisition directory (where scores_*.mat live).
        beh_types (iterable): JAABA behaviors to merge.
        thresholds (dict): per-behavior binarization threshold on the raw score.

    Returns:
        (pd.DataFrame, list): df with merged scores, list of missing behaviors.
    """
    if thresholds is None:
        thresholds = {'chasing': 0.0, 'unilateral_extension': 0.0}
    missing = []
    for beh in beh_types:
        mat_fpath = find_jaaba_score_file(acqdir, beh)
        if mat_fpath is None:
            missing.append(beh)
            continue
        scores = load_jaaba_from_mat(mat_fpath)
        if scores is None:
            missing.append(beh)
            continue
        col = 'jaaba_{}'.format(beh)
        stacked = stack_jaaba_scores(scores).rename(columns={'score': col})
        if col in df.columns:
            df = df.drop(columns=[col])
        df = df.merge(stacked, how='left', on=['frame', 'id'])
        df['{}_binary'.format(col)] = df[col].ge(thresholds.get(beh, 0.0))
    return df, missing


# ---------------------------------------------------------------------------
# Derived courtship labels (analysis-time, tunable)
# ---------------------------------------------------------------------------
# Distinct from the *imported* labels above (manual -actions via
# assign_action_frames_to_df, JAABA via add_jaaba_scores): these are derived
# from kinematics/geometry at analysis time, with tunable thresholds, so they
# can be re-computed off the processed parquet without re-running the pipeline.
#
# derive_courtship_labels() writes SOURCE-AGNOSTIC canonical columns
# `is_<behavior>` (0/1) so downstream metrics never need to know whether a label
# came from JAABA, manual annotation, or a kinematic gate. 'singing' == the
# unilateral wing-extension behavior.
CANONICAL_BEHAVIORS = ['orienting', 'chasing', 'singing']

# Which processed-parquet column backs each canonical behavior, per source.
# ('orienting' is always derived from facing_angle and has no source column.)
_LABEL_SOURCE_COLUMNS = {
    'jaaba': {
        'chasing': 'jaaba_chasing_binary',
        'singing': 'jaaba_unilateral_extension_binary',
    },
    'manual': {
        'chasing': 'chasing',                 # manual -actions binary column
        'singing': 'unilateral_extension',    # manual -actions binary column
    },
}


# ---------------------------------------------------------------------------
# Subset / dataset-discovery helpers
# ---------------------------------------------------------------------------
def list_acquisitions_by_strain(rootdir, species_list=None):
    """Map every available acquisition to its strain for each species.

    Reads the per-species metadata CSV (from `load_strain_metadata`) and
    cross-references it against the acquisition directories that actually exist
    on disk.  Acquisitions in the metadata but absent from disk are silently
    skipped (they will be logged when processing is attempted).

    Args:
        rootdir (str): data root (same value as `ROOTDIR`).
        species_list (list or None): which species to query; defaults to all in
            `SPECIES_DIRS` ({'Dmel', 'Dyak'}).

    Returns:
        dict: ``{species: {strain: [acq_name, ...]}}`` sorted alphabetically.
    """
    if species_list is None:
        species_list = list(SPECIES_DIRS.keys())
    result = {}
    for sp in species_list:
        species_dir = os.path.join(rootdir, SPECIES_DIRS[sp])
        if not os.path.isdir(species_dir):
            continue
        try:
            meta = load_strain_metadata(species_dir)
        except (AssertionError, FileNotFoundError):
            continue
        # Acquisitions on disk
        on_disk = set(
            f for f in os.listdir(species_dir)
            if f.startswith('20') and os.path.isdir(os.path.join(species_dir, f))
        )
        strain_map = {}
        for acq in sorted(meta['acquisition'].unique()):
            if acq not in on_disk:
                continue
            strain_rows = meta[meta['acquisition'] == acq]['strain_male']
            if strain_rows.empty or strain_rows.dropna().empty:
                continue
            strain = str(strain_rows.dropna().iloc[0])
            strain_map.setdefault(strain, []).append(acq)
        result[sp] = {k: sorted(v) for k, v in sorted(strain_map.items())}
    return result


def select_subset_acquisitions(rootdir, n_strains_per_species=4,
                               n_acqs_per_strain=4, species_list=None,
                               strains_to_exclude=None, seed=None):
    """Auto-select a reproducible subset of acquisitions for pilot analysis.

    Chooses the `n_strains_per_species` strains with the most available
    acquisitions (ties broken alphabetically), then picks the first
    `n_acqs_per_strain` acquisition folders from each strain (sorted
    alphabetically → fully reproducible without a random seed).

    Pass `seed` to shuffle within each strain before picking (useful for
    leaving-out a hold-out set).

    Args:
        rootdir (str): data root.
        n_strains_per_species (int): strains to include per species.
        n_acqs_per_strain (int): acquisitions to include per strain.
        species_list (list or None): species to query; defaults to all.
        strains_to_exclude (set or None): strain names to skip.
        seed (int or None): shuffle seed; None = alphabetical order.

    Returns:
        dict: ``{species: {strain: [acq_name, ...]}}`` — the subset selection.
    """
    available = list_acquisitions_by_strain(rootdir, species_list=species_list)
    exclude = set(strains_to_exclude or [])
    subset = {}
    rng = np.random.default_rng(seed) if seed is not None else None

    for sp, strain_map in available.items():
        # Strains with enough acquisitions, sorted by count desc then name asc
        eligible = {s: acqs for s, acqs in strain_map.items()
                    if s not in exclude and len(acqs) > 0}
        chosen_strains = sorted(
            eligible.keys(),
            key=lambda s: (-len(eligible[s]), s)
        )[:n_strains_per_species]

        sp_subset = {}
        for strain in chosen_strains:
            acqs = list(eligible[strain])
            if rng is not None:
                rng.shuffle(acqs)
            sp_subset[strain] = acqs[:n_acqs_per_strain]
        subset[sp] = sp_subset
    return subset


def print_subset_summary(subset):
    """Pretty-print a subset dict to stdout for easy inspection."""
    for sp, strain_map in sorted(subset.items()):
        print('\n{} ({} strains):'.format(sp, len(strain_map)))
        for strain, acqs in sorted(strain_map.items()):
            print('  {:20s}  {} acqs'.format(strain, len(acqs)))
            for a in acqs:
                print('    {}'.format(a))


def filter_chasing(df, use_jaaba=True, beh_type='jaaba_chasing_binary',
                   min_vel=10, max_facing_angle=np.deg2rad(90),
                   max_dist_to_other=20,
                   max_targ_pos_theta=np.deg2rad(270),
                   min_targ_pos_theta=np.deg2rad(-270),
                   min_wing_ang=np.deg2rad(45)):
    """Select male frames of a chasing/singing-like behavior.

    Two modes (ported from
    `analyses/multichamber/src/multichamber_strains.py:filter_chasing`):

    - `use_jaaba=True`: rows where the JAABA/manual binary column `beh_type` is
      set (plus a facing-angle gate, and a wing-angle gate for singing).
    - `use_jaaba=False`: a purely kinematic gate on velocity, facing angle,
      target position angle, wing angle and interfly distance.

    Returns the filtered sub-DataFrame (the caller maps `.index` back to a
    binary column).
    """
    if use_jaaba:
        sel = (df[beh_type] == True) & (df['sex'] == 'm') \
            & (df['facing_angle'] <= max_facing_angle)
        if 'singing' in beh_type or 'unilateral' in beh_type:
            sel = sel & (df['max_wing_ang'] >= min_wing_ang)
        return df[sel].copy()

    sel = (df['sex'] == 'm') \
        & (df['vel'] >= min_vel) \
        & (df['targ_pos_theta'] <= max_targ_pos_theta) \
        & (df['targ_pos_theta'] >= min_targ_pos_theta) \
        & (df['facing_angle'] <= max_facing_angle) \
        & (df['max_wing_ang'] >= min_wing_ang) \
        & (df['dist_to_other'] <= max_dist_to_other)
    return df[sel].copy()


def derive_courtship_labels(df, source='jaaba', orienting_angle_deg=10,
                            chasing_kws=None, singing_kws=None):
    """Add canonical per-frame courtship labels to a processed parquet df.

    Adds integer 0/1 columns `is_orienting`, `is_chasing`, `is_singing`, a
    `courtship_sum` (their sum) and `is_courting` (sum > 0). Operates on a copy.

    Args:
        df (pd.DataFrame): processed-parquet df (one or many acquisitions).
        source (str): where chasing/singing come from:
            - 'jaaba'    : JAABA binary columns (`jaaba_*_binary`)
            - 'manual'   : manual -actions binary columns
            - 'kinematic': recomputed from kinematics via `filter_chasing`
        orienting_angle_deg (float): facing-angle threshold (deg) for orienting.
        chasing_kws, singing_kws (dict): only used when source='kinematic';
            passed to `filter_chasing(use_jaaba=False, ...)`.

    Returns:
        pd.DataFrame: copy of df with the canonical label columns added.
    """
    df = df.copy()

    # Orienting is always geometric (facing the target), regardless of source.
    df['is_orienting'] = (df['facing_angle'] <= np.deg2rad(orienting_angle_deg)).astype(int)

    if source in ('jaaba', 'manual'):
        colmap = _LABEL_SOURCE_COLUMNS[source]
        for beh, col in colmap.items():
            if col not in df.columns:
                raise KeyError(
                    "source='{}' needs column '{}' for '{}'; available label "
                    "columns: {}".format(source, col, beh,
                        [c for c in df.columns if 'chasing' in c or 'singing' in c
                         or 'unilateral' in c]))
            df['is_{}'.format(beh)] = df[col].fillna(False).astype(int)
    elif source == 'kinematic':
        # Chasing: target ahead within 60deg, moving, close; NO wing gate
        # (chasing doesn't require wing extension). Singing: wing extended,
        # target ahead, any speed, slightly wider distance.
        chasing_kws = chasing_kws or dict(min_vel=10, max_facing_angle=np.deg2rad(60),
                                          min_wing_ang=0, max_dist_to_other=20)
        singing_kws = singing_kws or dict(min_vel=0, min_wing_ang=np.deg2rad(30),
                                          max_dist_to_other=35,
                                          max_facing_angle=np.deg2rad(90))
        chasedf = filter_chasing(df, use_jaaba=False, **chasing_kws)
        singdf = filter_chasing(df, use_jaaba=False, **singing_kws)
        df['is_chasing'] = 0
        df.loc[chasedf.index, 'is_chasing'] = 1
        df['is_singing'] = 0
        df.loc[singdf.index, 'is_singing'] = 1
    else:
        raise ValueError("source must be 'jaaba', 'manual' or 'kinematic', "
                         "got {!r}".format(source))

    df['courtship_sum'] = df[['is_{}'.format(b) for b in CANONICAL_BEHAVIORS]].sum(axis=1)
    df['is_courting'] = (df['courtship_sum'] > 0).astype(int)
    return df
