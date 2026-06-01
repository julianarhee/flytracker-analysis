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
def load_jaaba_from_mat(mat_fpath, return_dict=False):
    """Load a JAABA `scores_*.mat` file.

    Adapted from `analyses/multichamber/src/multichamber_strains.py`.

    Returns:
        pd.DataFrame (frames x fly_ids) of raw scores, or the full allScores
        dict if `return_dict=True`. Returns None if scores can't be stacked.
    """
    mat = scipy.io.loadmat(mat_fpath)
    allscores_data = mat['allScores'][0][0]
    fields = ['scores', 'tStart', 'tEnd', 'postprocessed', 'postprocessparams',
              't0s', 't1s', 'scoreNorm']
    allscores = {}
    for k, v in zip(fields, allscores_data):
        if k in ['scores', 'postprocessed']:
            try:
                allscores[k] = np.vstack(v[0]).T
            except ValueError:
                print("Error stacking JAABA scores: {}".format(mat_fpath))
                return None
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
