#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared data-loading and LED/speed block parsing utilities for the
calibration (P1 levels) dataset.

Consolidates block-parsing logic from:
  - analyses/calibration/src/compare_led_and_speed.py
  - analyses/calibration/src/test_cw_ccw.py

NOTE: In this dataset, the stimulus is always CCW, so theta<0 = progressive
"""
#%%
import os
import glob

import numpy as np
import pandas as pd

import libs.utils as util
import transform_data.relative_metrics as rel
from analyses.gain.src import gain_funcs as gf


# ============================================================
# Shared configuration and paths
# ============================================================
DEFAULT_ROOTDIR = '/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector'
DEFAULT_PROCESSEDMAT_DIR = (
    '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_calibrated'
    '/FlyTracker/processed_mats_6led'
)
DEFAULT_MIN_COURTSHIP_FRAC = 0.2

PLOT_STYLE = 'dark'
MIN_FONTSIZE = 12
SPECIES_PALETTE = {'Dmel': 'plum', 'Dyak': 'mediumseagreen'}


# ============================================================
# Block-parsing helpers
# ============================================================

def infer_led_blocks(led_onset, n_LED_blocks, led_block_nframes, curr_leds, total_nframes):
    """
    Infer LED block start/end frames from the LED onset frame.

    Level 0 starts at max(0, led_onset - led_block_nframes) and ends at led_onset - 1.
    Subsequent levels start at led_onset + (i-1)*led_block_nframes.

    Parameters
    ----------
    led_onset : int
        Frame number of LED onset (transition from level 0 to level 1).
    n_LED_blocks : int
        Total number of LED blocks (including the level-0 baseline).
    led_block_nframes : int
        Number of frames per LED block.
    curr_leds : list of int/float
        LED intensities for levels 1..n_LED_blocks-1 (length = n_LED_blocks - 1).
    total_nframes : int
        Total number of frames in the session.

    Returns
    -------
    pd.DataFrame
        Columns: led_level, led_start, led_end, led_intensity.
    """
    led0_start = max(0, int(led_onset) - led_block_nframes)
    led_starts = [led0_start]
    for i in range(1, n_LED_blocks):
        led_starts.append(int(led_onset) + (i - 1) * led_block_nframes)
    led_starts = np.array(led_starts, dtype=int)

    led_ends = led_starts + led_block_nframes - 1
    led_ends[0] = int(led_onset) - 1
    led_ends[-1] = total_nframes

    led_blocks = pd.DataFrame({
        'led_level': np.arange(n_LED_blocks, dtype=int),
        'led_start': led_starts,
        'led_end': led_ends,
    })

    led_blocks.loc[led_blocks['led_level'] == 0, 'led_intensity'] = 0
    for lvl in range(1, n_LED_blocks):
        led_blocks.loc[led_blocks['led_level'] == lvl, 'led_intensity'] = curr_leds[lvl - 1]

    return led_blocks


def build_speed_blocks(led_blocks, speed_onset_df, n_speed_blocks,
                       speed_block_nframes, total_nframes, curr_speeds):
    """
    Build per-speed-epoch DataFrame using annotated speed-block onset events.

    Expects exactly n_LED_blocks * n_speed_blocks - 1 onset events (first speed
    block in LED level 0 starts at frame 0 without an explicit onset).

    Parameters
    ----------
    led_blocks : pd.DataFrame
        Output of infer_led_blocks.
    speed_onset_df : pd.DataFrame
        Sorted DataFrame with a 'start' column for each speed-block onset event.
    n_speed_blocks : int
        Number of speed epochs per LED block.
    speed_block_nframes : int
        Expected number of frames per speed epoch.
    total_nframes : int
        Total session frames.
    curr_speeds : list
        Speed values (Hz) for each speed level 0..n_speed_blocks-1.

    Returns
    -------
    pd.DataFrame
        One row per speed epoch with columns: led_level, led_intensity,
        speed_level, speed_hz, start, end.
    """
    speed_onsets = speed_onset_df.sort_values('start').reset_index(drop=True)

    if max(led_blocks['led_intensity']) < 50:
        led_type = 'low_led'
    else:
        led_type = 'full_led'

    if max(curr_speeds) < 80:
        speed_type = 'slow_speed'
    else:
        speed_type = 'standard_speed'

    speed_rows = []
    last_ix = 0
    for li, led_row in led_blocks.iterrows():
        led_level = int(led_row['led_level'])
        led_end = int(led_row['led_end'])
        led_intensity = int(led_row['led_intensity'])

        if li == 0:
            indices = np.arange(last_ix, last_ix + n_speed_blocks - 1)
            curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
            starts = np.concatenate([[0], curr_onsets]).astype(int)
            ends = np.concatenate([starts[1:] - 1, [led_end]])
        else:
            indices = np.arange(last_ix, last_ix + n_speed_blocks)
            curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
            starts = curr_onsets
            ends = np.concatenate([starts[1:] - 1, [led_end]])

        last_ix = indices[-1] + 1

        for s_level in range(n_speed_blocks):
            speed_rows.append({
                'led_level': led_level,
                'led_intensity': led_intensity,
                'led_type': led_type,
                'speed_level': s_level,
                'speed_hz': curr_speeds[s_level],
                'speed_type': speed_type,
                'start': int(starts[s_level]),
                'end': int(ends[s_level]),
            })

    return pd.DataFrame(speed_rows)


def assign_frames_to_blocks(df, speed_blocks):
    """
    Assign LED level, intensity, speed level, and speed Hz to each frame in df.

    Uses vectorized interval lookup based on the frame column.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'frame' column.
    speed_blocks : pd.DataFrame
        Output of build_speed_blocks.

    Returns
    -------
    pd.DataFrame
        Input df with added columns: led_level, led_intensity, led_type,
        speed_level, speed_hz, speed_type.
    """
    df = df.copy()
    df['led_level'] = np.nan
    df['led_intensity'] = np.nan
    df['led_type'] = None
    df['speed_level'] = np.nan
    df['speed_hz'] = np.nan
    df['speed_type'] = None

    for _, blk in speed_blocks.iterrows():
        mask = (df['frame'] >= blk['start']) & (df['frame'] <= blk['end'])
        df.loc[mask, 'led_level'] = blk['led_level']
        df.loc[mask, 'led_intensity'] = blk['led_intensity']
        df.loc[mask, 'led_type'] = blk['led_type']
        df.loc[mask, 'speed_level'] = blk['speed_level']
        df.loc[mask, 'speed_hz'] = blk['speed_hz']
        df.loc[mask, 'speed_type'] = blk['speed_type']

    return df


# ============================================================
# High-level data loader
# ============================================================

def load_all_calibration_data(rootdir, meta, processedmat_dir,
                              n_LED_blocks=6, n_speed_blocks=5,
                              block_dur_sec=20, fps=60,
                              create_new=False):
    """
    Load and label all calibration acquisitions with LED/speed block info.

    For each file in meta, loads the processed tracking DataFrame, parses
    LED and speed blocks from the actions file, and labels every frame.

    Results are cached to a single parquet file in processedmat_dir so
    subsequent calls load instantly (unless create_new=True).

    Parameters
    ----------
    rootdir : str
        Root directory containing raw video/actions data.
    meta : pd.DataFrame
        Metadata filtered to calibration trials. Must contain columns:
        file_name, led_onset_frame, intensity_light, speeds, species_male,
        age_male, days_on_retinol.
    processedmat_dir : str
        Directory containing processed parquet files.
    n_LED_blocks : int
    n_speed_blocks : int
    block_dur_sec : int
    fps : int
    create_new : bool
        If True, reprocess all files even if cache exists.

    Returns
    -------
    df_all : pd.DataFrame
        Concatenated per-frame data for all files with block labels.
    errors : list of (str, Exception)
        Files that failed to load/parse.
    """
    cache_path = os.path.join(processedmat_dir, '_calibration_df_all.parquet')

    if not create_new and os.path.exists(cache_path):
        print(f"Loading cached calibration data from: {cache_path}")
        df_all = pd.read_parquet(cache_path)
        print(f"  {len(df_all)} rows, "
              f"{df_all['acquisition'].nunique()} acquisitions")
        # Ensure velocity-based pr_direction is present (may be missing
        # in caches generated before this column was added).
        if ('targ_pos_theta' in df_all.columns
                and 'targ_ang_vel' in df_all.columns
                and ('pr_direction' not in df_all.columns
                     or df_all['pr_direction'].isna().all())):
            print("  Computing pr_direction from target velocity...")
            df_all = gf.assign_progressive_regressive_from_velocity(df_all)
        return df_all, []

    speed_block_nframes = int(round(block_dur_sec * fps))
    led_block_nframes = int(round(n_speed_blocks * speed_block_nframes))
    total_nframes = fps * n_LED_blocks * n_speed_blocks * block_dur_sec

    df_list = []
    errors = []

    for i, fn in enumerate(meta['file_name'].unique()):
        if i % 5 == 0:
            print(f'Processing {i}/{len(meta["file_name"].unique())}: {fn}')

        try:
            # --- Load processed tracking data (generate if not yet cached) ---
            df_ = rel.load_processed_df(processedmat_dir, acq=fn, create_new=False)
            if df_ is None:
                print(f"  No parquet found for {fn}, processing from raw FlyTracker data...")
                acq_dir = os.path.join(rootdir, fn)
                df_ = rel.get_metrics_relative_to_focal_fly(
                    acq_dir,
                    savedir=processedmat_dir,
                    save=True,
                    movie_fmt='.avi',
                    flyid1=0, flyid2=1,
                    plot_checks=False,
                    create_new=True,
                    get_relative_sizes=False,
                )
            if df_ is None:
                raise FileNotFoundError(f"Could not load or generate processed df for {fn}")

            # --- Load actions and extract speed onsets ---
            found_actions = glob.glob(os.path.join(rootdir, fn, '*', '*-actions.mat'))
            if len(found_actions) == 0:
                found_actions = glob.glob(os.path.join(rootdir, fn, f'{fn}-actions.mat'))
            assert len(found_actions) >= 1, f'No actions file for {fn}'
            actions_df = util.load_ft_actions(found_actions[:1], split_end=False)

            # Assign courtship labels to frames
            df_ = util.assign_action_frames_to_df(df_, actions_df)

            # Speed onset events: 'IR LED on' is a placeholder label in the
            # FlyTracker actions file that actually marks speed block transitions.
            speed_onset_df = actions_df[actions_df['action'] == 'IR LED on'].copy()
            speed_onset_df = speed_onset_df.sort_values(by='start')

            # --- Parse LED blocks ---
            led_onset = meta[meta['file_name'] == fn]['led_onset_frame'].values[0]
            curr_leds = [int(l) for l in str(
                meta[meta['file_name'] == fn]['intensity_light'].values[0]).split(',')]
            led_blocks = infer_led_blocks(
                led_onset, n_LED_blocks, led_block_nframes, curr_leds, total_nframes)

            # --- Parse speed blocks ---
            curr_speeds = [int(s) for s in str(
                meta[meta['file_name'] == fn]['speeds'].values[0]).split(',')]
            speed_blocks = build_speed_blocks(
                led_blocks, speed_onset_df, n_speed_blocks,
                speed_block_nframes, total_nframes, curr_speeds)

            # --- Assign block labels to every frame ---
            df_ = assign_frames_to_blocks(df_, speed_blocks)

            # --- Add metadata columns ---
            df_['file_name'] = fn
            species = 'Dyak' if 'yak' in fn.lower() else 'Dmel'
            df_['species'] = species

            currm = meta[meta['file_name'] == fn].iloc[0]
            age = currm['age_male']
            atr = currm['days_on_retinol']
            df_['age'] = age
            df_['ATR'] = atr
            df_['age_ATR'] = f'{age}-{atr}'

            # Store the raw condition strings so we can filter later
            df_['speeds_condition'] = str(currm['speeds'])
            df_['led_condition'] = str(currm['intensity_light'])

            # Species-specific metadata from the CSV
            if 'species_male' in meta.columns:
                df_['species'] = currm['species_male'] if pd.notna(currm['species_male']) else species
            if 'genotype_male' in meta.columns:
                df_['genotype'] = currm['genotype_male']

            # Acquisition identifier
            date_fly = '_'.join([fn.split('-')[0], fn.split('_')[1]])
            df_['acquisition'] = f'{date_fly}_{df_["species"].iloc[0]}'

            # Assign stim direction if available
            if 'stim_direction' in meta.columns:
                stim_dir = currm['stim_direction']
                if pd.notna(stim_dir):
                    df_['stim_direction'] = stim_dir

            # Assign progressive/regressive from instantaneous target
            # velocity relative to the male's midline.  This works
            # regardless of whether CW, CCW, or both directions are used.
            if 'targ_pos_theta' in df_.columns and 'targ_ang_vel' in df_.columns:
                df_ = gf.assign_progressive_regressive_from_velocity(df_)

            df_list.append(df_)

        except Exception as e:
            print(f"  Error processing {fn}: {e}")
            errors.append((fn, e))
            continue

    if len(df_list) == 0:
        raise ValueError("No files loaded successfully")

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df_list)} files, {len(errors)} errors")

    # Cache to disk
    os.makedirs(processedmat_dir, exist_ok=True)
    df_all.to_parquet(cache_path, index=False)
    print(f"Saved cached calibration data to: {cache_path}")

    return df_all, errors


# ============================================================
# Courtship-fraction matching utilities
# ============================================================

def compute_courtship_fraction(df_all, grouper=None):
    """
    Compute courtship fraction per block for id=0 (focal male).

    Parameters
    ----------
    df_all : pd.DataFrame
        Full dataset (both ids). Must contain 'id', 'courtship', 'frame',
        'species', 'acquisition', 'led_level', 'speed_hz' columns.
    grouper : list of str, optional
        Columns to group by. Defaults to
        ['species', 'acquisition', 'age_ATR', 'led_type', 'led_level', 'speed_hz'].

    Returns
    -------
    pd.DataFrame
        One row per group with columns: <grouper> + court_frames, total_frames,
        courtship_frac.
    """
    if grouper is None:
        grouper = ['species', 'acquisition', 'age_ATR', 'led_type',
                   'led_level', 'led_intensity', 'speed_hz']

    f1 = df_all[df_all['id'] == 0].copy()
    court_frames = (
        f1[f1['courtship'] == 1]
        .groupby(grouper)['frame'].count()
        .reset_index(name='court_frames')
    )
    total_frames = (
        f1.groupby(grouper)['frame'].count()
        .reset_index(name='total_frames')
    )
    court_frac = court_frames.merge(total_frames, on=grouper, how='right')
    court_frac['court_frames'] = court_frac['court_frames'].fillna(0)
    court_frac['courtship_frac'] = (
        court_frac['court_frames'] / court_frac['total_frames']
    )
    return court_frac


def get_courtship_matched_blocks(df_all, min_courtship_frac=0.3,
                                  exclude_led0=True, exclude_speed0=True):
    """
    Identify (acquisition, led_level, speed_hz) blocks where courtship
    fraction exceeds a threshold, suitable for species comparisons.

    Only includes blocks where BOTH species have at least some flies
    meeting the threshold.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full dataset (both ids).
    min_courtship_frac : float
        Minimum per-block courtship fraction to include.
    exclude_led0 : bool
        Exclude LED level 0 blocks.
    exclude_speed0 : bool
        Exclude speed 0 blocks.

    Returns
    -------
    matched_df : pd.DataFrame
        Subset of df_all (id=0, courtship=1 only) containing only frames
        from blocks meeting the courtship threshold.
    court_frac : pd.DataFrame
        Full courtship fraction table (for diagnostics/plotting).
    block_keys : pd.DataFrame
        The (acquisition, led_level, speed_hz) combinations that passed.
    """
    grouper = ['species', 'acquisition', 'age_ATR', 'led_type',
               'led_level', 'led_intensity', 'speed_hz']
    court_frac = compute_courtship_fraction(df_all, grouper=grouper)

    # Filter to active blocks
    mask = pd.Series(True, index=court_frac.index)
    if exclude_led0:
        mask &= court_frac['led_level'] > 0
    if exclude_speed0:
        mask &= court_frac['speed_hz'] > 0
    active = court_frac[mask].copy()

    # Apply threshold — each fly's block is included independently;
    # species are not required to share the same (led_intensity, speed_hz).
    above_thr = active[active['courtship_frac'] >= min_courtship_frac].copy()

    # Block keys use led_intensity (not led_level) so that blocks are
    # comparable across led_types with different intensity mappings.
    block_keys = above_thr[['acquisition', 'led_intensity', 'speed_hz']].drop_duplicates()

    # Filter main dataframe to courtship frames in matched blocks
    f1_court = df_all[(df_all['id'] == 0) & (df_all['courtship'] == 1)].copy()
    matched_df = f1_court.merge(
        block_keys, on=['acquisition', 'led_intensity', 'speed_hz'], how='inner'
    )

    n_mel = matched_df[matched_df['species'] == 'Dmel']['acquisition'].nunique()
    n_yak = matched_df[matched_df['species'] == 'Dyak']['acquisition'].nunique()
    print(f"Courtship-matched blocks (frac >= {min_courtship_frac}):")
    print(f"  {len(block_keys)} blocks pass threshold")
    print(f"  Dmel: {n_mel} acquisitions, Dyak: {n_yak} acquisitions")
    print(f"  Total matched courtship frames: {len(matched_df)}")

    return matched_df, court_frac, block_keys


# ============================================================
# Age-ATR condition selection
# ============================================================

def find_comparable_age_atr(df_all, exclude_led0=True, exclude_speed0=True,
                             max_conditions=2, min_flies_per_species=3,
                             top_n_speeds=None):
    """
    Identify age_ATR condition(s) that yield the most comparable
    courtship levels between species, evaluated at the best speeds.

    For each candidate (single or pair of age_ATR conditions):
      1. Compute mean courtship per (species, speed_hz), averaged across
         LED levels and flies.
      2. Rank speeds by combined courtship from both species.
      3. Keep only the top N speeds (or all, if top_n_speeds is None).
      4. Score by mean |species_diff| across those top speeds.

    Candidates are ranked primarily by the number of viable speeds
    (where both species court above 0), then by species similarity
    at the top speeds.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full dataset (both ids).
    exclude_led0, exclude_speed0 : bool
        Exclude baseline blocks when computing courtship fractions.
    max_conditions : int
        Maximum number of age_ATR conditions to combine (1 or 2).
    min_flies_per_species : int
        Minimum unique acquisitions per species for a candidate.
    top_n_speeds : int or None
        Number of top speeds to use for scoring.  If None, uses all
        speeds where both species have data.

    Returns
    -------
    selected : list of str
        The best age_ATR condition(s).
    fly_means : pd.DataFrame
        Per-fly mean courtship fractions (species, acquisition, age_ATR).
    candidates : list of dict
        All scored candidates, sorted best-first.
    """
    from itertools import combinations

    grouper = ['species', 'acquisition', 'age_ATR', 'led_level', 'speed_hz']
    court_frac = compute_courtship_fraction(df_all, grouper=grouper)

    active = court_frac.copy()
    if exclude_led0:
        active = active[active['led_level'] > 0]
    if exclude_speed0:
        active = active[active['speed_hz'] > 0]

    fly_means = active.groupby(
        ['species', 'acquisition', 'age_ATR']
    )['courtship_frac'].mean().reset_index()

    # Per-fly, per-speed means (average across LED levels)
    fly_speed_means = active.groupby(
        ['species', 'acquisition', 'age_ATR', 'speed_hz']
    )['courtship_frac'].mean().reset_index()

    all_age_atr = sorted(fly_means['age_ATR'].unique())
    species_list = sorted(fly_means['species'].unique())

    if len(species_list) < 2 or len(all_age_atr) <= 1:
        return all_age_atr, fly_means, []

    sp0, sp1 = species_list[0], species_list[1]

    candidates = []
    for r in range(1, min(max_conditions, len(all_age_atr)) + 1):
        for combo in combinations(all_age_atr, r):
            combo_fly = fly_means[fly_means['age_ATR'].isin(combo)]
            sp_stats = combo_fly.groupby('species').agg(
                n_flies=('acquisition', 'nunique'),
            )
            if len(sp_stats) < 2:
                continue
            if (sp_stats['n_flies'] < min_flies_per_species).any():
                continue

            # Per-speed analysis: each species scored at its own
            # best speeds (species may have different speed prefs).
            combo_speed = fly_speed_means[
                fly_speed_means['age_ATR'].isin(combo)
            ]
            speed_by_sp = combo_speed.groupby(
                ['species', 'speed_hz']
            )['courtship_frac'].mean().reset_index()

            sp_info = {}
            min_viable = np.inf
            for sp in [sp0, sp1]:
                sp_speeds = speed_by_sp[
                    speed_by_sp['species'] == sp
                ].sort_values('courtship_frac', ascending=False)

                if len(sp_speeds) == 0:
                    break

                n_use = top_n_speeds if top_n_speeds else len(sp_speeds)
                top_sp = sp_speeds.head(n_use)

                sp_info[sp] = {
                    'mean_frac': top_sp['courtship_frac'].mean(),
                    'top_speeds': top_sp['speed_hz'].tolist(),
                    'n_viable': len(top_sp),
                }
                min_viable = min(min_viable, sp_info[sp]['n_viable'])

            if len(sp_info) < 2:
                continue

            species_diff = abs(
                sp_info[sp0]['mean_frac'] - sp_info[sp1]['mean_frac'])

            candidates.append({
                'age_atr_set': list(combo),
                'species_diff': species_diff,
                'n_flies': dict(sp_stats['n_flies']),
                'mean_frac': {sp: sp_info[sp]['mean_frac']
                              for sp in [sp0, sp1]},
                'n_viable_speeds': int(min_viable),
                'top_speeds': {sp: sp_info[sp]['top_speeds']
                               for sp in [sp0, sp1]},
            })

    if not candidates:
        print("Warning: no age_ATR candidates met criteria, using all")
        return all_age_atr, fly_means, []

    # Prefer more viable speeds, then smaller species difference
    candidates.sort(key=lambda c: (-c['n_viable_speeds'],
                                    c['species_diff']))
    best = candidates[0]

    n_label = f'top {top_n_speeds}' if top_n_speeds else 'all'
    print(f"Age-ATR selection: {best['age_atr_set']} "
          f"(each species scored on its own {n_label} speeds)")
    for sp, frac in best['mean_frac'].items():
        sp_speeds = best['top_speeds'].get(sp, [])
        print(f"  {sp}: mean courtship frac = {frac:.3f} "
              f"({best['n_flies'][sp]} flies, "
              f"best speeds: {sp_speeds})")
    print(f"  |Species diff| at own best speeds: "
          f"{best['species_diff']:.4f}")

    return best['age_atr_set'], fly_means, candidates


# ============================================================
# Convenience loader
# ============================================================

def load_and_prepare_dataset(rootdir=None, processedmat_dir=None,
                              create_new=False, min_courtship_frac=None,
                              exclude_led0=True, exclude_speed0=True,
                              age_atr=None, max_age_atr_conditions=2,
                              min_flies_per_species=3,
                              top_n_speeds=None):
    """
    One-call setup: load metadata, load data, select comparable age_ATR
    conditions, compute courtship-matched blocks.

    Two-stage matching:
      1) Select age_ATR condition(s) where species have the most comparable
         courtship levels at the best speeds (or use caller-specified).
      2) Within selected conditions, match (led_level, speed_hz) blocks
         where courtship fraction >= threshold for both species.

    Parameters
    ----------
    rootdir, processedmat_dir : str or None
        Data paths (defaults from module constants).
    create_new : bool
        Force reprocessing of raw data.
    min_courtship_frac : float or None
        Block-level courtship threshold (default 0.2).
    exclude_led0, exclude_speed0 : bool
        Exclude baseline blocks.
    age_atr : str, list of str, or None
        Override automatic age_ATR selection.  If None, automatically
        selects the condition(s) with most comparable courtship.
    max_age_atr_conditions : int
        Max number of age_ATR conditions to combine (auto mode).
    min_flies_per_species : int
        Minimum acquisitions per species for an age_ATR candidate.
    top_n_speeds : int or None
        Score age_ATR candidates using only the top N speeds (ranked
        by combined courtship).  None uses all available speeds.

    Returns
    -------
    dict with keys:
        df_all              Full dataset (all age_ATR conditions).
        df_filtered         After age_ATR selection.
        matched_df          Courtship frames from matched blocks.
        court_frac          Courtship-fraction table from matching step.
        block_keys          (acquisition, led_level, speed_hz) that passed.
        meta                Metadata DataFrame.
        basedir             Base output directory.
        min_courtship_frac  Threshold used.
        selected_age_atr    The age_ATR condition(s) used.
        age_atr_candidates  All scored age_ATR candidates (or None).
        age_atr_fly_means   Per-fly courtship fracs for all age_ATR.
        errors              Files that failed to load.
    """
    if rootdir is None:
        rootdir = DEFAULT_ROOTDIR
    if processedmat_dir is None:
        processedmat_dir = DEFAULT_PROCESSEDMAT_DIR
    if min_courtship_frac is None:
        min_courtship_frac = DEFAULT_MIN_COURTSHIP_FRAC

    os.makedirs(processedmat_dir, exist_ok=True)
    basedir = os.path.split(processedmat_dir)[0]

    meta_fpath = glob.glob(os.path.join(rootdir, '*.csv'))[0]
    meta0 = pd.read_csv(meta_fpath)
    meta = meta0[
        (meta0['tracked in matlab and checked for swaps'] == 1)
        & (meta0['calibration'] == 1)
        & (meta0['speed_blocks_marked'] == 1)
        & ~(meta0['led_onset_frame'].isna())
    ].copy()
    print(f"Calibration files: {meta.shape[0]}")

    df_all, errors = load_all_calibration_data(
        rootdir, meta, processedmat_dir,
        n_LED_blocks=6, n_speed_blocks=5,
        block_dur_sec=20, fps=60,
        create_new=create_new,
    )

    # --- Stage 1: age_ATR selection ---
    if age_atr is not None:
        if isinstance(age_atr, str):
            age_atr = [age_atr]
        selected_age_atr = list(age_atr)
        age_atr_fly_means, age_atr_candidates = None, None
        print(f"Using user-specified age_ATR: {selected_age_atr}")
    else:
        selected_age_atr, age_atr_fly_means, age_atr_candidates = \
            find_comparable_age_atr(
                df_all,
                exclude_led0=exclude_led0,
                exclude_speed0=exclude_speed0,
                max_conditions=max_age_atr_conditions,
                min_flies_per_species=min_flies_per_species,
                top_n_speeds=top_n_speeds,
            )

    df_filtered = df_all[df_all['age_ATR'].isin(selected_age_atr)].copy()
    print(f"After age_ATR filter ({selected_age_atr}): "
          f"{df_filtered['acquisition'].nunique()} acquisitions, "
          f"{len(df_filtered)} frames")

    # --- Stage 2: block-level courtship matching ---
    matched_df, court_frac, block_keys = get_courtship_matched_blocks(
        df_filtered, min_courtship_frac=min_courtship_frac,
        exclude_led0=exclude_led0, exclude_speed0=exclude_speed0,
    )

    return {
        'df_all': df_all,
        'df_filtered': df_filtered,
        'matched_df': matched_df,
        'court_frac': court_frac,
        'block_keys': block_keys,
        'meta': meta,
        'basedir': basedir,
        'min_courtship_frac': min_courtship_frac,
        'selected_age_atr': selected_age_atr,
        'age_atr_candidates': age_atr_candidates,
        'age_atr_fly_means': age_atr_fly_means,
        'errors': errors,
    }
