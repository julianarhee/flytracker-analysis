#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_multichamber.py

Preprocessing pipeline for the `strain_variation` analysis: load the Dmel and
Dyak 2x2 multichamber courtship assays (Caitlin RA data), transform each
fly-pair to egocentric / relative metrics, attach manual `-actions.mat`
annotations and automated JAABA scores, and save an aggregated parquet for
downstream analyses.

Processing style follows `analyses/multichamber/src/multichamber.py`:
  1. Load FlyTracker -track.mat / -feat.mat / calibration.mat.
  2. Assign sex (even id = male, odd = female) and fly_pair / strain by arena
     (2x2 lookup; see strain_funcs.meta_flynum_to_ft_id).
  3. Read copulation onset from -actions.mat and transform each fly-pair to
     relative metrics (transform_data.relative_metrics.do_transformations_on_df),
     truncating at copulation.
  4. Merge manual annotations (courtship / chasing / copulation / unilateral
     extension) per fly id and automated JAABA scores.
  5. Save per-acquisition parquet + an aggregated parquet under the data root.

Run (single acquisition, e.g. for QC):
    python analyses/strain_variation/src/process_multichamber.py \
        --species Dmel --single 20250404-1431_fly1-4_Dmel-strain_4do_gh

Run (aggregate a species, or both):
    python analyses/strain_variation/src/process_multichamber.py --species Dmel
    python analyses/strain_variation/src/process_multichamber.py --species both --new
"""
import os
import sys
import glob
import argparse
import traceback

import numpy as np
import pandas as pd

import libs.utils as util
import libs.qc as qc
import transform_data.relative_metrics as rel

# Support both `python process_multichamber.py` and package imports.
try:
    import analyses.strain_variation.src.strain_funcs as sf
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    import strain_funcs as sf


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def get_output_dirs(rootdir, make=True):
    """Output dirs under the data root (mirrors multichamber's processed/ dir).

    Returns (procdir, aggdir):
        procdir - per-acquisition parquet files
        aggdir  - aggregated per-species / combined parquet
    """
    outbase = os.path.join(rootdir, '2x2_strains_processed')
    procdir = os.path.join(outbase, 'processed')
    aggdir = outbase
    if make:
        os.makedirs(procdir, exist_ok=True)
    return procdir, aggdir


def list_acquisitions(species_dir):
    """List acquisition folders (named `20*`) under a species directory."""
    return sorted([f for f in os.listdir(species_dir)
                   if f.startswith('20') and os.path.isdir(os.path.join(species_dir, f))])


# ---------------------------------------------------------------------------
# Per-acquisition processing
# ---------------------------------------------------------------------------
def get_frame_dimensions(acqdir, calib):
    """Frame width/height in pixels: prefer calibration.mat, fall back to video."""
    w = calib.get('w') if calib else None
    h = calib.get('h') if calib else None
    if np.isscalar(w) and np.isscalar(h) and w and h:
        return float(w), float(h)
    # Fall back to the movie (does not read frames).
    cap = rel.get_video_cap(acqdir, movie_fmt='avi')
    import cv2
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return float(w), float(h)


# Orientation head/tail-polarity methods (shared; see libs.qc.resolve_orientation).
ORI_METHODS = qc.ORI_METHODS


def process_single_acquisition(acqdir, meta, species=None, fps=60, array_size='2x2',
                               add_actions=True, add_jaaba=True, ori_method='velocity',
                               verbose=True):
    """Load one acquisition, transform to relative metrics, attach annotations.

    Args:
        acqdir (str): acquisition directory (contains nested FlyTracker output).
        meta (pd.DataFrame): full metadata; the acq subset is selected here.
        species (str or None): 'Dmel' / 'Dyak' (stored on the output).
        fps (float): frame rate.
        array_size (str): chamber layout ('2x2').
        add_actions (bool): merge manual -actions.mat annotations.
        add_jaaba (bool): merge automated JAABA scores.
        ori_method (str): how head/tail orientation polarity is handled, one of
            'velocity' (default) / 'wing' / 'none'. This is dispatched to the
            shared, dataset-agnostic `libs.qc.resolve_orientation` (see its
            docstring for the full description of each method, the forward-motion
            assumption behind 'velocity', when to prefer 'wing'/'none' for assays
            with substantial sideways/backward motion, and the rationale for
            deferring a sideways-aware middle-ground option).

    Returns:
        pd.DataFrame or None: transformed + annotated data for the acquisition.
    """
    if ori_method not in ORI_METHODS:
        raise ValueError("ori_method must be one of {}, got {!r}".format(ORI_METHODS, ori_method))
    acq = os.path.basename(acqdir.rstrip('/'))
    if verbose:
        print("[{}] loading FlyTracker data (ori_method={})".format(acq, ori_method))

    # Always load the raw body-axis ori (filter_ori=False); the orientation
    # polarity handling is applied uniformly below by qc.resolve_orientation,
    # which implements all three methods (including the legacy wing filter).
    calib, trk_, feat_ = util.load_flytracker_data(
        acqdir, fps=fps, calib_is_upstream=False, subfolder='*', filter_ori=False)
    if trk_ is None or feat_ is None:
        print("[{}] no tracking data; skipping".format(acq))
        return None
    trk_['acquisition'] = acq

    meta_acq = meta[meta['acquisition'] == acq]
    if len(meta_acq) == 0:
        print("[{}] no metadata row; skipping".format(acq))
        return None

    n_ids = len(trk_['id'].unique())
    n_pairs = len(meta_acq['fly_num'].unique())
    if n_ids != 2 * n_pairs:
        print("[{}] WARNING: {} fly ids but {} metadata pairs (expected {} ids)".format(
            acq, n_ids, n_pairs, 2 * n_pairs))

    # Conditions
    trk_ = sf.assign_sex(trk_)
    trk_ = sf.assign_frame_number(trk_)
    trk_ = sf.assign_strain_to_multichamber(trk_, meta_acq, array_size=array_size)
    if species is not None:
        trk_['dataset_species'] = species

    # Copulation onset per fly id (for truncation)
    actions_fpath = sf.find_actions_mat(acqdir)
    if actions_fpath is not None:
        cop_dict = sf.get_copulation_frames(actions_fpath)
    else:
        print("[{}] WARNING: no -actions.mat".format(acq))
        cop_dict = {k: -1 for k in trk_['id'].unique()}

    # Frame dimensions (for centering during the transform)
    frame_width, frame_height = get_frame_dimensions(acqdir, calib)

    # Handle head/tail orientation polarity on the RAW ori (before the negation
    # below, which is the convention the velocity resolver expects). Dispatched
    # to the shared libs.qc.resolve_orientation.
    trk_, ori_info = qc.resolve_orientation(trk_, method=ori_method, fps=fps, verbose=verbose)
    if verbose and ori_info.get('per_fly') is not None:
        n_flip = sum(s['frac_frames_flipped'] > 0 for s in ori_info['per_fly'].values())
        print("[{}] flip-resolved orientation: {} of {} flies had flipped chunks".format(
            acq, n_flip, len(ori_info['per_fly'])))

    # Flip ORI for FlyTracker -> match convention used by relative_metrics
    trk_['ori'] = -1 * trk_['ori']

    # Transform each fly-pair to relative metrics
    acq_dfs = []
    for flypair, curr_trk in trk_.groupby('fly_pair'):
        curr_feat = feat_.loc[curr_trk.index].copy()
        flyid1 = int(curr_trk['id'].min())  # even = male
        flyid2 = int(curr_trk['id'].max())  # odd  = female
        cop_ix = cop_dict.get(flyid1, -1)
        cop_ix = None if cop_ix in (-1, None) else cop_ix
        try:
            transf_df = rel.do_transformations_on_df(
                curr_trk, frame_width, frame_height, feat_=curr_feat,
                cop_ix=cop_ix, flyid1=flyid1, flyid2=flyid2,
                get_relative_sizes=False, fps=fps)
        except Exception:
            print("[{}] ERROR transforming pair {} (ids {},{})".format(
                acq, flypair, flyid1, flyid2))
            traceback.print_exc()
            continue
        # Carry over the per-pair condition columns dropped by the transform.
        for col in ['acquisition', 'fly_pair', 'strain', 'strain_male',
                    'strain_female', 'species', 'dataset_species']:
            if col in curr_trk.columns and col not in transf_df.columns:
                if col in ('acquisition', 'dataset_species', 'fly_pair'):
                    transf_df[col] = curr_trk[col].iloc[0]
                else:
                    # strain/species are per-pair constants here
                    transf_df[col] = curr_trk[col].dropna().iloc[0] \
                        if curr_trk[col].notna().any() else np.nan
        acq_dfs.append(transf_df)

    if not acq_dfs:
        print("[{}] no transformed pairs; skipping".format(acq))
        return None
    acq_df = pd.concat(acq_dfs, ignore_index=True)
    acq_df['frame'] = acq_df['frame'].astype(int)

    # Manual annotations (per fly id)
    if add_actions and actions_fpath is not None:
        actions = sf.ft_actions_to_bout_df(actions_fpath, behaviors=sf.COURTSHIP_BEHAVIORS)
        acq_df = sf.assign_action_frames_to_df(acq_df, actions)

    # Automated JAABA scores
    if add_jaaba:
        acq_df, missing = sf.add_jaaba_scores(acq_df, acqdir)
        if missing:
            print("[{}] no JAABA scores for: {}".format(acq, missing))

    if verbose:
        print("[{}] done: {} rows, {} pairs".format(acq, len(acq_df), n_pairs))
    return acq_df


# ---------------------------------------------------------------------------
# Batch / aggregation
# ---------------------------------------------------------------------------
def process_species(species, rootdir=sf.ROOTDIR, fps=60, array_size='2x2',
                    create_new=False, acqs=None, save_each=True, ori_method='velocity'):
    """Process every acquisition for one species and return the aggregate df.

    Per-acquisition parquet files are written to <rootdir>/2x2_strains_processed/
    processed/<acq>.parquet (skipped if present unless `create_new`).
    `ori_method` selects the orientation-polarity handling; see
    `process_single_acquisition`.
    """
    species_dir = os.path.join(rootdir, sf.SPECIES_DIRS[species])
    meta = sf.load_strain_metadata(species_dir)
    procdir, _ = get_output_dirs(rootdir)

    if acqs is None:
        acqs = list_acquisitions(species_dir)

    d_list = []
    for acq in acqs:
        acqdir = os.path.join(species_dir, acq)
        out_fpath = os.path.join(procdir, '{}.parquet'.format(acq))
        if os.path.exists(out_fpath) and not create_new:
            print("[{}] cached; loading".format(acq))
            d_list.append(pd.read_parquet(out_fpath))
            continue
        acq_df = process_single_acquisition(
            acqdir, meta, species=species, fps=fps, array_size=array_size,
            ori_method=ori_method)
        if acq_df is None:
            continue
        if save_each:
            acq_df.to_parquet(out_fpath, engine='pyarrow', compression='snappy')
        d_list.append(acq_df)

    if not d_list:
        return pd.DataFrame()
    df0 = pd.concat(d_list, ignore_index=True)
    df0 = assign_global_id(df0)
    return df0


def assign_global_id(df0):
    """Assign a unique `global_id` across acquisitions (per acquisition+id)."""
    df0 = df0.copy()
    df0['global_id'] = -1
    curr = 0
    for (_, _), idx in df0.groupby(['acquisition', 'id']).groups.items():
        df0.loc[idx, 'global_id'] = curr
        curr += 1
    return df0


def aggregate_and_save(species_list, rootdir=sf.ROOTDIR, fps=60, array_size='2x2',
                       create_new=False, acqs=None, ori_method='velocity'):
    """Process the requested species and save per-species + combined parquet."""
    _, aggdir = get_output_dirs(rootdir)
    per_species = {}
    for sp in species_list:
        print("\n=== Processing species: {} ===".format(sp))
        df_sp = process_species(sp, rootdir=rootdir, fps=fps, array_size=array_size,
                                create_new=create_new, acqs=acqs, ori_method=ori_method)
        if df_sp.empty:
            print("=== {}: no data ===".format(sp))
            continue
        sp_fpath = os.path.join(aggdir, '{}_2x2_strains.parquet'.format(sp))
        df_sp.to_parquet(sp_fpath, engine='pyarrow', compression='snappy')
        print("Saved {}: {} rows -> {}".format(sp, len(df_sp), sp_fpath))
        per_species[sp] = df_sp

    if len(per_species) > 1:
        combined = pd.concat(per_species.values(), ignore_index=True)
        comb_fpath = os.path.join(aggdir, 'mel_yak_2x2_strains.parquet')
        combined.to_parquet(comb_fpath, engine='pyarrow', compression='snappy')
        print("Saved combined: {} rows -> {}".format(len(combined), comb_fpath))
    return per_species


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process Dmel/Dyak 2x2 multichamber strain data into relative metrics.')
    parser.add_argument('--rootdir', type=str, default=sf.ROOTDIR,
                        help='Data root containing the species directories.')
    parser.add_argument('--species', type=str, default='both',
                        choices=['Dmel', 'Dyak', 'both'],
                        help='Species to process (default: both).')
    parser.add_argument('--single', type=str, default=None,
                        help='Process a single acquisition name (requires --species Dmel/Dyak).')
    parser.add_argument('--fps', type=float, default=60.0, help='Frame rate (default: 60).')
    parser.add_argument('--array', type=str, default='2x2', help='Array size (default: 2x2).')
    parser.add_argument('--new', action='store_true',
                        help='Recompute even if a per-acquisition parquet exists.')
    parser.add_argument('--ori_method', type=str, default='velocity',
                        choices=list(ORI_METHODS),
                        help="Head/tail orientation-polarity handling: 'velocity' (default; "
                             "keep body-axis ori, NaN only motion-contradicting flips), "
                             "'wing' (NaN ori where wings undetected; no motion assumption), "
                             "'none' (raw FlyTracker ori). Use 'wing'/'none' for assays with "
                             "substantial sideways/backward motion.")
    args = parser.parse_args()

    species_list = ['Dmel', 'Dyak'] if args.species == 'both' else [args.species]

    if args.single is not None:
        assert args.species in ('Dmel', 'Dyak'), '--single requires --species Dmel or Dyak'
        aggregate_and_save(species_list, rootdir=args.rootdir, fps=args.fps,
                           array_size=args.array, create_new=True, acqs=[args.single],
                           ori_method=args.ori_method)
    else:
        aggregate_and_save(species_list, rootdir=args.rootdir, fps=args.fps,
                           array_size=args.array, create_new=args.new,
                           ori_method=args.ori_method)
