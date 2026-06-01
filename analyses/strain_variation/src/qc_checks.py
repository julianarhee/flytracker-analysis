#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_checks.py

Quality-control visualizations for the `strain_variation` pipeline. Checks:

  (1) Arena layout: scatter every fly's raw position colored by fly_pair and
      styled by sex, to confirm the 2x2 arenas are correctly assigned and each
      male/female pair sits in the same quadrant.

  (2) Example bout: for a chosen fly_pair, pull the longest manual courtship /
      chasing bout on the *focal male*, and plot the 2D trajectories + the
      target's egocentric position + the theta-error timecourse, plus a short
      video clip of the bout (heading arrow + target dot overlaid). The chosen
      fly_pair (and its male/female fly ids) are written into the figure titles
      and the output goes into a per-pair subdirectory so it is unambiguous
      which pair is shown. If the correct fly-ID pair is labeled, the target
      (female) should sit in front of the focal male (egocentric dot cloud
      forward of 0; theta_error near 0).

  (3) Head-tail flips: a per-acquisition flip-rate panel, plus a per-pair video
      montage of the frames around a flip cluster (montage only). See
      `libs.qc.resolve_flip_chunks` for how flipped chunks are identified and
      excluded.

Shared QC primitives (flip detection/resolution, the bout-video writer, and the
flip montage) live in `libs.qc`; this module orchestrates them for the 2x2
strain dataset. Reuses the diagnostic plotters in `libs.plotting` and the bout
finder in `libs.utils` (as in analyses/preprocessing/src/example_one_target.py).

Run as a script for a single acquisition:
    python analyses/strain_variation/src/qc_checks.py \
        --species Dmel --acq 20250404-1431_fly1-4_Dmel-strain_4do_gh
Or step through interactively with the #%% cells.
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import libs.utils as util
import libs.plotting as putil
import libs.qc as qc

try:
    import analyses.strain_variation.src.strain_funcs as sf
    import analyses.strain_variation.src.process_multichamber as pm
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    import strain_funcs as sf
    import process_multichamber as pm


# Thin re-exports of the shared head-tail flip primitives (kept so existing
# callers / tests can `import qc_checks` and find them here).
wrapped_ori_jump_deg = qc.wrapped_ori_jump_deg
detect_headtail_flips = qc.detect_headtail_flips
find_flip_window = qc.find_flip_window
resolve_flip_chunks = qc.resolve_flip_chunks


# ---------------------------------------------------------------------------
# Loading / paths
# ---------------------------------------------------------------------------
def load_processed_acquisition(acq, rootdir=sf.ROOTDIR):
    """Load a per-acquisition processed parquet written by process_multichamber."""
    procdir, _ = pm.get_output_dirs(rootdir, make=False)
    fpath = os.path.join(procdir, '{}.parquet'.format(acq))
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            "No processed parquet for {}. Run process_multichamber.py --single first."
            .format(acq))
    return pd.read_parquet(fpath)


def acq_qc_dir(savedir, acq, make=True):
    """Acquisition-level QC dir: <savedir>/<acq>/ (arena layout, flip summary)."""
    d = os.path.join(savedir, acq)
    if make:
        os.makedirs(d, exist_ok=True)
    return d


def pair_qc_dir(savedir, acq, fly_pair, make=True):
    """Per-flypair QC dir: <savedir>/<acq>/pair<N>/ so all plots for a given
    fly_pair from a given acquisition are grouped together."""
    d = os.path.join(savedir, acq, 'pair{}'.format(int(fly_pair)))
    if make:
        os.makedirs(d, exist_ok=True)
    return d


def truncate_at_copulation(trk, cop_dict):
    """Drop each fly's frames at/after its copulation onset.

    The processed parquet is already truncated at copulation (the transform
    does `fly.iloc[:cop_ix]`), but RAW tracking loaded for the flip QC is not —
    and post-copulation flies are stationary/mounted with unreliable
    orientation, which inflates the flip rate. `cop_dict` maps fly id ->
    copulation onset frame (-1/None if the pair never copulated); see
    `strain_funcs.get_copulation_frames`.
    """
    keep = pd.Series(True, index=trk.index)
    for fid, onset in (cop_dict or {}).items():
        if onset is not None and onset > 0:
            keep &= ~((trk['id'] == fid) & (trk['frame'] >= onset))
    return trk[keep].copy()


# ---------------------------------------------------------------------------
# QC (1): arena layout
# ---------------------------------------------------------------------------
def plot_arena_layout(acq_df, acq=None, max_points=3000, figsize=(7, 7)):
    """Scatter all flies' raw positions, colored by fly_pair, styled by sex.

    Verifies the 2x2 arena assignment: each fly_pair should occupy one quadrant
    and contain exactly one male (circle) and one female (x).
    """
    fig, ax = plt.subplots(figsize=figsize)
    pairs = sorted(acq_df['fly_pair'].dropna().unique())
    palette = putil.get_palette_dict(acq_df, 'fly_pair', cmap='viridis')
    for fp in pairs:
        for sex, marker in [('m', 'o'), ('f', 'x')]:
            sub = acq_df[(acq_df['fly_pair'] == fp) & (acq_df['sex'] == sex)]
            if sub.empty:
                continue
            if len(sub) > max_points:
                sub = sub.sample(max_points, random_state=0)
            strain = sub['strain'].iloc[0] if 'strain' in sub.columns else ''
            ax.scatter(sub['pos_x'], sub['pos_y'], s=3, marker=marker,
                       color=palette.get(fp, 'gray'), alpha=0.4,
                       label='pair {} {} ({})'.format(int(fp), sex, strain))
    ax.set_xlabel('pos_x (px)')
    ax.set_ylabel('pos_y (px)')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend(frameon=False, fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5),
              markerscale=3)
    ax.set_title('Arena layout / fly-pair assignment{}'.format(
        '\n{}'.format(acq) if acq else ''))
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# QC (2): example bout, focal/target pairing
# ---------------------------------------------------------------------------
def get_pair_focal_target(acq_df, fly_pair):
    """Return (focal_male_df, target_female_df) for a fly_pair."""
    pair_df = acq_df[acq_df['fly_pair'] == fly_pair]
    male = pair_df[pair_df['sex'] == 'm'].sort_values('frame').reset_index(drop=True)
    female = pair_df[pair_df['sex'] == 'f'].sort_values('frame').reset_index(drop=True)
    return male, female


def _resolve_action_column(df, action):
    """Pick the best available binary column for `action` (manual first, then JAABA)."""
    for col in [action, 'jaaba_{}_binary'.format(action), '{}_binary'.format(action)]:
        if col in df.columns:
            return col
    return None


def qc_example_bout(acq_df, fly_pair=None, action='chasing', fps=60,
                    snippet_dur_sec=3, acq=None, acqdir=None, savedir=None,
                    video_pad_frames=60):
    """Plot trajectories + egocentric target + theta-error for a courtship bout,
    and (if `acqdir`/`savedir` given) write a short video clip of the bout.

    Picks the longest manual `action` bout (chasing/courtship) annotated on the
    focal male of `fly_pair`. If none is annotated, falls back to the JAABA
    binary column. The chosen fly_pair and its male/female fly ids are written
    into every figure title, and (when `savedir` is given) all outputs go into
    `pair_qc_dir(savedir, acq, fly_pair)` so the pair is unambiguous.

    Returns:
        dict with 'fly_pair', 'male_id', 'female_id', 'f_start', 'f_end', the
        matplotlib figures, QC sanity numbers, and 'video_path' (or None).
    """
    # Pick a fly_pair that actually has the action annotated, if not specified.
    candidates = [fly_pair] if fly_pair is not None \
        else sorted(acq_df['fly_pair'].dropna().unique())

    chosen = None
    for fp in candidates:
        male, female = get_pair_focal_target(acq_df, fp)
        action_col = _resolve_action_column(male, action)
        if action_col is None:
            continue
        if male[action_col].fillna(0).sum() > 0:
            chosen = (fp, male, female, action_col)
            break
    if chosen is None:
        raise ValueError("No '{}' bouts found for the requested fly_pair(s)."
                         .format(action))
    fp, male, female, action_col = chosen
    male_id = int(male['id'].iloc[0])
    female_id = int(female['id'].iloc[0])
    print("Using fly_pair {} (focal male id={}, target female id={}), action='{}'".format(
        int(fp), male_id, female_id, action_col))

    stretch, f_start, f_end = util.find_action_snippet(
        male, action=action_col, fps=fps, snippet_dur_sec=snippet_dur_sec)
    stretch_targ = female[(female['frame'] >= f_start) & (female['frame'] <= f_end)].copy()

    # Pair/ids in the title so it is always clear which pair is shown.
    title = '{} | pair {} (male id{}, female id{}) | {} | frames {}-{}'.format(
        acq or '', int(fp), male_id, female_id, action_col, f_start, f_end)
    fig_traj, _ = putil.diagnostics_plot_2d_traj_and_rel(
        stretch, stretch_targ, f_start, f_end, title=title)

    fig_tc = None
    if 'theta_error' in stretch.columns:
        # FlyTracker's `ang_vel` (feat.mat) is an unsigned angular *speed*
        # (always >= 0); `ang_vel_fly` is the signed turn rate. We set
        # `ang_vel_deg` to the unsigned speed as-is (matching the
        # diagnostics_plot_timecourses default, which no longer negates it).
        stretch = stretch.copy()
        stretch['ang_vel_deg'] = np.degrees(stretch['ang_vel'])
        try:
            fig_tc = putil.diagnostics_plot_timecourses(
                stretch, f_start, f_end, fps=fps, title=title)
            if isinstance(fig_tc, (tuple, list)):
                fig_tc = fig_tc[0]
        except Exception as e:
            print("Skipping timecourse plot: {}".format(e))

    # Sanity numbers: during the bout the target should be in front (forward>0)
    # and theta_error should be small.
    fwd_frac = float((stretch['targ_rel_pos_x'] > 0).mean())
    med_theta = float(np.nanmedian(np.abs(np.degrees(stretch['theta_error'])))) \
        if 'theta_error' in stretch.columns else np.nan
    print("QC: fraction of bout with target in front = {:.2f}; "
          "median |theta_error| = {:.1f} deg".format(fwd_frac, med_theta))

    # Save figures + a bout video clip into the per-pair subdir.
    video_path = None
    if savedir is not None:
        pdir = pair_qc_dir(savedir, acq, fp)
        if fig_traj is not None:
            fig_traj.savefig(os.path.join(pdir, '{}_traj.png'.format(action)),
                             dpi=150, bbox_inches='tight')
        if fig_tc is not None:
            fig_tc.savefig(os.path.join(pdir, '{}_timecourse.png'.format(action)),
                           dpi=150, bbox_inches='tight')
        if acqdir is not None:
            # Pad the bout for a little context, clamped to available frames.
            vid_start = int(max(f_start - video_pad_frames, male['frame'].min()))
            vid_end = int(min(f_end + video_pad_frames, male['frame'].max()))
            # Transformed `ori` is drawn directly (ori_image_sign=+1). Color the
            # male's heading arrow by signed ang_vel_fly; female is the target dot.
            color_col = 'ang_vel_fly' if 'ang_vel_fly' in male.columns else 'ang_vel'
            video_path = qc.save_bout_video(
                acqdir, male, vid_start, vid_end,
                os.path.join(pdir, '{}_bout.avi'.format(action)),
                targdf=female, color_col=color_col, ori_col='ori',
                fps=fps, ori_image_sign=1)

    return {'fly_pair': fp, 'male_id': male_id, 'female_id': female_id,
            'f_start': f_start, 'f_end': f_end,
            'fig_traj': fig_traj, 'fig_timecourse': fig_tc,
            'fwd_frac': fwd_frac, 'median_abs_theta_error_deg': med_theta,
            'video_path': video_path}


# ---------------------------------------------------------------------------
# QC (3): head-tail flip summary (per acquisition, all flies)
# ---------------------------------------------------------------------------
def plot_headtail_flips(acq_df, acq=None, fps=60, figsize=(11, 5)):
    """Per-acquisition flip panel: flip-rate per fly + an event raster over time.

    Returns (fig, flip_summary_df).
    """
    if 'headtail_flip' not in acq_df.columns:
        acq_df = qc.detect_headtail_flips(acq_df)

    ids = sorted(acq_df['id'].unique())
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2.5]})

    labels, colors, fracs = [], [], []
    summary_rows = []
    for fid in ids:
        g = acq_df[acq_df['id'] == fid]
        sex = g['sex'].iloc[0] if 'sex' in g else '?'
        strain = g['strain'].iloc[0] if 'strain' in g and g['strain'].notna().any() else ''
        frac = 100.0 * g['headtail_flip'].mean()
        fracs.append(frac)
        colors.append('deepskyblue' if sex == 'm' else 'magenta')
        labels.append('id {} ({}{})'.format(fid, sex, ', ' + str(strain) if strain else ''))
        summary_rows.append({'id': fid, 'sex': sex, 'strain': strain,
                             'pct_flip_frames': round(frac, 2),
                             'n_flip_frames': int(g['headtail_flip'].sum())})

    y = np.arange(len(ids))
    ax0.barh(y, fracs, color=colors)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=7)
    ax0.invert_yaxis()
    ax0.set_xlabel('% frames flagged')
    ax0.set_title('Head-tail flip rate per fly')

    for k, fid in enumerate(ids):
        g = acq_df[acq_df['id'] == fid].sort_values('frame')
        t = g['frame'].values / fps
        flip_t = t[g['headtail_flip'].values.astype(bool)]
        if len(flip_t):
            ax1.eventplot(flip_t, lineoffsets=k, linelengths=0.8,
                          colors=colors[k], linewidths=0.5)
    ax1.set_yticks(range(len(ids)))
    ax1.set_yticklabels(['id {}'.format(i) for i in ids], fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Flip events over recording (each tick = flagged frame)')

    if acq:
        fig.suptitle(acq, y=1.02)
    plt.tight_layout()
    return fig, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Script / cell-style entry
# ---------------------------------------------------------------------------
#%%
if __name__ == '__main__' and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description='QC checks for strain_variation processing.')
    parser.add_argument('--rootdir', type=str, default=sf.ROOTDIR)
    parser.add_argument('--species', type=str, default='Dmel', choices=['Dmel', 'Dyak'])
    parser.add_argument('--acq', type=str, required=True, help='Acquisition name.')
    parser.add_argument('--fly_pair', type=int, default=None,
                        help='Restrict to one fly pair (default: every pair with a bout).')
    parser.add_argument('--action', type=str, default='courtship',
                        help='Behavior to find a bout for. Prefers a manual annotation '
                             'column, falling back to JAABA (default: courtship).')
    parser.add_argument('--savedir', type=str, default=None,
                        help='Base QC directory (default: <rootdir>/2x2_strains_processed/qc). '
                             'Outputs go to <savedir>/<acq>/ and <savedir>/<acq>/pair<N>/.')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip the bout videos and head-tail-flip video montages.')
    args = parser.parse_args()

    mpl.use('Agg')
    putil.set_sns_style(style='dark', min_fontsize=12)

    savedir = args.savedir or os.path.join(args.rootdir, '2x2_strains_processed', 'qc')
    acq_dir = acq_qc_dir(savedir, args.acq)
    acqdir = os.path.join(args.rootdir, sf.SPECIES_DIRS[args.species], args.acq)

    # Processed parquet is already truncated at copulation (transform does
    # fly.iloc[:cop_ix]), so the example-bout QC below is pre-copulation by
    # construction. NOTE: the `copulation` column here is a per-pair boolean
    # flag, not a per-frame indicator, so it is NOT used to filter.
    acq_df = load_processed_acquisition(args.acq, rootdir=args.rootdir)
    print("Loaded {}: {} rows, pairs={}, ids={}".format(
        args.acq, len(acq_df), sorted(acq_df['fly_pair'].dropna().unique()),
        sorted(acq_df['id'].unique())))

    # (1) Arena layout (acq-level).
    fig_arena, _ = plot_arena_layout(acq_df, acq=args.acq)
    fig_arena.savefig(os.path.join(acq_dir, 'arena_layout.png'),
                      dpi=150, bbox_inches='tight')

    # (2) Example courtship bout for EVERY pair (relative-trajectory + theta-error
    # timecourse + bout video into each pair's subdir). Pre-copulation by
    # construction (see note above).
    pairs = [args.fly_pair] if args.fly_pair is not None \
        else sorted(acq_df['fly_pair'].dropna().unique())
    for fp in pairs:
        try:
            res = qc_example_bout(acq_df, fly_pair=fp, action=args.action,
                                  acq=args.acq, savedir=savedir,
                                  acqdir=None if args.no_video else acqdir)
            print("pair {}: example {} bout -> {}".format(
                int(fp), args.action,
                pair_qc_dir(savedir, args.acq, res['fly_pair'], make=False)))
        except ValueError as e:
            print("pair {}: skipped ({})".format(int(fp), e))

    # (3) Head-tail flip QC. Run on RAW tracking (filter_ori=False): the processed
    # parquet has already NaN'd unreliable orientations (filter_ori=True), so the
    # real flips only show up in the raw data. Truncate each fly at copulation
    # onset first — post-copulation flies are stationary/mounted with unreliable
    # orientation, which would inflate the flip rate.
    _, raw_trk, _ = util.load_flytracker_data(
        acqdir, fps=60, calib_is_upstream=False, subfolder='*', filter_ori=False)
    raw_trk = sf.assign_sex(raw_trk)
    raw_trk = sf.assign_frame_number(raw_trk)
    meta = sf.load_strain_metadata(os.path.join(args.rootdir, sf.SPECIES_DIRS[args.species]))
    raw_trk = sf.assign_strain_to_multichamber(raw_trk, meta[meta['acquisition'] == args.acq])

    actions_fpath = sf.find_actions_mat(acqdir)
    if actions_fpath is not None:
        raw_trk = truncate_at_copulation(raw_trk, sf.get_copulation_frames(actions_fpath))
    raw_trk = qc.detect_headtail_flips(raw_trk)

    # Mark courtship frames on raw_trk so the flip-example montages can be drawn
    # from courtship bouts. Courtship is annotated on the male id only; apply it
    # to BOTH flies of the pair (the pair is courting during those frames).
    raw_trk['courtship'] = 0
    for fp in sorted(acq_df['fly_pair'].dropna().unique()):
        male, female = get_pair_focal_target(acq_df, fp)
        if 'courtship' not in male.columns or male['courtship'].fillna(0).sum() == 0:
            continue
        court_frames = set(male.loc[male['courtship'] == 1, 'frame'].astype(int))
        pair_ids = [int(male['id'].iloc[0]), int(female['id'].iloc[0])]
        mask = raw_trk['id'].isin(pair_ids) & raw_trk['frame'].isin(court_frames)
        raw_trk.loc[mask, 'courtship'] = 1

    fig_flip, flip_summary = plot_headtail_flips(raw_trk, acq=args.acq + ' (raw ori, pre-cop)')
    fig_flip.savefig(os.path.join(acq_dir, 'headtail_flips_summary.png'),
                     dpi=150, bbox_inches='tight')
    print("Head-tail flip rate per fly (raw ori, pre-copulation, ~180deg jumps):")
    print(flip_summary.to_string(index=False))

    # Per-pair flip montage of the densest-flip fly in each pair, selected from
    # courtship bouts (montage only).
    if not args.no_video:
        for fp, g in raw_trk.dropna(subset=['fly_pair']).groupby('fly_pair'):
            if args.fly_pair is not None and int(fp) != args.fly_pair:
                continue
            # Pick the fly in this pair with the most flips during courtship.
            best_id, best_n = None, -1
            for fid, fg in g.groupby('id'):
                _, nflip = qc.find_flip_window(fg, courtship_col='courtship')
                if nflip > best_n:
                    best_id, best_n = int(fid), nflip
            pdir = pair_qc_dir(savedir, args.acq, fp)
            vid_res = qc.plot_flip_montage(
                acqdir, fly_id=best_id, trk=raw_trk, courtship_col='courtship',
                savepath=os.path.join(pdir, 'headtail_flip_montage.png'),
                ori_image_sign=-1)
            fs = vid_res['flip_summary'] or {}
            print("pair {}: flip montage id {} ({} flips, parity={}, {:.3%} frames flipped)".format(
                int(fp), vid_res['fly_id'], vid_res['n_flips'],
                fs.get('parity_source', '?'), fs.get('frac_frames_flipped', 0.0)))

    print("Saved QC figures under {}".format(acq_dir))
