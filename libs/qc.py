#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc.py

Shared quality-control primitives for FlyTracker-style courtship data, used by
multiple analyses (e.g. `analyses/strain_variation`, `analyses/p1_levels`).

Two groups of helpers live here:

  Head-tail flips
  ---------------
  FlyTracker occasionally swaps a fly's head and tail, producing a ~180 degree
  frame-to-frame jump in `ori`. These split a track into contiguous *chunks*,
  alternating between the correct orientation and a flipped one.

    - `wrapped_ori_jump_deg`     frame-to-frame |Δori| wrapped to [0, 180]
    - `detect_headtail_flips`    per-id boolean `headtail_flip` column
    - `find_flip_window`         densest flip window for one fly
    - `resolve_flip_chunks`      decide which chunks are correct vs flipped
    - `exclude_flipped_orientation`  NaN `ori` (+derived cols) on flipped chunks

  Flip policy (see `resolve_flip_chunks`): a real ~180 deg flip toggles the true
  orientation state, so the correct chunks are one of two alternating-parity
  sets. We pick the parity using **forward-velocity alignment** — during forward
  walking the body orientation aligns with the direction of motion (a courting
  male does not chase by walking backward), so the correct parity is the one
  where `cos(ori - heading)` is positive on moving frames. If no chunk has
  enough sustained forward motion to decide, we **fall back to anchoring the
  first chunk as correct**. Flipped chunks are then *excluded* (orientation
  NaN'd), matching `load_flytracker_data(filter_ori=True)`.

  Bout video
  ----------
    - `save_bout_video`  extract a bout's video frames, overlay a heading arrow
      (colored by an angular-velocity column) on the focal fly + a target dot,
      and write an AVI. Promoted from `analyses/p1_levels/src/plot_gain.py`.

Angle conventions
-----------------
FlyTracker's raw `ori` is in math convention (CCW positive, y-up), while video
images use y-down. The on-image heading angle is therefore `-ori` (verified
empirically against the direction of forward walking). `save_bout_video` takes
an `ori_image_sign` to switch between the *transformed* `ori` used downstream
(drawn directly, sign +1) and *raw* FlyTracker `ori` (sign -1).
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import libs.utils as util


# ---------------------------------------------------------------------------
# Head-tail flip detection
# ---------------------------------------------------------------------------
def wrapped_ori_jump_deg(ori):
    """Frame-to-frame absolute orientation change, wrapped to [0, 180] degrees.

    Invariant to a global sign flip of `ori` (so it works on either FlyTracker's
    raw orientation or a negated `ori`). The first frame is 0.
    """
    ori = np.asarray(ori, dtype=float)
    d = np.abs(np.angle(np.exp(1j * np.diff(ori))))
    return np.concatenate([[0.0], np.degrees(d)])


def detect_headtail_flips(df, flip_threshold_deg=150.0, ori_col='ori', id_col='id'):
    """Flag head-tail flips: frames where the orientation jumps ~180 degrees
    from the previous frame (a FlyTracker head/tail swap).

    The detector is applied per fly id on the `ori_col` column and is sign-flip
    invariant. In processed data, `filter_ori=True` has usually already NaN'd
    the unreliable frames where FlyTracker flips, so counts are low there; run
    on raw (filter_ori=False) tracking to see the underlying flips.

    Adds a boolean `headtail_flip` column (per id) and returns the df.
    """
    df = df.copy()
    df['headtail_flip'] = False
    for _, g in df.groupby(id_col):
        g = g.sort_values('frame')
        flip = wrapped_ori_jump_deg(g[ori_col].values) > flip_threshold_deg
        df.loc[g.index, 'headtail_flip'] = flip
    return df


def find_flip_window(flydf, n_frames=20, flip_threshold_deg=150.0, ori_col='ori',
                     courtship_col=None):
    """Find the `n_frames`-long window of one fly's track containing the most
    head-tail flips. Returns (f_start, n_flips_in_window).

    If `courtship_col` names a 0/1 column present in `flydf`, the search is
    restricted to windows that lie entirely within a courtship bout (and only
    flips on courtship frames are counted), so the montage shows flips that
    actually occur while the pair is courting. Falls back to the unrestricted
    densest window if no fully-courtship window exists.
    """
    g = flydf.sort_values('frame').reset_index(drop=True)
    flip = (wrapped_ori_jump_deg(g[ori_col].values) > flip_threshold_deg).astype(int)
    use_court = courtship_col is not None and courtship_col in g.columns
    if len(flip) < n_frames:
        if use_court:
            flip = flip * g[courtship_col].fillna(0).values.astype(int)
        return int(g['frame'].iloc[0]), int(flip.sum())
    ones = np.ones(n_frames, int)
    csum = np.convolve(flip, ones, 'valid')  # unrestricted flip count per window
    if use_court:
        court = g[courtship_col].fillna(0).values.astype(int)
        court_flip = np.convolve(flip * court, ones, 'valid')  # courtship-only flips
        wcourt = np.convolve(court, ones, 'valid')             # courtship coverage
        court_csum = np.where(wcourt == n_frames, court_flip, -1)
        if court_csum.max() >= 0:  # at least one fully-courtship window exists
            csum = court_csum      # else fall back to the unrestricted count
    k = int(csum.argmax())
    return int(g['frame'].iloc[k]), int(max(csum[k], 0))


# ---------------------------------------------------------------------------
# Head-tail flip resolution (which chunks are correct vs flipped)
# ---------------------------------------------------------------------------
def resolve_flip_chunks(flydf, fps=60, speed_thresh_px_s=20.0, min_moving_frames=10,
                        flip_threshold_deg=150.0, ori_col='ori'):
    """Decide which orientation chunks (between head-tail flips) are correct.

    Head-tail flips split a track into contiguous chunks. We classify EACH
    chunk independently by forward-velocity alignment: during forward walking
    the body orientation aligns with the direction of motion, so on moving
    frames `cos(ori - heading)` is positive for a correctly-oriented chunk and
    negative for a flipped one (the fly would appear to walk backward). A chunk
    with enough sustained motion is "decidable"; chunks with too little motion
    inherit the nearest decidable chunk's state. If *no* chunk has enough motion
    to decide, we fall back to anchoring the first chunk as correct and
    alternating at each detected flip.

    We deliberately do NOT assume each detected flip is a real persistent toggle
    (i.e. we do not force an alternating-parity labeling from a single anchor):
    spurious flip-and-back jumps are common, and that assumption would mislabel
    a correctly-oriented chunk as flipped. Per-chunk classification is robust to
    such jumps.

    Args:
        flydf: one fly's tracking df with `frame`, `pos_x`, `pos_y`, `ori_col`.
        fps: frames per second (scales displacement to px/s).
        speed_thresh_px_s: only frames moving faster than this inform the vote.
        min_moving_frames: a chunk needs this many fast frames to be "decidable".
        flip_threshold_deg: ~180 deg jump threshold for a flip.
        ori_col: orientation column name (radians, math convention).

    Returns:
        (df, summary) where df has added integer `chunk_id` and boolean
        `ori_flipped` columns (sorted by frame), and summary is a dict with
        n_chunks, n_flipped_chunks, parity_source ('velocity'|'start_anchor'),
        and frac_frames_flipped.
    """
    df = flydf.sort_values('frame').reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        df['chunk_id'] = pd.Series([], dtype=int)
        df['ori_flipped'] = pd.Series([], dtype=bool)
        return df, {'n_chunks': 0, 'n_flipped_chunks': 0,
                    'parity_source': 'start_anchor', 'frac_frames_flipped': 0.0}

    ori = df[ori_col].values.astype(float)

    # Chunk boundaries at each detected flip.
    flip = wrapped_ori_jump_deg(ori) > flip_threshold_deg
    chunk_id = np.cumsum(flip.astype(int))  # increments at each flip frame
    df['chunk_id'] = chunk_id

    # Heading from frame-to-frame displacement, in the same (math, y-up)
    # convention as `ori`: image y is down, so negate dy. During forward
    # walking, cos(ori - heading) ~ +1.
    dx = np.diff(df['pos_x'].values, prepend=df['pos_x'].values[0])
    dy = np.diff(df['pos_y'].values, prepend=df['pos_y'].values[0])
    speed = np.hypot(dx, dy) * fps
    heading = np.arctan2(-dy, dx)
    align = np.cos(ori - heading)  # +1 forward, -1 backward (flipped)

    # Classify EACH chunk by its own forward-velocity alignment: a chunk is
    # flipped if its moving frames are anti-aligned with the direction of
    # motion (the fly would be walking backward). We deliberately do NOT assume
    # each detected flip is a real persistent toggle and alternate parity from
    # an anchor — a spurious flip-and-back jump would then force a perfectly
    # well-oriented chunk to be labeled flipped (observed on real data: a fly
    # with two chunks both aligned at +0.65 / +0.85 had its long correct chunk
    # mislabeled by the alternating rule).
    decided = {}
    for cid, g in df.groupby('chunk_id'):
        idx = g.index.values
        moving = speed[idx] > speed_thresh_px_s
        if moving.sum() < min_moving_frames:
            continue
        mean_align = np.nanmean(align[idx][moving])
        if not np.isnan(mean_align):
            decided[cid] = bool(mean_align < 0)

    chunk_ids = list(df['chunk_id'].unique())
    flipped_by_chunk = {}
    if decided:
        parity_source = 'velocity'
        # Decidable chunks use their own velocity sign; undecidable chunks
        # (too little motion to tell) are KEPT (not flipped) — we only NaN a
        # chunk on positive backward-motion evidence, never on assumption.
        # Flipped-but-stationary chunks are downstream-filtered by velocity
        # anyway, so conservatively keeping them is the safe choice.
        for cid in chunk_ids:
            flipped_by_chunk[cid] = decided.get(cid, False)
    else:
        # Nothing is decidable (e.g. a fly that never moves enough): fall back
        # to anchoring the first chunk as correct and alternating at each flip.
        parity_source = 'start_anchor'
        for cid in chunk_ids:
            flipped_by_chunk[cid] = (cid % 2) != 0

    df['ori_flipped'] = df['chunk_id'].map(flipped_by_chunk).astype(bool)

    summary = {
        'n_chunks': int(df['chunk_id'].nunique()),
        'n_flipped_chunks': int(df.loc[df['ori_flipped'], 'chunk_id'].nunique()),
        'parity_source': parity_source,
        'frac_frames_flipped': float(df['ori_flipped'].mean()),
    }
    return df, summary


def exclude_flipped_orientation(flydf, cols=('ori',), fps=60, **kwargs):
    """NaN orientation (and any ori-derived `cols`) on flipped chunks.

    Runs `resolve_flip_chunks`, then sets the requested columns to NaN wherever
    `ori_flipped` is True. This is a velocity-based alternative to
    `load_flytracker_data(filter_ori=True)` (which NaNs `ori` wherever wing info
    is missing): it keeps FlyTracker's body-axis orientation and discards only
    the head/tail-polarity-flipped chunks.

    NOTE: `resolve_flip_chunks` sorts by frame and resets the index, so the
    returned df is re-indexed. For modifying a multi-fly df *in place* without
    disturbing its index (e.g. before a transform that aligns on index), use
    `nan_flipped_orientation_per_id` instead.

    Args:
        flydf: one fly's tracking df.
        cols: columns to NaN on flipped chunks (default just 'ori'; pass e.g.
            ('ori', 'ang_vel_fly', 'theta_error') to drop derived metrics too).
        fps, **kwargs: forwarded to `resolve_flip_chunks`.

    Returns:
        (df, summary) — df with flipped-chunk orientation NaN'd.
    """
    df, summary = resolve_flip_chunks(flydf, fps=fps, **kwargs)
    present = [c for c in cols if c in df.columns]
    if present:
        df.loc[df['ori_flipped'], present] = np.nan
    return df, summary


# Orientation head/tail-polarity handling methods, shared across FlyTracker
# analyses (see `resolve_orientation`).
ORI_METHODS = ('velocity', 'wing', 'none')
WING_COLS = ('wing_l_x', 'wing_l_y', 'wing_r_x', 'wing_r_y')


def resolve_orientation(trk, method='velocity', fps=60, id_col='id',
                        wing_cols=WING_COLS, verbose=False, **kwargs):
    """Apply head/tail orientation-polarity handling to RAW FlyTracker tracking.

    This is the shared entry point for deciding how much to trust FlyTracker's
    body-axis orientation. It is generic to *any* FlyTracker dataset, not just
    the 2x2 strain assay: call it on freshly loaded tracking
    (`load_flytracker_data(..., filter_ori=False)`, `ori` NOT yet negated),
    BEFORE the `ori = -1*ori` convention flip and the relative-metrics transform.

    Methods
    -------
    'velocity' (default):
        Keep FlyTracker's body-axis `ori` and NaN only the chunks whose
        orientation is anti-aligned with the direction of motion — a genuine
        head-tail flip — per fly (via `nan_flipped_orientation_per_id`).
        Assumes the fly translates *forward*, not backward or sideways. This
        holds well for walking/chasing flies (a courting male does not chase by
        walking backward) and recovers wing-missing-but-correctly-oriented
        frames that 'wing' discards.
    'wing':
        The conservative legacy rule — NaN `ori` wherever FlyTracker did not
        detect wings (all `wing_cols` NaN; equivalent to
        `load_flytracker_data(filter_ori=True)`). Makes NO motion assumption,
        but discards many valid frames, and its data loss is behaviour-correlated
        (wing detection tracks wing extension / song), which can bias
        orientation-based metrics.
    'none':
        Trust FlyTracker's raw `ori` unchanged (no filtering, no flip
        resolution). Makes no assumptions; keeps everything.

    Choosing a method
    -----------------
    'velocity' is the right default for courtship / pursuit assays where the
    focal fly walks toward its target. Prefer 'wing' or 'none' for assays with
    substantial **sideways or backward** motion, where the forward-motion
    assumption is weak and 'velocity' could mislabel a chunk.

    A future MIDDLE-GROUND option could keep 'velocity' but ignore
    near-perpendicular (sideways) frames when scoring each chunk's alignment, so
    occasional sideways motion does not influence the call while real flips are
    still caught. It is intentionally NOT implemented yet: the per-chunk *mean*
    alignment already down-weights sporadic sideways frames (cos ~ 0), so the
    extra knob only helps when a chunk is *predominantly* sideways — we add it
    when that case is actually observed in data, rather than speculatively.

    Args:
        trk: raw FlyTracker tracking df (needs `ori`; 'velocity' also needs
            `frame`, `pos_x`, `pos_y`; 'wing' needs the `wing_cols`).
        method: one of `ORI_METHODS`.
        fps, id_col, **kwargs: forwarded to `nan_flipped_orientation_per_id`
            for the 'velocity' method.
        wing_cols: wing columns whose joint absence defines "no wing info".
        verbose: print a one-line summary of how much `ori` was NaN'd.

    Returns:
        (trk, info) — `trk` with `ori` NaN'd per the method; `info` dict with
        keys 'method', 'frac_ori_nan', and (velocity only) 'per_fly' flip
        summaries.
    """
    if method not in ORI_METHODS:
        raise ValueError("method must be one of {}, got {!r}".format(ORI_METHODS, method))
    trk = trk.copy()
    info = {'method': method, 'per_fly': None}
    n_before = int(trk['ori'].notna().sum())

    if method == 'velocity':
        trk, summaries = nan_flipped_orientation_per_id(
            trk, id_col=id_col, cols=('ori',), fps=fps, **kwargs)
        info['per_fly'] = summaries
    elif method == 'wing':
        have = [c for c in wing_cols if c in trk.columns]
        if len(have) == len(wing_cols):
            no_wing = trk[list(wing_cols)].isna().sum(axis=1) == len(wing_cols)
            trk.loc[no_wing, 'ori'] = np.nan
        elif verbose:
            print("resolve_orientation('wing'): wing columns {} missing; ori unchanged"
                  .format([c for c in wing_cols if c not in trk.columns]))
    # 'none': leave ori as-is.

    n_after = int(trk['ori'].notna().sum())
    info['frac_ori_nan'] = float(1 - n_after / n_before) if n_before else 0.0
    if verbose:
        print("resolve_orientation(method={}): NaN'd {:.2%} of valid-ori frames".format(
            method, info['frac_ori_nan']))
    return trk, info


def nan_flipped_orientation_per_id(df, id_col='id', cols=('ori',), fps=60, **kwargs):
    """Per fly id, resolve head-tail flip chunks (velocity-based) and NaN `cols`
    on flipped chunks, preserving the original index and row order.

    This is the multi-fly entry point used by the processing pipeline: it lets
    you load FlyTracker tracking with `filter_ori=False` (keeping body-axis
    orientation even when wings are undetected) and then discard only the
    genuinely flipped chunks, instead of NaN-ing orientation wherever wings are
    missing.

    Requires `frame` and `pos_x`/`pos_y` columns (per `resolve_flip_chunks`).

    Returns:
        (df, summaries) where df has `cols` NaN'd on flipped chunks (index
        preserved) and summaries maps fly id -> the `resolve_flip_chunks` summary.
    """
    df = df.copy()
    summaries = {}
    present = [c for c in cols if c in df.columns]
    for fid, g in df.groupby(id_col):
        res, s = resolve_flip_chunks(g, fps=fps, **kwargs)
        summaries[fid] = s
        flipped_frames = set(res.loc[res['ori_flipped'], 'frame'].astype(int))
        if flipped_frames and present:
            mask = (df[id_col] == fid) & (df['frame'].isin(flipped_frames))
            df.loc[mask, present] = np.nan
    return df, summaries


# ---------------------------------------------------------------------------
# Flip montage (overlay heading on video frames around a flip cluster)
# ---------------------------------------------------------------------------
def plot_flip_montage(acqdir, fly_id=None, f_start=None, n_frames=20,
                      flip_threshold_deg=150.0, pad=55, arrow_len=15, fps=60,
                      ncols=5, ori_image_sign=-1, resolve_chunks=True,
                      savepath=None, subfolder='*', trk=None, courtship_col=None):
    """Overlay each fly's FlyTracker heading on consecutive video frames around a
    cluster of head-tail flips, as a single montage figure (no per-frame PNGs).

    Uses *raw* tracking (filter_ori=False) so the flips are visible (the
    processed parquet has already NaN'd unreliable orientations). The heading
    arrow points from the centroid toward FlyTracker's reported head; on a flip
    frame it reverses ~180 degrees. Flip frames are titled in red, and (if
    `resolve_chunks`) each frame is tagged correct/FLIPPED from
    `resolve_flip_chunks`.

    Args:
        acqdir: acquisition directory (contains movie + FlyTracker output).
        fly_id: fly to inspect (default: fly with the densest flip window).
        f_start: first frame (default: that fly's densest flip window).
        n_frames: number of consecutive frames to plot.
        pad: pixels of padding around the fly's bounding box for the crop.
        arrow_len: heading arrow length in pixels.
        ori_image_sign: on-image heading angle = ori_image_sign * ori. Raw
            FlyTracker ori uses -1 (default).
        resolve_chunks: also classify each frame's chunk correct/flipped.
        savepath: if given, save the montage png here.
        subfolder: subfolder glob passed to load_flytracker_data.
        trk: optional pre-loaded raw tracking df (e.g. already truncated at
            copulation). If None, raw tracking is loaded from `acqdir`.
        courtship_col: if set, restrict the flip-window search (both the auto
            fly pick and `f_start`) to windows within a courtship bout; see
            `find_flip_window`.

    Returns:
        dict with fly_id, f_start, n_flips, fig, savepath, and (if resolved) the
        per-fly flip summary.
    """
    import cv2
    if trk is None:
        _, trk, _ = util.load_flytracker_data(
            acqdir, fps=fps, calib_is_upstream=False, subfolder=subfolder, filter_ori=False)

    # Pick the fly with the densest flip window if not specified.
    if fly_id is None:
        best = None
        for fid, g in trk.groupby('id'):
            fs, nf = find_flip_window(g, n_frames=n_frames, flip_threshold_deg=flip_threshold_deg,
                                      courtship_col=courtship_col)
            if best is None or nf > best[2]:
                best = (fid, fs, nf)
        fly_id, _, _ = best
    g = trk[trk['id'] == fly_id].sort_values('frame').reset_index(drop=True)
    if f_start is None:
        f_start, _ = find_flip_window(g, n_frames=n_frames, flip_threshold_deg=flip_threshold_deg,
                                      courtship_col=courtship_col)

    seg = g[(g['frame'] >= f_start) & (g['frame'] < f_start + n_frames)].copy()
    jumps = wrapped_ori_jump_deg(seg['ori'].values)
    seg['is_flip'] = jumps > flip_threshold_deg
    n_flips = int(seg['is_flip'].sum())

    flip_summary = None
    if resolve_chunks:
        resolved, flip_summary = resolve_flip_chunks(
            g, fps=fps, flip_threshold_deg=flip_threshold_deg)
        seg = seg.merge(resolved[['frame', 'ori_flipped']], on='frame', how='left')

    # Crop box around the fly over the window.
    x0 = max(int(seg['pos_x'].min()) - pad, 0)
    x1 = int(seg['pos_x'].max()) + pad
    y0 = max(int(seg['pos_y'].min()) - pad, 0)
    y1 = int(seg['pos_y'].max()) + pad

    # Read the consecutive frames.
    video_path = util.find_video(acqdir)
    cap = cv2.VideoCapture(video_path)
    frames = list(seg['frame'].values)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    imgs = {}
    for fr in range(int(frames[0]), int(frames[-1]) + 1):
        ret, img = cap.read()
        if not ret:
            break
        if fr in frames:
            imgs[fr] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap.release()

    nrows = int(np.ceil(len(frames) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')

    for i, (_, row) in enumerate(seg.iterrows()):
        ax = axes[i]
        fr = int(row['frame'])
        if fr not in imgs:
            continue
        ax.imshow(imgs[fr])
        px, py, ori = row['pos_x'], row['pos_y'], row['ori']
        theta_img = ori_image_sign * ori
        dx, dy = arrow_len * np.cos(theta_img), arrow_len * np.sin(theta_img)
        ax.arrow(px, py, dx, dy, color='yellow', width=1.2, head_width=5,
                 length_includes_head=True, zorder=5)
        ax.plot(px, py, 'o', color='cyan', ms=3, zorder=6)  # centroid
        ax.set_xlim([x0, x1])
        ax.set_ylim([y1, y0])  # y down
        tags = []
        if bool(row['is_flip']):
            tags.append('FLIP')
        if 'ori_flipped' in seg.columns and bool(row.get('ori_flipped', False)):
            tags.append('flipped-chunk')
        ax.set_title('f{}  {:.0f}°{}'.format(
            fr, np.degrees(theta_img), ('  ' + ' '.join(tags)) if tags else ''),
            color='red' if tags else 'white', fontsize=8)

    sex = seg['sex'].iloc[0] if 'sex' in seg.columns else '?'
    court_tag = ' | courtship bout' if courtship_col is not None else ''
    fig.suptitle('Head-tail flips on video | id {} ({}) | frames {}-{} | {} flips{}'.format(
        fly_id, sex, f_start, f_start + n_frames - 1, n_flips, court_tag), y=1.0)
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')

    return {'fly_id': fly_id, 'f_start': f_start, 'n_flips': n_flips,
            'fig': fig, 'savepath': savepath, 'flip_summary': flip_summary}


# ---------------------------------------------------------------------------
# QC: theta_error from orientation vs from heading
# ---------------------------------------------------------------------------
def recompute_theta_error_heading(df, los_col='abs_ang_between', heading_col='heading'):
    """Recompute heading-based theta_error from the stored line-of-sight angle
    and heading: a one-liner you can apply to an existing (processed) df.

        df['theta_error_heading'] = qc.recompute_theta_error_heading(df)
        # equivalently: util.circular_distance(df['abs_ang_between'], df['heading'])

    Returns the heading-based theta_error as a Series (radians).
    """
    return util.circular_distance(df[los_col], df[heading_col])


def plot_theta_error_ori_vs_heading(df, ori_col='theta_error',
                                    heading_col='theta_error_heading',
                                    in_degrees=True, lim=180, title=None,
                                    rasterized=True):
    """Compare the orientation-based vs heading-based theta_error for a dataset.

    `theta_error` (the analysis default) is measured from the fly's body-axis
    **orientation**; `theta_error_heading` is measured from its **heading**
    (direction of motion). They diverge when the fly sideslips. This QC scatters
    one against the other (with a unity line + Pearson r) and shows the
    difference distribution — a sanity check that the two largely agree and that
    orientation (the principled choice for retinal/steering error) is sound.

    Generic to any FlyTracker dataset; pass an already-filtered df (e.g. moving
    frames of one acquisition). Reads the base radian columns and converts to
    degrees here.

    NOTE: `theta_error_heading` is QC-only — it is not maintained or used in
    downstream analyses, so the value stored in older processed parquets predates
    fixes to the transform, and the heading derivation itself (smoothed
    frame-to-frame position differencing) has not been validated for analysis
    use. If you decide to rely on this column, recompute it fresh rather than
    trusting the stored value (and consider revisiting how `heading` is
    computed): `df['theta_error_heading'] = recompute_theta_error_heading(df)`.

    Args:
        df: df with `ori_col` and `heading_col` (radians if `in_degrees`).
        ori_col, heading_col: column names.
        in_degrees: convert radians -> degrees for display.
        lim: axis limit (degrees) for the unity line / square aspect.
        title: optional suptitle.

    Returns:
        (fig, r) — figure and Pearson correlation; (None, nan) if `heading_col`
        is absent or there are no valid paired points.
    """
    if heading_col not in df.columns or ori_col not in df.columns:
        print("plot_theta_error_ori_vs_heading: missing {!r}/{!r}; skipping.".format(
            ori_col, heading_col))
        return None, float('nan')

    conv = np.rad2deg if in_degrees else (lambda x: x)
    te_ori = conv(df[ori_col].values.astype(float))
    te_hdg = conv(df[heading_col].values.astype(float))
    valid = ~np.isnan(te_ori) & ~np.isnan(te_hdg)
    if valid.sum() < 2:
        print("plot_theta_error_ori_vs_heading: <2 valid points; skipping.")
        return None, float('nan')
    te_ori, te_hdg = te_ori[valid], te_hdg[valid]
    r = float(np.corrcoef(te_ori, te_hdg)[0, 1])
    unit = '°' if in_degrees else ' (rad)'

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    ax.scatter(te_ori, te_hdg, s=2, alpha=0.15, rasterized=rasterized)
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=0.8, label='unity')
    ax.set_xlabel('θ error from orientation{}'.format(unit))
    ax.set_ylabel('θ error from heading{}'.format(unit))
    ax.set_title('Correlation: r = {:.3f}'.format(r))
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    ax = axes[1]
    diff = te_ori - te_hdg
    ax.hist(diff, bins=100, color='steelblue', edgecolor='none')
    ax.axvline(0, color='r', ls='--', lw=0.8)
    ax.set_xlabel('Orientation − Heading θ error{}'.format(unit))
    ax.set_ylabel('Count')
    ax.set_title('Difference (mean={:.1f}, std={:.1f})'.format(diff.mean(), diff.std()))

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig, r


# ---------------------------------------------------------------------------
# Bout video (heading arrow + target dot overlaid on extracted frames)
# ---------------------------------------------------------------------------
def _angvel_to_bgr(value, vmax):
    """Map a (signed) value to a BGR color using the coolwarm colormap.

    blue = negative / leftward, red = positive / rightward.
    """
    import matplotlib.cm as cm
    norm_val = np.clip(value / vmax, -1.0, 1.0)   # [-1, 1]
    rgba = cm.coolwarm((norm_val + 1.0) / 2.0)    # [0, 1]
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return (b, g, r)   # OpenCV uses BGR


def _resolve_video_path(video_or_acq_dir):
    """Return a playable video path from a direct path, an acq dir, or one of its
    immediate subdirs (FlyTracker output is often one level down)."""
    if video_or_acq_dir is None:
        return None
    if os.path.isfile(video_or_acq_dir):
        return video_or_acq_dir
    vp = util.find_video(video_or_acq_dir)
    if vp is not None:
        return vp
    for sd in glob.glob(os.path.join(video_or_acq_dir, '*')):
        vp = util.find_video(sd)
        if vp is not None:
            return vp
    return None


def save_bout_video(video_or_acq_dir, flydf, f_start, f_end, out_path,
                    targdf=None, color_col='ang_vel_fly_shifted_deg', ori_col='ori',
                    fps=60, output_fps=15, arrow_len=25, vmax=None,
                    ori_image_sign=1):
    """Extract a bout's video frames, overlay a heading arrow on the focal fly
    (colored by `color_col`) and a target dot, and write an AVI.

    Promoted from `analyses/p1_levels/src/plot_gain.py:_save_qc_video` and
    generalized so multiple analyses share one implementation.

    Args:
        video_or_acq_dir: video file, acquisition dir, or parent containing one.
        flydf: focal fly df with `frame`, `pos_x`, `pos_y`, `ori_col`, `color_col`.
        f_start, f_end: inclusive frame range of the bout.
        out_path: output AVI path (parent dirs are created).
        targdf: optional target df with `frame`, `pos_x`, `pos_y` (white dot).
        color_col: signed column used to color the arrow (coolwarm; blue=neg,
            red=pos). If absent/NaN, the arrow is drawn at the neutral color.
        ori_col: orientation column (radians).
        fps: source frame rate (unused except for documentation parity).
        output_fps: playback frame rate of the written AVI.
        arrow_len: heading arrow length in pixels.
        vmax: color scale max (default: 95th pct of |color_col| over the bout).
        ori_image_sign: on-image heading angle = ori_image_sign * ori. Use +1
            for transformed `ori` (default), -1 for raw FlyTracker `ori`.

    Returns:
        out_path on success, else None.
    """
    import cv2

    video_path = _resolve_video_path(video_or_acq_dir)
    if video_path is None:
        print("  WARNING: No video found for {}".format(video_or_acq_dir))
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  WARNING: Could not open video {}".format(video_path))
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter(out_path, fourcc, output_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print("  WARNING: VideoWriter failed to open for {}".format(out_path))
        cap.release()
        return None

    fly_cols = ['pos_x', 'pos_y', ori_col]
    has_color = color_col in flydf.columns
    if has_color:
        fly_cols = fly_cols + [color_col]
    fly_by_frame = flydf.set_index('frame')[fly_cols].to_dict('index')

    targ_by_frame = {}
    if targdf is not None and not targdf.empty:
        targ_by_frame = targdf.set_index('frame')[['pos_x', 'pos_y']].to_dict('index')

    if vmax is None and has_color:
        vmax = float(np.nanpercentile(np.abs(flydf[color_col].values), 95))
    if not vmax or np.isnan(vmax):
        vmax = 1.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_start))
    n_written = 0
    for fr in range(int(f_start), int(f_end) + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        row = fly_by_frame.get(fr)
        if row is not None and not np.isnan(row['pos_x']) and not np.isnan(row[ori_col]):
            cx = int(round(row['pos_x']))
            cy = int(round(row['pos_y']))
            theta_img = ori_image_sign * row[ori_col]
            tip_x = int(round(cx + arrow_len * np.cos(theta_img)))
            tip_y = int(round(cy + arrow_len * np.sin(theta_img)))
            cval = row.get(color_col, 0.0) if has_color else 0.0
            color = _angvel_to_bgr(0.0 if (cval is None or np.isnan(cval)) else cval, vmax)
            cv2.arrowedLine(frame, (cx, cy), (tip_x, tip_y),
                            color=color, thickness=2, tipLength=0.3)
            cv2.circle(frame, (cx, cy), radius=4, color=color, thickness=-1)

        trow = targ_by_frame.get(fr)
        if trow is not None and not np.isnan(trow['pos_x']):
            tx = int(round(trow['pos_x']))
            ty = int(round(trow['pos_y']))
            cv2.circle(frame, (tx, ty), radius=6, color=(255, 255, 255), thickness=2)

        writer.write(frame)
        n_written += 1

    writer.release()
    cap.release()
    print("  Saved video: {} ({} frames @ {}fps, {:.1f}s, vmax={:.1f})".format(
        out_path, n_written, output_fps, n_written / output_fps, vmax))
    return out_path
