#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steering gain analysis for the calibration (P1 levels) dataset.

Sign conventions (applied after loading raw transform data)
-----------------------------------------------------------
**theta_error** is computed from the fly's *orientation* (body axis), NOT
heading (direction of motion).  Both are computed during the transform step
(``relative_metrics.calculate_theta_error`` uses ``ori``;
``calculate_theta_error_from_heading`` uses ``heading``), but the gain
analyses consistently use the orientation-based version.

Raw transform convention (CW-positive, due to FlyTracker image coords y-down):
  theta_error > 0 → target is CW  from the fly's midline (fly's RIGHT)
  theta_error < 0 → target is CCW from the fly's midline (fly's LEFT)
  ang_vel_fly > 0 → CW rotation  (rightward turn)
  ang_vel_fly < 0 → CCW rotation (leftward turn)

These already match the desired display convention (positive = fly's RIGHT),
so no negation is applied — matching the gain/ analysis folder.  The
corrective steering response appears in the upper-right (target right → turn
right) and lower-left (target left → turn left) quadrants.

Progressive motion: target moving *away* from the midline (|theta_error|
increases).  Regressive motion: target moving *toward* the midline
(|theta_error| decreases).

QC Plot A: Heading-vs-orientation theta_error correlation for one example acq.
QC Plot B: Example courtship bouts with leftward / rightward turning bias.

Plot 1: Gain curves (angular velocity vs. binned theta_error) per LED level.
  Rows = species, cols = LED level, progressive/regressive overlaid.

Plot 2: Gain per species with hue = LED level (overview).

Plot 3: Both species on same axes per LED level, progressive/regressive split.
"""
#%%
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import libs.utils as util
import libs.plotting as putil
import libs.qc as qc
from analyses.gain.src import gain_funcs as gf
from analyses.p1_levels.src.load_calibration_data import (
    load_and_prepare_dataset, PLOT_STYLE, MIN_FONTSIZE, SPECIES_PALETTE,
    DEFAULT_ROOTDIR,
)

# ============================================================
# SETTINGS
# ============================================================
#%%
putil.set_sns_style(PLOT_STYLE, min_fontsize=MIN_FONTSIZE)
bg_color = [0.7] * 3 if PLOT_STYLE == 'dark' else 'w'
species_palette = SPECIES_PALETTE.copy()

fps = 60
lag_frames = 12  # ~200ms at 60fps
vel_lim = 10
deg_lim = 180
bin_size = 20
max_ang_vel_deg = 500  # filter unrealistic angular velocity (tracking artifacts)

# ============================================================
# LOAD DATA
# ============================================================
#%%
data = load_and_prepare_dataset(create_new=False)
df_all = data['df_filtered']
block_keys = data['block_keys']
basedir = data['basedir']
min_courtship_frac = data['min_courtship_frac']

figdir = os.path.join(basedir, 'p1_levels', 'gain')
os.makedirs(figdir, exist_ok=True)
print(f"Saving figures to: {figdir}")

figid = 'p1_levels_gain'

#%%
# Reassign pr_direction using the deterministic position-based rule.
# The velocity-based assignment used during loading is too noisy frame-by-frame.
# Raw targ_pos_theta: positive = fly's RIGHT, negative = fly's LEFT
# (FlyTracker image coords, CW-positive).
if 'stim_direction' in df_all.columns and df_all['stim_direction'].notna().any():
    df_all = gf.assign_progressive_regressive_from_cw_ccw(
        df_all, stim_direction_var='stim_direction')
else:
    # CCW stimulus: target on fly's LEFT (theta < 0) is receding → progressive;
    # target on fly's RIGHT (theta > 0) is approaching → regressive.
    df_all['pr_direction'] = None
    df_all.loc[df_all['targ_pos_theta'] < 0, 'pr_direction'] = 'progressive'
    df_all.loc[df_all['targ_pos_theta'] > 0, 'pr_direction'] = 'regressive'

print(df_all['pr_direction'].unique())


# ============================================================
# PREPARE: courtship-matched blocks, shift ang_vel, bin theta_error
# ============================================================
#%%

def detect_tracking_jumps(df, grouper='file_name', max_jump_px=30, margin=2):
    """Flag frames with implausibly large position jumps (tracking errors).

    Uses two criteria:
    1. Frame-to-frame displacement > max_jump_px pixels
    2. Fly position within 20px of target position (identity swap)

    Marks bad frames (plus `margin` surrounding frames) as NaN in
    position/orientation columns.

    Returns the DataFrame with a boolean column 'bad_tracking' (True = bad).
    """
    df = df.sort_values([grouper, 'frame']).copy()

    # Criterion 1: large position jumps
    df['_dx'] = df.groupby(grouper)['pos_x'].diff()
    df['_dy'] = df.groupby(grouper)['pos_y'].diff()
    df['_displacement'] = np.sqrt(df['_dx']**2 + df['_dy']**2)
    bad_jump = df['_displacement'] > max_jump_px

    # Criterion 2: fly position very close to target (identity swap)
    bad_swap = pd.Series(False, index=df.index)
    if 'targ_centered_to_focal_x' in df.columns:
        dist_to_targ = np.sqrt(df['targ_rel_pos_x']**2 + df['targ_rel_pos_y']**2)
        bad_swap = dist_to_targ < 5  # < 5 pixels in egocentric coords

    bad = bad_jump | bad_swap
    for shift_n in range(1, margin + 1):
        bad = bad | bad.shift(shift_n, fill_value=False)
        bad = bad | bad.shift(-shift_n, fill_value=False)

    df['bad_tracking'] = bad
    n_bad = bad.sum()
    pct = n_bad / len(df) * 100
    n_jump = bad_jump.sum()
    n_swap = bad_swap.sum()
    print(f"  Tracking jumps detected: {n_bad} frames ({pct:.2f}%) "
          f"[jumps>{max_jump_px}px: {n_jump}, near-target: {n_swap}, "
          f"margin: ±{margin}]")

    nan_cols = ['pos_x', 'pos_y', 'ori', 'theta_error', 'targ_pos_theta',
                'ang_vel_fly', 'ang_vel_fly_smoothed', 'targ_rel_pos_x',
                'targ_rel_pos_y', 'theta_error_deg']
    nan_cols = [c for c in nan_cols if c in df.columns]
    df.loc[bad, nan_cols] = np.nan

    df.drop(columns=['_dx', '_dy', '_displacement'], inplace=True)
    return df


f1 = df_all[df_all['id'] == 0].copy()
f1 = f1.merge(block_keys, on=['acquisition', 'led_intensity', 'speed_hz'],
              how='inner')
print(f"Courtship-matched frames (all, id=0): {len(f1)}")
print(f1.groupby(['species'])['acquisition'].nunique())

# Filter out tracking jumps (fly pos jumps to target pos)
f1 = detect_tracking_jumps(f1, grouper='file_name', max_jump_px=30, margin=2)

# Shift ang_vel_fly by lag (uses the standard transform ang_vel_fly from ori)
f1 = util.shift_variables_by_lag(f1, file_grouper='file_name',
                                  lag=lag_frames)

# Convert to degrees — no negation needed.  FlyTracker image coords (y-down)
# make CW-positive, so the raw values already have positive = fly's RIGHT
# and positive ang_vel = rightward turn, matching the gain/ folder.
f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])
f1['ang_vel_fly_shifted_deg'] = np.rad2deg(f1['ang_vel_fly_shifted'])

# Ensure bad-tracking frames are NaN in derived columns too
if 'bad_tracking' in f1.columns:
    bad_mask = f1['bad_tracking'].fillna(False)
    f1.loc[bad_mask, ['ang_vel_fly_shifted_deg', 'theta_error_deg']] = np.nan

f1 = gf.bin_by_object_position(f1, start_bin=-deg_lim, end_bin=deg_lim,
                                bin_size=bin_size)
f1['binned_theta_error_num'] = pd.to_numeric(f1['binned_theta_error'],
                                              errors='coerce')
f1.reset_index(drop=True, inplace=True)

chasedf = f1[
    (f1['theta_error_deg'].abs() < deg_lim)
    & (f1['vel'] > vel_lim)
    & (f1['ang_vel_fly_shifted_deg'].abs() < max_ang_vel_deg)
].copy()
chasedf.reset_index(drop=True, inplace=True)
print(f"Frames after ang_vel filter (<{max_ang_vel_deg} deg/s): {len(chasedf)}")


# ============================================================
# QC PLOT A: Heading vs. Orientation theta_error correlation
# ============================================================
#%%
_example_acqs = f1['acquisition'].unique()
if len(_example_acqs) > 0:
    _qc_acq = _example_acqs[0]
    _qc_data = f1[(f1['acquisition'] == _qc_acq) & (f1['vel'] > vel_lim)].copy()

    _has_heading = 'theta_error_heading' in _qc_data.columns
    if not _has_heading:
        print(f"theta_error_heading not in columns — skipping heading QC plot.")
    else:
        _te_ori_deg = np.rad2deg(_qc_data['theta_error'])
        _te_hdg_deg = np.rad2deg(_qc_data['theta_error_heading'])

        _valid = _te_ori_deg.notna() & _te_hdg_deg.notna()
        _r = np.corrcoef(_te_ori_deg[_valid], _te_hdg_deg[_valid])[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        ax = axes[0]
        ax.scatter(_te_ori_deg[_valid], _te_hdg_deg[_valid], s=2, alpha=0.15,
                   rasterized=True)
        lims = [-deg_lim, deg_lim]
        ax.plot(lims, lims, 'r--', lw=0.8, label='unity')
        ax.set_xlabel('θ error from orientation (°)')
        ax.set_ylabel('θ error from heading (°)')
        ax.set_title(f'Correlation: r = {_r:.3f}')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)

        ax = axes[1]
        _diff = _te_ori_deg[_valid] - _te_hdg_deg[_valid]
        ax.hist(_diff, bins=100, color='steelblue', edgecolor='none')
        ax.axvline(0, color='r', ls='--', lw=0.8)
        ax.set_xlabel('Orientation – Heading θ error (°)')
        ax.set_ylabel('Count')
        ax.set_title(f'Difference distribution (mean={_diff.mean():.1f}°, '
                     f'std={_diff.std():.1f}°)')

        fig.suptitle(f'θ error: Orientation vs Heading — {_qc_acq}',
                     fontsize=MIN_FONTSIZE + 1)
        plt.tight_layout()
        putil.label_figure(fig, figid)
        figname = 'qc_theta_error_ori_vs_heading'
        plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150,
                    bbox_inches='tight')
        print(f"Saved: {figname} (r={_r:.3f})")


# ============================================================
# QC PLOT B: Example courtship bouts — leftward / rightward turns
# ============================================================
#%%
n_per_direction = 2   # 2 leftward-dominant + 2 rightward-dominant
min_bout_sec = 1.0    # minimum bout duration in seconds
min_bout_frames = int(min_bout_sec * fps)

qc_species = 'Dmel'
qc_df = f1[(f1['species'] == qc_species) & (f1['courtship'] == 1)].copy()


def _find_turning_bouts(qc_df, direction='right', n_bouts=2,
                        min_bout_frames=120, ang_vel_thresh=10,
                        courtship_gap_tol=3):
    """Find long contiguous courtship bouts with a turning bias.

    Courtship frames separated by <= *courtship_gap_tol* frames are merged
    into the same bout (tolerates brief annotation gaps).  A gap larger than
    that starts a new bout.

    Among qualifying bouts, the longest ones that also meet the angular-
    velocity threshold are returned (preferring duration over extreme bias).

    Parameters
    ----------
    direction : 'left' or 'right'
    n_bouts : number of bouts to return
    min_bout_frames : minimum number of courtship frames in a bout
    ang_vel_thresh : mean ang_vel must exceed this (deg/s) in the given direction
    courtship_gap_tol : max gap (frames) between courtship frames that still
        counts as the same bout

    Returns list of (acquisition, bout_start, bout_end, n_frames, mean_ang_vel)
    tuples sorted by bout length (longest first).
    """
    results = []
    for acq, adf in qc_df.groupby('acquisition'):
        cdf = adf[adf['courtship'] == 1].sort_values('frame')
        if len(cdf) < min_bout_frames:
            continue

        # Merge courtship frames with small gaps into single bouts
        breaks = cdf['frame'].diff() > (1 + courtship_gap_tol)
        bout_id = breaks.cumsum()

        for _, bout in cdf.groupby(bout_id):
            n = len(bout)
            if n < min_bout_frames:
                continue
            mean_av = bout['ang_vel_fly_shifted_deg'].mean()
            if direction == 'right' and mean_av > ang_vel_thresh:
                results.append((acq, bout['frame'].iloc[0],
                                bout['frame'].iloc[-1], n, mean_av))
            elif direction == 'left' and mean_av < -ang_vel_thresh:
                results.append((acq, bout['frame'].iloc[0],
                                bout['frame'].iloc[-1], n, mean_av))

    # Prefer long bouts (clearer examples)
    results.sort(key=lambda x: x[3], reverse=True)
    return results[:n_bouts]


def _plot_qc_example(bout_data, targ_bout, qc_acq, example_idx, turn_label):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    mean_theta = bout_data['theta_error_deg'].mean()
    mean_av = bout_data['ang_vel_fly_shifted_deg'].mean()
    led_vals = bout_data['led_intensity'].dropna().unique()
    speed_vals = bout_data['speed_hz'].dropna().unique()
    species_val = bout_data['species'].iloc[0]
    n_frames = len(bout_data)
    dur_s = n_frames / fps

    info_text = (
        f"Acquisition: {qc_acq}\n"
        f"Species: {species_val}  |  Turn bias: {turn_label}\n"
        f"Frames: {bout_data['frame'].iloc[0]}–{bout_data['frame'].iloc[-1]}"
        f"  ({n_frames} frames, {dur_s:.1f} s)\n"
        f"Mean θ error: {mean_theta:.1f}° (+ = right)\n"
        f"Mean ω shifted: {mean_av:.1f}°/s (+ = rightward)\n"
        f"LED intensity: {led_vals}   Speed: {speed_vals} Hz"
    )

    # Egocentric display: fly faces UP, right on right.
    # targ_rel_pos_y > 0 → fly's RIGHT → positive plot-X
    # targ_rel_pos_x > 0 → in front     → positive plot-Y (top)
    ego_x = bout_data['targ_rel_pos_y']
    ego_y = bout_data['targ_rel_pos_x']

    # Symmetric colorbar limits centered at 0
    vmax_av = max(
        np.percentile(bout_data['ang_vel_fly_shifted_deg'].abs().dropna(), 95),
        1.0)

    # --- Panel 1: Egocentric, colored by frame ---
    ax = axes[0, 0]
    sc = ax.scatter(ego_x, ego_y, c=bout_data['frame'], cmap='viridis', s=15, alpha=0.7)
    ax.plot(0, 0, 'r^', markersize=14, zorder=5)
    ax.annotate('', xy=(0, 5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.set_xlabel('← Fly\'s left  |  Fly\'s right →')
    ax.set_ylabel('Fly\'s front ↑')
    ax.set_title('Target position (color=frame)')
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.axvline(0, color='gray', ls='--', lw=0.5)
    plt.colorbar(sc, ax=ax, label='Frame')
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=7, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', alpha=0.6,
                      facecolor='black' if PLOT_STYLE == 'dark' else 'white'))

    # --- Panel 2: Egocentric, colored by ang vel (diverging, centered at 0) ---
    ax = axes[0, 1]
    sc2 = ax.scatter(ego_x, ego_y,
                     c=bout_data['ang_vel_fly_shifted_deg'], cmap='coolwarm',
                     vmin=-vmax_av, vmax=vmax_av, s=15, alpha=0.7)
    ax.plot(0, 0, 'r^', markersize=14, zorder=5)
    ax.annotate('', xy=(0, 5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.set_xlabel('← Fly\'s left  |  Fly\'s right →')
    ax.set_ylabel('Fly\'s front ↑')
    ax.set_title('Target position (color=ω shifted)')
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.axvline(0, color='gray', ls='--', lw=0.5)
    cbar2 = plt.colorbar(sc2, ax=ax, label='ω shifted (°/s, +=rightward)')
    cbar2.ax.axhline(0.5, color='k', lw=0.8)

    # --- Panel 3: Arena view ---
    ax = axes[1, 0]
    ax.scatter(bout_data['pos_x'], bout_data['pos_y'],
               c=bout_data['frame'], cmap='viridis', s=10, alpha=0.5, label='Male')
    if not targ_bout.empty:
        ax.scatter(targ_bout['pos_x'], targ_bout['pos_y'],
                   c=targ_bout['frame'], cmap='magma', s=10, alpha=0.5,
                   marker='s', label='Target')
    step = max(1, len(bout_data) // 25)
    for i in range(0, len(bout_data), step):
        row = bout_data.iloc[i]
        dx = 2.0 * np.cos(row['ori'])
        dy = 2.0 * np.sin(row['ori'])
        ci = i / max(1, len(bout_data) - 1)
        ax.arrow(row['pos_x'], row['pos_y'], dx, dy,
                 head_width=0.4, head_length=0.2,
                 fc=plt.cm.viridis(ci), ec='k', lw=0.3)
    ax.set_xlabel('X (arena, px)')
    ax.set_ylabel('Y (arena, px)')
    ax.set_title('Arena view (viridis=male, magma=target)')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='upper right')

    # --- Panel 4: Time series ---
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(bout_data['frame'], bout_data['theta_error_deg'],
            color=bg_color, lw=1.2, label='θ error (°, +=right)')
    ax2.plot(bout_data['frame'], bout_data['ang_vel_fly_shifted_deg'],
             color='cyan', lw=1, alpha=0.8, label='ω shifted (°/s, +=rightward)')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax2.axhline(0, color='cyan', ls='--', lw=0.3, alpha=0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('θ error (°, +=right)')
    ax2.set_ylabel('ω shifted (°/s, +=rightward)', color='cyan')
    ax.set_title(f'Time series — {turn_label} turning bias')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'QC example {example_idx} ({turn_label}): {qc_acq}',
                 fontsize=MIN_FONTSIZE)
    plt.tight_layout()
    putil.label_figure(fig, figid)
    figname = f'qc_{turn_label}_example{example_idx}'
    plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {figname}")
    plt.close(fig)


# ---- Find and plot QC bouts ----
left_bouts = _find_turning_bouts(qc_df, direction='left',
                                  n_bouts=n_per_direction,
                                  min_bout_frames=min_bout_frames)
right_bouts = _find_turning_bouts(qc_df, direction='right',
                                   n_bouts=n_per_direction,
                                   min_bout_frames=min_bout_frames)

print(f"Found {len(left_bouts)} leftward bouts, {len(right_bouts)} rightward bouts")
for b in left_bouts:
    print(f"  LEFT:  {b[0]}, frames {b[1]}-{b[2]} ({b[3]} frames, {b[3]/fps:.1f}s), "
          f"mean ω={b[4]:.1f}°/s")
for b in right_bouts:
    print(f"  RIGHT: {b[0]}, frames {b[1]}-{b[2]} ({b[3]} frames, {b[3]/fps:.1f}s), "
          f"mean ω={b[4]:.1f}°/s")

all_qc_bouts = [(b, 'leftward') for b in left_bouts] + \
               [(b, 'rightward') for b in right_bouts]

plotted = 0
for bout_info, turn_label in all_qc_bouts:
    acq, bout_start, bout_end, n_bout, mean_av = bout_info

    acq_courtship = f1[(f1['acquisition'] == acq) & (f1['courtship'] == 1)].copy()
    if acq_courtship.empty:
        continue
    bout_data = acq_courtship[
        (acq_courtship['frame'] >= bout_start) & (acq_courtship['frame'] <= bout_end)
    ].copy()

    if len(bout_data) < min_bout_frames:
        print(f"  Skipping {acq}: only {len(bout_data)} courtship frames in bout")
        continue

    if 'bad_tracking' in bout_data.columns:
        bad_frac = bout_data['bad_tracking'].sum() / len(bout_data)
        if bad_frac > 0.05:
            print(f"  Skipping {acq}: {bad_frac:.0%} bad tracking frames")
            continue

    targ_bout = df_all[
        (df_all['acquisition'] == acq) & (df_all['id'] == 1)
        & (df_all['frame'] >= bout_start) & (df_all['frame'] <= bout_end)
    ].copy()

    plotted += 1
    print(f"\nExample {plotted} ({turn_label}): {acq}, "
          f"frames {bout_start}-{bout_end} ({n_bout} courtship frames, "
          f"{n_bout/fps:.1f}s), mean ω={mean_av:.1f}°/s")
    _plot_qc_example(bout_data, targ_bout, acq, plotted, turn_label)

    # Video uses a wider window with margin for context
    margin = 60
    all_acq = f1[f1['acquisition'] == acq].copy()
    vid_start = max(bout_start - margin, all_acq['frame'].min())
    vid_end = min(bout_end + margin, all_acq['frame'].max())
    vid_data = all_acq[
        (all_acq['frame'] >= vid_start) & (all_acq['frame'] <= vid_end)
    ].copy()
    targ_vid = df_all[
        (df_all['acquisition'] == acq) & (df_all['id'] == 1)
        & (df_all['frame'] >= vid_start) & (df_all['frame'] <= vid_end)
    ].copy()

    acq_file_name = acq_courtship['file_name'].iloc[0]
    acq_dir = os.path.join(DEFAULT_ROOTDIR, acq_file_name)
    out_path = os.path.join(figdir, f'qc_{turn_label}_example{plotted}.avi')
    qc.save_bout_video(acq_dir, vid_data, vid_start, vid_end, out_path,
                       targdf=targ_vid, color_col='ang_vel_fly_shifted_deg',
                       ori_col='ori', fps=fps, ori_image_sign=1)


# ============================================================
# PLOT 1: Gain curves per LED panel, species comparison
# ============================================================
#%%
PLOT_BY_INTENSITY = False
_xvar    = 'led_intensity' if PLOT_BY_INTENSITY else 'led_level'
_xlabel  = 'LED intensity' if PLOT_BY_INTENSITY else 'LED level'

yvar = 'ang_vel_fly_shifted_deg'
lw = 2

_xvals = sorted(chasedf[_xvar].dropna().unique())
n_led  = len(_xvals)

_has_pr = 'pr_direction' in chasedf.columns and chasedf['pr_direction'].notna().any()

species_list_p1 = sorted(chasedf['species'].dropna().unique())
n_rows = len(species_list_p1)

pr_palette = {'progressive': 'darkgreen', 'regressive': 'purple'}

fig, axn = plt.subplots(n_rows, n_led, figsize=(3.5 * n_led, 4 * n_rows),
                        sharex=True, sharey=True, squeeze=False)

for ri, sp in enumerate(species_list_p1):
    sp_data = chasedf[chasedf['species'] == sp]

    for li, led_val in enumerate(_xvals):
        ax = axn[ri, li]
        led_data = sp_data[sp_data[_xvar] == led_val]

        for sign, subset in [(-1, led_data[led_data['binned_theta_error_num'] < 0]),
                             (+1, led_data[led_data['binned_theta_error_num'] > 0])]:
            if subset.empty:
                continue
            sns.lineplot(data=subset, x='binned_theta_error_num', y=yvar,
                         hue='pr_direction', palette=pr_palette,
                         errorbar='se', err_style='bars', lw=lw,
                         err_kws={'linewidth': lw},
                         legend=(ri == 0 and li == n_led - 1 and sign == 1),
                         ax=ax)

        ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
        ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
        if ri == 0:
            ax.set_title(f'{_xlabel} {int(led_val)}', fontsize=MIN_FONTSIZE)
        if ri == n_rows - 1:
            ax.set_xlabel('← Fly\'s left  |  Target position (°)  |  Fly\'s right →')
        if li == 0:
            ax.set_ylabel(f'{sp}\nAng. velocity (°/s)\n← leftward | rightward →')
        ax.set_xticks(np.linspace(-deg_lim, deg_lim, 5))

if axn[0, -1].get_legend():
    sns.move_legend(axn[0, -1], loc='upper left', bbox_to_anchor=(1, 1),
                    frameon=False, title='Direction', fontsize=8)

fig.suptitle(
    f'Steering gain by {_xlabel} (courtship-matched >= {min_courtship_frac})',
    fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = f'gain_by_led_{_xvar}_species'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150,
            bbox_inches='tight')


# ============================================================
# PLOT 2: Gain per species, hue = LED level (overview)
# ============================================================
#%%
species_list = sorted(chasedf['species'].unique())
n_species = len(species_list)
led_palette = dict(zip(_xvals, sns.color_palette('rocket', n_colors=n_led)))

fig, axn = plt.subplots(1, n_species, figsize=(5 * n_species, 4),
                        sharex=True, sharey=True)
if n_species == 1:
    axn = [axn]

for si, sp in enumerate(species_list):
    ax = axn[si]
    sp_data = chasedf[chasedf['species'] == sp]

    for sign, subset in [(-1, sp_data[sp_data['binned_theta_error_num'] < 0]),
                         (+1, sp_data[sp_data['binned_theta_error_num'] > 0])]:
        if subset.empty:
            continue
        sns.lineplot(data=subset, x='binned_theta_error_num', y=yvar,
                     hue=_xvar, palette=led_palette,
                     errorbar='se', err_style='bars', lw=lw,
                     err_kws={'linewidth': lw},
                     legend=(si == n_species - 1 and sign == 1),
                     ax=ax)

    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_title(sp, fontsize=MIN_FONTSIZE + 2)
    ax.set_xlabel('← Fly\'s left  |  Target position (°)  |  Fly\'s right →')
    if si == 0:
        ax.set_ylabel('Ang. velocity (°/s)\n← leftward | rightward →')
    ax.set_xticks(np.linspace(-deg_lim, deg_lim, 5))

if axn[-1].get_legend():
    sns.move_legend(axn[-1], loc='upper left', bbox_to_anchor=(1, 1),
                    frameon=False, title=_xlabel, fontsize=8)

fig.suptitle(
    f'Steering gain by species (courtship-matched >= {min_courtship_frac})',
    fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = f'gain_by_species_led_{_xvar}_hue'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150,
            bbox_inches='tight')


# ============================================================
# PLOT 3: Species comparison per LED level, progressive/regressive
# ============================================================
#%%
# Both species on the same axes for each LED level, with line style
# distinguishing progressive vs. regressive and color distinguishing species.
n_pr = 2  # progressive, regressive
pr_styles = {'progressive': '-', 'regressive': '--'}

fig, axn = plt.subplots(1, n_led, figsize=(4 * n_led, 4.5),
                        sharex=True, sharey=True, squeeze=False)

for li, led_val in enumerate(_xvals):
    ax = axn[0, li]
    led_data = chasedf[chasedf[_xvar] == led_val]

    for pr_dir in ['progressive', 'regressive']:
        pr_data = led_data[led_data['pr_direction'] == pr_dir]
        if pr_data.empty:
            continue

        for sign, subset in [(-1, pr_data[pr_data['binned_theta_error_num'] < 0]),
                             (+1, pr_data[pr_data['binned_theta_error_num'] > 0])]:
            if subset.empty:
                continue
            sns.lineplot(data=subset, x='binned_theta_error_num', y=yvar,
                         hue='species', palette=species_palette,
                         style='species', dashes=False,
                         errorbar='se', err_style='bars', lw=lw,
                         err_kws={'linewidth': lw},
                         linestyle=pr_styles[pr_dir],
                         legend=(li == n_led - 1 and sign == 1),
                         ax=ax)

    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_title(f'{_xlabel} {int(led_val)}', fontsize=MIN_FONTSIZE)
    ax.set_xticks(np.linspace(-deg_lim, deg_lim, 5))
    if li == 0:
        ax.set_ylabel('Ang. velocity (°/s)\n← leftward | rightward →')
    else:
        ax.set_ylabel('')
    ax.set_xlabel('← Fly\'s left  |  Target position (°)  |  Fly\'s right →')

# Build a combined legend: species colors + pro/reg line styles
if axn[0, -1].get_legend():
    axn[0, -1].get_legend().remove()

from matplotlib.lines import Line2D
legend_handles = []
for sp in sorted(species_palette.keys()):
    if sp in species_list:
        legend_handles.append(Line2D([0], [0], color=species_palette[sp],
                                     lw=lw, label=sp))
legend_handles.append(Line2D([0], [0], color='gray', lw=lw, ls='-',
                              label='progressive'))
legend_handles.append(Line2D([0], [0], color='gray', lw=lw, ls='--',
                              label='regressive'))
axn[0, -1].legend(handles=legend_handles, loc='upper left',
                   bbox_to_anchor=(1, 1), frameon=False, fontsize=8)

fig.suptitle(
    f'Species comparison by {_xlabel} (courtship-matched >= {min_courtship_frac})',
    fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = f'gain_species_comparison_by_{_xvar}'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150,
            bbox_inches='tight')

# %%
