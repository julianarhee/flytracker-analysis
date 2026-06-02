#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_strain_analysis.py

Orchestrator for the `strain_variation` comparison figures. Written #%%
cell-style (prototype interactively in the VSCode interactive window, per the
repo workflow), reading the aggregate parquet that `process_multichamber.py`
writes and producing the strain/species comparison plots.

Pipeline:
    process_multichamber.py  ->  <sp>_2x2_strains.parquet   (load/transform/annotate)
    derive_courtship_labels  ->  canonical is_<behavior> columns (analysis-time)
    strain_metrics           ->  tidy per-pair summaries
    strain_plots / spatial_maps -> figures

Behavior-label source is selectable via --labels (default: jaaba).

CLI usage:
    python run_strain_analysis.py [--labels jaaba|kinematic|manual]
                                  [--orienting-angle DEG]
                                  [--all-frames]
                                  [--rootdir PATH]
"""
#%%
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import libs.plotting as putil

try:
    import analyses.strain_variation.src.strain_funcs as sf
    import analyses.strain_variation.src.process_multichamber as pm
    import analyses.strain_variation.src.strain_metrics as sm
    import analyses.strain_variation.src.strain_plots as splot
    import analyses.strain_variation.src.spatial_maps as smaps
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import strain_funcs as sf
    import process_multichamber as pm
    import strain_metrics as sm
    import strain_plots as splot
    import spatial_maps as smaps

#%%
# ---------------------------------------------------------------------------
# Config  (defaults; overridden by CLI args when run as a script)
# ---------------------------------------------------------------------------
# parse_known_args lets the script also be run in interactive / #%% mode where
# sys.argv contains ipykernel or VSCode runner arguments.
_p = argparse.ArgumentParser(
    description='Full-dataset strain-variation analysis.')
_p.add_argument('--labels', dest='label_source', default='jaaba',
                choices=['jaaba', 'kinematic', 'manual'],
                help='Behavior-label source (default: jaaba).')
_p.add_argument('--orienting-angle', type=float, default=10,
                metavar='DEG',
                help='Facing-angle threshold for orienting in degrees (default: 10).')
_p.add_argument('--all-frames', action='store_true',
                help='Compute p(behavior) over all frames, not just courting frames.')
_p.add_argument('--rootdir', default=None,
                help='Data root directory (default: sf.ROOTDIR).')
_args, _ = _p.parse_known_args()

rootdir              = _args.rootdir or sf.ROOTDIR
LABEL_SOURCE         = _args.label_source       # 'jaaba' | 'kinematic' | 'manual'
ORIENTING_ANGLE_DEG  = _args.orienting_angle
COURTING_FRAMES_ONLY = not _args.all_frames     # p(behavior|courtship) when True

plot_style = 'white'
putil.set_sns_style(plot_style, min_fontsize=7)
bg_color = [0.7] * 3 if plot_style == 'dark' else 'k'

_, aggdir = pm.get_output_dirs(rootdir, make=False)
figdir = os.path.join(aggdir, 'figures', 'strain_comparison')
os.makedirs(figdir, exist_ok=True)
figid = aggdir

#%%
# ---------------------------------------------------------------------------
# Load aggregate parquet + derive canonical labels
# ---------------------------------------------------------------------------
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Load each per-species parquet through Arrow and filter to males before
# converting to pandas.  This avoids the OOM that occurs when the full
# combined dataset (both sexes, all acquisitions) is materialised in RAM:
#   • Arrow in-memory footprint is 3–5x smaller than pandas
#   • Male-only filter halves the row count before the pandas conversion
#   • Per-species loading means only one species' data is in Arrow at a time
#
# The per-species parquets are read with pq.ParquetFile (footer schema) rather
# than pd.read_parquet / pq.read_table to avoid a schema-fragmentation issue
# where the jaaba_unilateral_extension* columns are silently dropped by the
# dataset scanner when row groups were written at different times.
#
# Female rows are kept in a slim table (spatial-map columns only) for
# male_position_from_female_view; everything else uses males exclusively.

_SP_PARQUETS = {sp: os.path.join(aggdir, '{}_2x2_strains.parquet'.format(sp))
                for sp in ('Dmel', 'Dyak')}
_missing = [p for p in _SP_PARQUETS.values() if not os.path.exists(p)]
if _missing:
    raise FileNotFoundError(
        'Per-species parquets not found:\n  {}\n'
        'Run process_multichamber.py first.'.format('\n  '.join(_missing)))

# Columns needed from female rows for the female-centered spatial map only.
_FEM_SPATIAL_COLS = ['species', 'strain', 'acquisition', 'fly_pair',
                     'frame', 'sex', 'targ_rel_pos_x', 'targ_rel_pos_y']

_male_dfs = []
_fem_dfs  = []
for _sp, _sp_path in _SP_PARQUETS.items():
    print('Loading {} ...'.format(_sp), flush=True)
    _tbl = pq.ParquetFile(_sp_path).read()          # Arrow table (footer schema)
    _male_mask = pc.equal(_tbl.column('sex'), 'm')
    _male_dfs.append(_tbl.filter(_male_mask).to_pandas())
    _fem_cols = [c for c in _FEM_SPATIAL_COLS if c in _tbl.schema.names]
    _fem_dfs.append(
        _tbl.filter(pc.invert(_male_mask)).select(_fem_cols).to_pandas())
    del _tbl, _male_mask                            # free Arrow table immediately

df0 = pd.concat(_male_dfs, ignore_index=True)
df0_females = pd.concat(_fem_dfs, ignore_index=True)
del _male_dfs, _fem_dfs
print('Loaded: {:,} male rows | {:,} female rows (spatial maps only)'.format(
    len(df0), len(df0_females)))

df = sf.derive_courtship_labels(df0, source=LABEL_SOURCE,
                                orienting_angle_deg=ORIENTING_ANGLE_DEG)
# Free df0 immediately: it is a full copy of the raw data and is only needed
# again for the JAABA comparison (reloaded slim there).  Holding both df0 and
# df simultaneously doubles peak RAM.
del df0

df = sm.add_strain_name(df)
df, counts = sm.add_legend_column_with_n(df, key='strain_name')
print(counts)

#%%
# ---------------------------------------------------------------------------
# Mean velocity: overall vs. during chasing / singing
# ---------------------------------------------------------------------------
velbeh = sm.velocity_by_behavior(df, behaviors=('chasing', 'singing'))
velbeh, _ = sm.add_legend_column_with_n(velbeh, key='strain_name')

fig, axn = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ai, beh in enumerate(['all', 'chasing', 'singing']):
    sub = velbeh[velbeh['behavior'] == beh]
    splot.compare_metric(sub, 'vel', ax=axn[ai], x='strain_name_legend',
                         palette=splot.DEFAULT_PALETTE, edgecolor=bg_color,
                         show_legend=(ai == 2), between_group_spacing=10,
                         within_group_spacing=1.1, box_width=1.0)
    axn[ai].set_title(beh)
    axn[ai].set_ylabel('Mean velocity (mm/s)')
    axn[ai].set_ylim([0, 20])
splot.label_figure(fig, figid)
plt.savefig(os.path.join(figdir, 'mean_vel_by_behavior_per_strain.png'))

#%%
# ---------------------------------------------------------------------------
# p(behavior): courting / chasing / singing per strain
# ---------------------------------------------------------------------------
probs = sm.behavior_probabilities(
    df, behaviors=['is_courting'] + sm.BEHAVIOR_COLS,
    restrict_courting=COURTING_FRAMES_ONLY)
probs, _ = sm.add_legend_column_with_n(probs, key='strain_name')

fig, axn = splot.compare_metrics_row(
    probs, metrics=['is_chasing', 'is_singing', 'is_orienting'],
    labels=['p(chasing)', 'p(singing)', 'p(orienting)'], x='strain_name_legend',
    palette=splot.DEFAULT_PALETTE, edgecolor=bg_color,
    between_group_spacing=5, within_group_spacing=0.6, box_width=0.5)
splot.label_figure(fig, figid)
data_type = 'courtframes' if COURTING_FRAMES_ONLY else 'allframes'
plt.savefig(os.path.join(figdir, 'p-behaviors_{}.png'.format(data_type)))

#%%
# ---------------------------------------------------------------------------
# Interfly distance during courtship
# ---------------------------------------------------------------------------
dist = sm.interfly_distance(df, restrict_courting=COURTING_FRAMES_ONLY)
dist, _ = sm.add_legend_column_with_n(dist, key='strain_name')
fig, ax = plt.subplots(figsize=(5, 4))
splot.compare_metric(dist, 'dist_to_other', ax=ax, x='strain_name_legend',
                     edgecolor=bg_color, between_group_spacing=10,
                     within_group_spacing=1.2, box_width=1.0)
ax.set_ylabel('Interfly distance (mm)')
ax.set_ylim([0, 25])
splot.label_figure(fig, figid)
plt.savefig(os.path.join(figdir, 'dist_to_other_{}.png'.format(data_type)))

#%%
# ---------------------------------------------------------------------------
# Behavior vs. binned interfly distance (one figure per species)
# ---------------------------------------------------------------------------
import seaborn as sns

distbeh, dist_bins = sm.behavior_by_distance(
    df, behaviors=sm.BEHAVIOR_COLS, restrict_courting=True, bin_size=5, max_dist=30)
distbeh, _ = sm.add_legend_column_with_n(distbeh, key='strain_name')

for curr_species, sub in distbeh.groupby('species'):
    fig, axn = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ai, (beh, label) in enumerate(zip(
            sm.BEHAVIOR_COLS, ['p(orienting)', 'p(chasing)', 'p(singing)'])):
        sns.barplot(data=sub, x='binned_dist_to_other', y=beh, ax=axn[ai],
                    hue='strain_name_legend', palette='cubehelix', errorbar='ci')
        # seaborn version here rejects a `legend=` kwarg on barplot; drop the
        # legend on all but the last panel after drawing.
        if ai != 2 and axn[ai].legend_ is not None:
            axn[ai].legend_.remove()
        axn[ai].set_xlabel('distance to other (mm)')
        axn[ai].set_ylabel(label)
        axn[ai].set_ylim([0, 0.8])
    splot.label_figure(fig, figid)
    plt.savefig(os.path.join(
        figdir, 'p-behaviors_v_binned_dist_{}.png'.format(curr_species)))

#%%
# ---------------------------------------------------------------------------
# Spatial occupancy
# ---------------------------------------------------------------------------
court_ = smaps.courting_frames(df, mask_col='is_courting')

# Female-centered view: where is the male from the female's egocentric frame.
# Uses df0_females (slim: only spatial-map columns), not the full male df.
male_from_female = smaps.male_position_from_female_view(df0_females, court_)
for curr_species, sub in male_from_female.groupby('species'):
    fig, axn = smaps.plot_occupancy_grid(
        sub, grouper='strain', ncols=4, marker_color=bg_color, figid=figid,
        suptitle='Male position, female-centered ({})'.format(curr_species))
    plt.savefig(os.path.join(
        figdir, 'male-rel-pos_all-pairs_{}.png'.format(curr_species)))

# Male-centered view: where does the male keep the female.
female_from_male = smaps.female_position_from_male_view(court_)
for curr_species, sub in female_from_male.groupby('species'):
    fig, axn = smaps.plot_occupancy_grid(
        sub, grouper='strain', ncols=4, marker_color=bg_color, figid=figid,
        suptitle='Female position, male-centered ({})'.format(curr_species))
    plt.savefig(os.path.join(
        figdir, 'female-rel-pos_all-pairs_{}.png'.format(curr_species)))

# %%
# ---------------------------------------------------------------------------
# JAABA vs kinematic label comparison (aggregate across all acquisitions)
# ---------------------------------------------------------------------------
try:
    import analyses.strain_variation.src.compare_jaaba_to_heuristics as cmp
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import compare_jaaba_to_heuristics as cmp

jaaba_figdir = os.path.join(aggdir, 'figures', 'jaaba_vs_heuristic')
os.makedirs(jaaba_figdir, exist_ok=True)

# Reload a SLIM version of df0 for the JAABA comparison.  add_both_labelings
# derives labels twice (jaaba + kinematic) and keeps 3 copies simultaneously,
# so the input must be small.  We select only the ~20 columns it actually
# needs — ~5-10x fewer than the full parquet — instead of reusing the df0
# that was freed above.
_JAABA_COLS = [
    # pair identifiers / groupers
    'species', 'strain', 'acquisition', 'fly_pair', 'frame', 'sex',
    # kinematic gate variables
    'facing_angle', 'vel', 'dist_to_other', 'max_wing_ang', 'targ_pos_theta',
    # precomputed degree variants (absent in older parquets — guarded below)
    'facing_angle_deg', 'targ_pos_theta_deg',
    # JAABA binary score columns
    'jaaba_chasing_binary', 'jaaba_unilateral_extension_binary',
    # unconsidered-feature candidates (cmp.CHASING_UNCONSIDERED_COLS)
    'rel_vel', 'ang_vel_fly', 'targ_ang_vel', 'target_vel',
]
print('Reloading slim data for JAABA comparison ({} columns)...'.format(
    len(_JAABA_COLS)), flush=True)
_jaaba_dfs = []
for _sp, _sp_path in _SP_PARQUETS.items():
    _tbl = pq.ParquetFile(_sp_path).read()
    _male_mask = pc.equal(_tbl.column('sex'), 'm')
    _cols = [c for c in _JAABA_COLS if c in _tbl.schema.names]
    _jaaba_dfs.append(_tbl.filter(_male_mask).select(_cols).to_pandas())
    del _tbl, _male_mask
df0 = pd.concat(_jaaba_dfs, ignore_index=True)
del _jaaba_dfs
print('  {:,} rows, {} columns'.format(len(df0), len(df0.columns)), flush=True)

# Re-derive both label sets from the raw (un-labeled) slim df0.
df_both = cmp.add_both_labelings(df0)

# Per-(species, strain, acquisition, fly_pair, behavior) agreement summary.
summary_jaaba = cmp.compare_labelings(df_both)

# --- a) Per-acquisition p(beh) bars + confusion breakdown ----------------
for acq, sub_acq in df_both.groupby('acquisition'):
    sum_acq = cmp.compare_labelings(sub_acq)
    if sum_acq.empty:
        continue
    fig, _ = cmp.plot_comparison(sum_acq, figid=figid)
    fig.suptitle('JAABA vs kinematic: {}'.format(acq), fontsize=9)
    fpath = os.path.join(jaaba_figdir,
                         '{}_jaaba_vs_heuristic.png'.format(acq))
    plt.savefig(fpath, bbox_inches='tight')
    plt.close(fig)

# --- b) Scatter: p(behavior) JAABA vs kinematic, all pairs ----------------
fig, _ = cmp.plot_pbeh_scatter(summary_jaaba)
putil.label_figure(fig, figid)
plt.savefig(os.path.join(jaaba_figdir, 'pbeh_scatter_jaaba_vs_kin.png'),
            bbox_inches='tight')
plt.close(fig)

# --- c) Agreement metrics (kappa, Jaccard, F1) by species x behavior ------
fig, _ = cmp.plot_agreement_metrics(summary_jaaba)
putil.label_figure(fig, figid)
plt.savefig(os.path.join(jaaba_figdir, 'agreement_metrics_by_species.png'),
            bbox_inches='tight')
plt.close(fig)

# --- d) Save tidy summary table -------------------------------------------
summary_jaaba.to_csv(
    os.path.join(jaaba_figdir, 'jaaba_vs_heuristic_summary.csv'), index=False)
print('Saved JAABA vs heuristic comparison figures to:\n  {}'.format(jaaba_figdir))

# %%
# ---------------------------------------------------------------------------
# Kinematic threshold diagnostics + unconsidered features (chasing focus)
# ---------------------------------------------------------------------------
# The kinematic chasing gate uses three variables:
#   vel >= 10 mm/s,  facing_angle <= 60 deg,  dist_to_other <= 20 mm
# The cells below show:
#   1. KDE per agreement category (both / JAABA-only / kin-only / neither)
#      for each gated variable, with the threshold line.  Tells us whether
#      the threshold is misplaced or whether the problem is elsewhere.
#   2. KDE per agreement category for variables the transform computes but
#      the kinematic gate ignores — the candidates that explain why JAABA
#      catches frames the heuristic misses (and vice-versa).

# Ensure degree-unit variants exist (transform normally computes them, but
# guard here in case we loaded an older parquet).
if 'facing_angle_deg' not in df_both.columns:
    df_both['facing_angle_deg'] = np.rad2deg(df_both['facing_angle'])
if 'targ_pos_theta_deg' not in df_both.columns:
    df_both['targ_pos_theta_deg'] = np.rad2deg(df_both['targ_pos_theta'])

# --- e) Kinematic variable distributions + threshold lines -----------------
fig, _ = cmp.plot_kinematic_threshold_diagnostics(df_both, behavior='chasing')
putil.label_figure(fig, figid)
fig.suptitle('Kinematic gate variables — chasing (by agreement category)',
             fontsize=9, y=1.02)
plt.savefig(os.path.join(jaaba_figdir, 'chasing_kin_threshold_diagnostics.png'),
            bbox_inches='tight')
plt.close(fig)

# --- f) Unconsidered features: what JAABA sees that kinematics ignores -----
# These variables are absent from the chasing gate (or effectively not gated
# with min_wing_ang=0 / wide targ_pos_theta bounds), yet JAABA's classifier
# likely weights them via its temporal/appearance features.
extra_cols   = cmp.CHASING_UNCONSIDERED_COLS
extra_labels = cmp.CHASING_UNCONSIDERED_XLABELS
present = [c for c in extra_cols if c in df_both.columns]
if present:
    fig, _ = cmp.plot_unconsidered_features(
        df_both, extra_cols=extra_cols,
        xlabels=extra_labels, behavior='chasing')
    putil.label_figure(fig, figid)
    fig.suptitle(
        'Features absent from kinematic chasing gate (by agreement category)',
        fontsize=9, y=1.02)
    plt.savefig(
        os.path.join(jaaba_figdir, 'chasing_unconsidered_features.png'),
        bbox_inches='tight')
    plt.close(fig)
else:
    print('Note: unconsidered-feature columns not found in this parquet '
          '(re-process with a current process_multichamber to get them).')

print('Diagnostic figures saved to:\n  {}'.format(jaaba_figdir))

# %%
