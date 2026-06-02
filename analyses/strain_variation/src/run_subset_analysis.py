#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_subset_analysis.py

Runs the full strain-variation + JAABA-vs-heuristic analysis on a **subset**
of acquisitions (e.g. a pilot with 4 strains × 4 acquisitions per species)
so the full dataset doesn't have to be preprocessed up-front.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHERE TO EDIT TO CHANGE WHICH DATA ARE INCLUDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
→  See the "Dataset selection" cell below (search: SUBSET_ACQUISITIONS).

Two modes:
  1. Manual:   Set SUBSET_ACQUISITIONS to an explicit dict (see template).
               This is the canonical place to lock in a reproducible subset.
  2. Auto:     Leave SUBSET_ACQUISITIONS = None.  The script calls
               sf.select_subset_acquisitions() using N_STRAINS /
               N_ACQS_PER_STRAIN / AUTO_SEED.  The auto-selection is printed
               so you can copy it into SUBSET_ACQUISITIONS to lock it in.

Files written by this script
──────────────────────────────────────────────────────────
CREATED (new, does not overwrite full-pipeline files):
  <aggdir>/mel_yak_2x2_strains_testset.parquet   ← subset aggregate
  <aggdir>/figures/strain_comparison_testset/     ← strain plots
  <aggdir>/figures/jaaba_vs_heuristic_testset/    ← JAABA diagnostic plots

SHARED (loaded from cache, identical to full-pipeline parquets):
  <aggdir>/processed/<acq>.parquet  per-acquisition parquets
    These are the same files the full pipeline writes; the subset reuses
    them rather than duplicating.  `--new` reprocesses them in place.

The `testset` suffix in every output path guarantees no full-pipeline
file (`mel_yak_2x2_strains.parquet`, `Dmel_2x2_strains.parquet`, etc.)
is ever overwritten.

CLI usage:
    python run_subset_analysis.py [--labels jaaba|kinematic|manual]
                                  [--orienting-angle DEG]
                                  [--all-frames]
                                  [--tag SUBSET_TAG]
                                  [--new]
                                  [--rootdir PATH]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
#%%
import os
import sys
import argparse

print("Starting subset analysis...")
print("Loading libraries (matplotlib may take a moment)...", flush=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt

print("Libraries loaded. Importing analysis modules...", flush=True)

import libs.plotting as putil

try:
    import analyses.strain_variation.src.strain_funcs as sf
    import analyses.strain_variation.src.process_multichamber as pm
    import analyses.strain_variation.src.strain_metrics as sm
    import analyses.strain_variation.src.strain_plots as splot
    import analyses.strain_variation.src.spatial_maps as smaps
    import analyses.strain_variation.src.compare_jaaba_to_heuristics as cmp
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import strain_funcs as sf
    import process_multichamber as pm
    import strain_metrics as sm
    import strain_plots as splot
    import spatial_maps as smaps
    import compare_jaaba_to_heuristics as cmp

print("All modules loaded successfully.", flush=True)


# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI argument parsing  (parse_known_args so #%% / ipykernel args are ignored)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_p = argparse.ArgumentParser(
    description='Subset strain-variation + JAABA-vs-heuristic analysis.')
_p.add_argument('--labels', dest='label_source', default='jaaba',
                choices=['jaaba', 'kinematic', 'manual'],
                help='Behavior-label source (default: jaaba).')
_p.add_argument('--orienting-angle', type=float, default=10, metavar='DEG',
                help='Facing-angle threshold for orienting in degrees (default: 10).')
_p.add_argument('--all-frames', action='store_true',
                help='Compute p(behavior) over all frames, not just courting frames.')
_p.add_argument('--tag', dest='subset_tag', default='testset',
                help='Tag appended to every output path (default: testset).')
_p.add_argument('--new', action='store_true',
                help='Force reprocessing of cached per-acquisition parquets.')
_p.add_argument('--rootdir', default=None,
                help='Data root directory (default: sf.ROOTDIR).')
_args, _ = _p.parse_known_args()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset selection  ← EDIT THIS BLOCK TO CHANGE THE SUBSET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rootdir = _args.rootdir or sf.ROOTDIR

# Tag appended to every output file/directory this script creates.
# The default 'testset' clearly separates subset outputs from the full-pipeline
# parquets.  Change to e.g. 'testset_v2' or 'pilot_mel4str' to version runs.
SUBSET_TAG = _args.subset_tag

# ── Manual selection (preferred once you've reviewed the auto-selection) ──────
# Format:
#   { species: { strain_name: [acq_name, ...], ... }, ... }
#
# Set to None to auto-select using N_STRAINS / N_ACQS_PER_STRAIN below.
SUBSET_ACQUISITIONS = {
    'Dmel': {
        'CO13N': [
            '20250410-1521_fly1-4_Dmel-strains_3do_gh',
            #'20250522-1531_fly1-4_Dmel-strains_3do_gh',
        ],
        'CS Mai': [
            '20250407-1458_fly1-4_Dmel-SD-CO4_3do_gh',
            #'20250417-1428_fly1-4_Dmel-strains_6do_gh',
        ],
        'RG11N': [
            '20250410-1417_fly1-4_Dmel-strains_3do_gh',
            #'20250414-1544_fly1-4_Dmel-strains_4do_gh',
        ],
        'SD105N': [
            '20250404-1431_fly1-4_Dmel-strain_4do_gh',
            #'20250411-1407_fly1-4_Dmel-strains_4do_gh',
        ],
    },
    'Dyak': {
        'Abidjan 12 Abidjan, Ivory Coast': [
            '20250306-1137_fly1-4_Dyak-cost-abid-tai-cy_3do_gh',
            #'20250307-0920_fly1-4_Dyak-cost-abid-tai-cy_3do_gh',
        ],
        'CY 23 Nguti, Cameroon': [
            '20250306-1024_fly1-4_Dyak-cost-abid-tai-cy_3do_gh',
            #'20250307-1034_fly1-4_Dyak-cost-abid-tai-cy_3do_gh',
        ],
        'Gabon 35 Gabon': [
            '20250318-0935_fly1-4_Dyak-gab_4do_gh',
            #'20250320-0917_fly1-4_Dyak-gab_3do_gh',
        ],
        'Tai 18E2 Tai forest, Ivory Coast': [
            '20250306-0917_fly1-4_Dyak-cost-abid-tai-cy_3do_gh',
            #'20250307-1145_fly1-4_Dyak-abid-tai-cy_3do_gh',
        ],
    },
}  # Dmel: 4 strains × 2 acqs = 8 acqs (32 pairs) | Dyak: 4 strains × 2 acqs = 8 acqs (32 pairs)

# ── Auto-selection parameters (used when SUBSET_ACQUISITIONS is None) ─────────
N_STRAINS           = 4    # strains per species
N_ACQS_PER_STRAIN   = 4    # acquisitions per strain
AUTO_SEED           = None # None = alphabetical order (reproducible); int = shuffle

LABEL_SOURCE         = _args.label_source   # 'jaaba' | 'kinematic' | 'manual'
ORIENTING_ANGLE_DEG  = _args.orienting_angle
COURTING_FRAMES_ONLY = not _args.all_frames

plot_style = 'white'
putil.set_sns_style(plot_style, min_fontsize=7)
bg_color = 'k' if plot_style == 'white' else [0.7] * 3

_, aggdir = pm.get_output_dirs(rootdir, make=False)
figid = aggdir   # used by putil.label_figure

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Discovery: print all available strains and acquisitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run this cell to see what's on disk before deciding the subset.
available = sf.list_acquisitions_by_strain(rootdir)
print('\n=== Available acquisitions by strain ===')
sf.print_subset_summary(available)

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resolve the subset (auto or manual)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if SUBSET_ACQUISITIONS is None:
    subset = sf.select_subset_acquisitions(
        rootdir,
        n_strains_per_species=N_STRAINS,
        n_acqs_per_strain=N_ACQS_PER_STRAIN,
        seed=AUTO_SEED,
    )
    print('\n=== Auto-selected subset ===')
    print('(copy the block below into SUBSET_ACQUISITIONS to lock it in)\n')
    print('SUBSET_ACQUISITIONS = {')
    for sp, strain_map in sorted(subset.items()):
        print("    {!r}: {{".format(sp))
        for strain, acqs in sorted(strain_map.items()):
            print("        {!r}: [".format(strain))
            for a in acqs:
                print("            {!r},".format(a))
            print("        ],")
        print("    },")
    print('}')
else:
    subset = SUBSET_ACQUISITIONS
    print('\n=== Using manually specified subset ===')

sf.print_subset_summary(subset)

# Flatten to a per-species acquisition list for processing.
subset_acqs_per_species = {
    sp: [a for acqs in strain_map.values() for a in acqs]
    for sp, strain_map in subset.items()
}

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Processing: load or compute per-acquisition parquets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-acquisition parquets are written to (and cached in) the same
# `processed/` directory as the full pipeline.  Already-processed acquisitions
# are loaded from cache; new ones are processed and saved.
#
# NOTE: we call process_species per-species rather than aggregate_and_save,
# because aggregate_and_save would overwrite the full-pipeline species-level
# parquets (Dmel_2x2_strains.parquet etc.) with subset data.
#
# Change create_new=True below to force reprocessing of cached parquets.
CREATE_NEW = _args.new  # --new flag forces reprocessing of cached parquets

subset_dfs = []
for sp, strain_map in subset.items():
    all_acqs_sp = [a for acqs in strain_map.values() for a in acqs]
    print('\n=== Processing {} ({} acquisitions) ==='.format(sp, len(all_acqs_sp)))
    sp_df = pm.process_species(
        sp, rootdir=rootdir, acqs=all_acqs_sp, create_new=CREATE_NEW)
    if sp_df.empty:
        print('WARNING: no data returned for species {}'.format(sp))
        continue
    missing = set(all_acqs_sp) - set(sp_df['acquisition'].unique())
    if missing:
        print('WARNING [{}]: {} acquisitions not in output:\n  {}'.format(
            sp, len(missing), sorted(missing)))
    # Keep only male rows — all downstream metrics operate on the male (focal fly).
    # Female rows are NOT needed: strain_metrics, compare_jaaba_to_heuristics, and
    # the male-centered spatial map all filter to sex=='m'. The one exception
    # (male_position_from_female_view) is skipped below.
    if 'sex' in sp_df.columns:
        n_all = len(sp_df)
        sp_df = sp_df[sp_df['sex'] == 'm'].reset_index(drop=True)
        print('  [{}] male rows only: {:,} -> {:,}'.format(sp, n_all, len(sp_df)))
    if 'species' not in sp_df.columns:
        sp_df['species'] = (sp_df['dataset_species']
                            if 'dataset_species' in sp_df.columns else sp)
    subset_dfs.append(sp_df)

if not subset_dfs:
    raise RuntimeError('No subset data loaded — check rootdir and subset acquisition names.')

df0_subset = pd.concat(subset_dfs, ignore_index=True)

# Filter to only the strains explicitly in the subset.  Multi-chamber
# acquisitions contain chambers for strains not in the subset; drop them
# before the expensive parquet write and all downstream operations.
subset_strains = {s for strain_map in subset.values() for s in strain_map.keys()}
if 'strain' in df0_subset.columns:
    n_before = len(df0_subset)
    df0_subset = df0_subset[df0_subset['strain'].isin(subset_strains)].reset_index(drop=True)
    print('  Filtered to {} subset strains: {} -> {} rows'.format(
        len(subset_strains), n_before, len(df0_subset)))

df0_subset = pm.assign_global_id(df0_subset)

included_acqs   = sorted(df0_subset['acquisition'].unique())
included_strains = sorted(df0_subset['strain'].dropna().unique())
print('\n=== Subset loaded ===')
print('  Acquisitions : {}'.format(len(included_acqs)))
print('  Strains      : {}'.format(included_strains))
print('  Rows         : {}'.format(len(df0_subset)))

# Save the testset aggregate parquet.
# Name always contains the SUBSET_TAG so it never conflicts with the
# full-pipeline files (mel_yak_2x2_strains.parquet, Dmel_2x2_strains.parquet,
# Dyak_2x2_strains.parquet).
subset_fpath = os.path.join(
    aggdir, 'mel_yak_2x2_strains_{}.parquet'.format(SUBSET_TAG))
df0_subset.to_parquet(subset_fpath, engine='pyarrow', compression='snappy')
print('  Testset parquet: {}'.format(subset_fpath))

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Derive courtship labels and make figure output dirs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df = sf.derive_courtship_labels(df0_subset, source=LABEL_SOURCE,
                                orienting_angle_deg=ORIENTING_ANGLE_DEG)
df = sm.add_strain_name(df)
df, counts = sm.add_legend_column_with_n(df, key='strain_name')
print(counts)

figdir  = os.path.join(aggdir, 'figures', 'strain_comparison_{}'.format(SUBSET_TAG))
jaaba_figdir = os.path.join(aggdir, 'figures', 'jaaba_vs_heuristic_{}'.format(SUBSET_TAG))
os.makedirs(figdir, exist_ok=True)
os.makedirs(jaaba_figdir, exist_ok=True)

data_type = 'courtframes' if COURTING_FRAMES_ONLY else 'allframes'

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strain comparison figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- Mean velocity by behavior -----------------------------------------------
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
plt.savefig(os.path.join(figdir, 'mean_vel_by_behavior_per_strain.png'),
            bbox_inches='tight')

# --- p(behavior) per strain --------------------------------------------------
probs = sm.behavior_probabilities(
    df, behaviors=['is_courting'] + sm.BEHAVIOR_COLS,
    restrict_courting=COURTING_FRAMES_ONLY)
probs, _ = sm.add_legend_column_with_n(probs, key='strain_name')

fig, axn = splot.compare_metrics_row(
    probs, metrics=['is_chasing', 'is_singing', 'is_orienting'],
    labels=['p(chasing)', 'p(singing)', 'p(orienting)'],
    x='strain_name_legend', palette=splot.DEFAULT_PALETTE, edgecolor=bg_color,
    between_group_spacing=5, within_group_spacing=0.6, box_width=0.5)
splot.label_figure(fig, figid)
plt.savefig(os.path.join(figdir, 'p-behaviors_{}.png'.format(data_type)),
            bbox_inches='tight')

# --- Interfly distance during courtship --------------------------------------
dist = sm.interfly_distance(df, restrict_courting=COURTING_FRAMES_ONLY)
dist, _ = sm.add_legend_column_with_n(dist, key='strain_name')
fig, ax = plt.subplots(figsize=(5, 4))
splot.compare_metric(dist, 'dist_to_other', ax=ax, x='strain_name_legend',
                     edgecolor=bg_color, between_group_spacing=10,
                     within_group_spacing=1.2, box_width=1.0)
ax.set_ylabel('Interfly distance (mm)')
ax.set_ylim([0, 25])
splot.label_figure(fig, figid)
plt.savefig(os.path.join(figdir, 'dist_to_other_{}.png'.format(data_type)),
            bbox_inches='tight')

# --- p(behavior) vs. binned interfly distance (per species) ------------------
import seaborn as sns

distbeh, dist_bins = sm.behavior_by_distance(
    df, behaviors=sm.BEHAVIOR_COLS, restrict_courting=True,
    bin_size=5, max_dist=30)
distbeh, _ = sm.add_legend_column_with_n(distbeh, key='strain_name')

for curr_species, sub in distbeh.groupby('species'):
    fig, axn = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ai, (beh, label) in enumerate(zip(
            sm.BEHAVIOR_COLS, ['p(orienting)', 'p(chasing)', 'p(singing)'])):
        sns.barplot(data=sub, x='binned_dist_to_other', y=beh, ax=axn[ai],
                    hue='strain_name_legend', palette='cubehelix', errorbar='ci')
        if ai != 2 and axn[ai].legend_ is not None:
            axn[ai].legend_.remove()
        axn[ai].set_xlabel('distance to other (mm)')
        axn[ai].set_ylabel(label)
        axn[ai].set_ylim([0, 0.8])
    splot.label_figure(fig, figid)
    plt.savefig(os.path.join(
        figdir, 'p-behaviors_v_binned_dist_{}.png'.format(curr_species)),
        bbox_inches='tight')

# --- Spatial occupancy -------------------------------------------------------
# Only the male-centered view is available: female rows were dropped at load
# time to reduce memory.  male_position_from_female_view (female-centered) is
# skipped; re-enable by loading both sexes if that map is needed.
court_ = smaps.courting_frames(df, mask_col='is_courting')
female_from_male = smaps.female_position_from_male_view(court_)

for curr_species, sub in female_from_male.groupby('species'):
    fig, axn = smaps.plot_occupancy_grid(
        sub, grouper='strain', ncols=4, marker_color=bg_color, figid=figid,
        suptitle='Female position, male-centered ({})'.format(curr_species))
    plt.savefig(os.path.join(
        figdir, 'female-rel-pos_all-pairs_{}.png'.format(curr_species)),
        bbox_inches='tight')

print('Strain comparison figures -> {}'.format(figdir))

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JAABA vs kinematic: aggregate comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Add both label sets to the raw (un-labeled) subset df.
df_both = cmp.add_both_labelings(df0_subset)
summary_jaaba = cmp.compare_labelings(df_both)

# a) Per-acquisition p(beh) bars + confusion breakdown
for acq, sub_acq in df_both.groupby('acquisition'):
    sum_acq = cmp.compare_labelings(sub_acq)
    if sum_acq.empty:
        continue
    fig, _ = cmp.plot_comparison(sum_acq, figid=figid)
    fig.suptitle('JAABA vs kinematic: {}'.format(acq), fontsize=9)
    plt.savefig(os.path.join(
        jaaba_figdir, '{}_jaaba_vs_heuristic.png'.format(acq)),
        bbox_inches='tight')
    plt.close(fig)

# b) Scatter: p(behavior) JAABA vs kinematic
fig, _ = cmp.plot_pbeh_scatter(summary_jaaba)
putil.label_figure(fig, figid)
plt.savefig(os.path.join(jaaba_figdir, 'pbeh_scatter_jaaba_vs_kin.png'),
            bbox_inches='tight')
plt.close(fig)

# c) Agreement metrics (kappa, Jaccard, F1) by species × behavior
fig, _ = cmp.plot_agreement_metrics(summary_jaaba)
putil.label_figure(fig, figid)
plt.savefig(os.path.join(jaaba_figdir, 'agreement_metrics_by_species.png'),
            bbox_inches='tight')
plt.close(fig)

# d) Save tidy summary table
summary_jaaba.to_csv(
    os.path.join(jaaba_figdir, 'jaaba_vs_heuristic_summary.csv'), index=False)

print('JAABA comparison figures -> {}'.format(jaaba_figdir))

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JAABA vs kinematic: threshold diagnostics + unconsidered features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ensure degree-unit columns exist (transform normally writes them; guard for
# older parquets).
if 'facing_angle_deg' not in df_both.columns:
    df_both['facing_angle_deg'] = np.rad2deg(df_both['facing_angle'])
if 'targ_pos_theta_deg' not in df_both.columns:
    df_both['targ_pos_theta_deg'] = np.rad2deg(df_both['targ_pos_theta'])

# e) Kinematic variable distributions + threshold lines
fig, _ = cmp.plot_kinematic_threshold_diagnostics(df_both, behavior='chasing')
putil.label_figure(fig, figid)
fig.suptitle('Kinematic gate variables — chasing (by agreement category)',
             fontsize=9, y=1.02)
plt.savefig(os.path.join(jaaba_figdir,
                         'chasing_kin_threshold_diagnostics.png'),
            bbox_inches='tight')
plt.close(fig)

# f) Features absent from the chasing gate
present = [c for c in cmp.CHASING_UNCONSIDERED_COLS if c in df_both.columns]
if present:
    fig, _ = cmp.plot_unconsidered_features(
        df_both, extra_cols=cmp.CHASING_UNCONSIDERED_COLS,
        xlabels=cmp.CHASING_UNCONSIDERED_XLABELS, behavior='chasing')
    putil.label_figure(fig, figid)
    fig.suptitle(
        'Features absent from kinematic chasing gate (by agreement category)',
        fontsize=9, y=1.02)
    plt.savefig(os.path.join(jaaba_figdir,
                             'chasing_unconsidered_features.png'),
                bbox_inches='tight')
    plt.close(fig)
else:
    print('Note: unconsidered-feature columns missing from parquet '
          '(reprocess with current process_multichamber to include them).')

print('Diagnostic figures -> {}'.format(jaaba_figdir))
print('\nAll outputs under:\n  {}'.format(aggdir))

# %%
