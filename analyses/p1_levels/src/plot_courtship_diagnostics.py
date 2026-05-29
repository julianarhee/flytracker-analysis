#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Courtship-level diagnostics for the calibration (P1 levels) dataset.

Verifies courtship engagement across species, LED levels, and speeds
before running downstream analyses (VF position, theta error, gain).

Two-stage matching:
  Stage 1 — Age-ATR selection: find condition(s) where Dmel and Dyak
            have the most comparable overall courtship levels.
  Stage 2 — Block matching: within selected conditions, keep only
            (led_level, speed_hz) blocks where courtship fraction >=
            threshold for both species.

Plots:
  1) Per-block courtship fraction by LED level x speed (all data).
  2) Courtship fraction by LED intensity, per species (all data).
  3) Courtship fraction by LED intensity, rows=species x cols=speed.
  4) Age-ATR selection: mean courtship by age_ATR per species.
  5) Age-ATR candidates ranked by species difference.
  6) Distribution of per-block courtship fractions (selected age_ATR).
  7) Heatmap: fraction of flies passing threshold (selected age_ATR).
  8) Box+strip: matched vs excluded blocks (selected age_ATR).
  9) Bar summary: N acquisitions, blocks, frames before/after matching.
"""
#%%
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import libs.plotting as putil

from analyses.p1_levels.src.load_calibration_data import (
    load_and_prepare_dataset, compute_courtship_fraction,
    PLOT_STYLE, MIN_FONTSIZE, SPECIES_PALETTE,
)

# ============================================================
# SETTINGS
# ============================================================
#%%
putil.set_sns_style(PLOT_STYLE, min_fontsize=MIN_FONTSIZE)
species_palette = SPECIES_PALETTE.copy()
bg_color = [0.7] * 3 if PLOT_STYLE == 'dark' else 'w'

# ============================================================
# LOAD DATA
# ============================================================
#%%
data = load_and_prepare_dataset(top_n_speeds=None)
# Note: top_n_speeds=None takes average over all speed blocks,
# s.t. id Dyak courts at 0.6 at best and 0.05 at worst, while 
# Dmel is at 0.8 and best and 0.3 at worst, the averages will be quite different.
df_all = data['df_all']
df_filtered = data['df_filtered']
matched_df = data['matched_df']
block_keys = data['block_keys']
basedir = data['basedir']
min_courtship_frac = data['min_courtship_frac']
selected_age_atr = data['selected_age_atr']
age_atr_candidates = data['age_atr_candidates']
age_atr_fly_means = data['age_atr_fly_means']

figdir = os.path.join(basedir, 'p1_levels', 'courtship_diagnostics')
os.makedirs(figdir, exist_ok=True)
print(f"Saving figures to: {figdir}")

figid = 'p1_levels_courtship_diagnostics'

# ============================================================
# COURTSHIP FRACTION PER BLOCK (all data, all age_ATR)
# ============================================================
#%%
court_frac_df = compute_courtship_fraction(
    df_all,
    grouper=['species', 'acquisition', 'age_ATR', 'led_type',
             'led_level', 'led_intensity', 'speed_hz'],
)

court_frac_active = court_frac_df[
    (court_frac_df['led_level'] > 0) & (court_frac_df['speed_hz'] > 0)
].copy()

species_list = sorted(court_frac_active['species'].unique())
n_species = len(species_list)

print("=== Per-block courtship fraction by species x led_type x age_ATR (LED>0, speed>0) ===")
print(court_frac_active.groupby(
    ['species', 'led_type', 'age_ATR']
)['courtship_frac'].agg(['mean', 'std', 'count']).to_string())

#%%
# ============================================================
# PLOT 1: Courtship fraction by LED level x speed
#         One subplot per species x age_ATR (ALL data)
# ============================================================
all_speeds = sorted(court_frac_active['speed_hz'].dropna().unique())
speed_palette = dict(zip(all_speeds,
                         sns.color_palette('viridis', n_colors=len(all_speeds))))

species_age_combos = (
    court_frac_active.groupby(['species', 'age_ATR'])
    .size().reset_index()[['species', 'age_ATR']]
)
n_facets = len(species_age_combos)
ncols = min(n_facets, 4)
nrows = int(np.ceil(n_facets / ncols))

fig, axn = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                        sharex=True, sharey=True, squeeze=False)
for idx, (_, row) in enumerate(species_age_combos.iterrows()):
    ax = axn.flat[idx]
    sp, age = row['species'], row['age_ATR']
    plotd = court_frac_active[
        (court_frac_active['species'] == sp) &
        (court_frac_active['age_ATR'] == age)
    ]
    sns.lineplot(data=plotd, ax=ax,
                 x='led_level', y='courtship_frac',
                 hue='speed_hz', palette=speed_palette,
                 legend=(idx == 0))
    ax.set_title(f'{sp} {age}', fontsize=MIN_FONTSIZE)
    ax.set_xlabel('LED level')
    ax.set_ylabel('Courtship frac')
    ax.set_ylim([-0.02, 1.02])
    ax.set_xticks(sorted(plotd['led_level'].dropna().unique()))

for idx in range(n_facets, nrows * ncols):
    axn.flat[idx].set_visible(False)

if n_facets > 0:
    sns.move_legend(axn.flat[0], loc='upper left', bbox_to_anchor=(1, 1),
                    frameon=False, title='Speed (Hz)', fontsize=8)
fig.suptitle('Courtship fraction per block (LED level x speed) — all data',
             fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'courtship_frac_by_led_and_speed_per_condition'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# PLOT 2: Courtship fraction by LED intensity, averaged across speeds
#         Faceted by species, style = age_ATR (ALL data)
# ============================================================
court_by_intensity = court_frac_active.groupby(
    ['species', 'acquisition', 'age_ATR', 'led_intensity']
)['courtship_frac'].mean().reset_index()

fig, axn = plt.subplots(1, n_species, figsize=(5 * n_species, 4), sharey=True)
if n_species == 1:
    axn = [axn]

for si, sp in enumerate(species_list):
    ax = axn[si]
    sp_data = court_by_intensity[court_by_intensity['species'] == sp]
    sns.lineplot(data=sp_data, ax=ax,
                 x='led_intensity', y='courtship_frac',
                 style='age_ATR', markers=True,
                 color=species_palette.get(sp, 'gray'),
                 errorbar='se')
    ax.set_xlabel('LED intensity')
    ax.set_ylabel('Courtship fraction' if si == 0 else '')
    ax.set_title(sp)
    ax.set_xticks(sorted(sp_data['led_intensity'].dropna().unique()))
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1),
                    frameon=False, fontsize=8)

fig.suptitle('Courtship fraction by LED intensity (avg. across speeds) — all data',
             fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'courtship_frac_by_led_intensity_per_species'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# PLOT 3: Courtship fraction by LED intensity
#         Rows = species, Columns = speed (ALL data)
# ============================================================
all_speeds_sorted = sorted(court_frac_active['speed_hz'].dropna().unique())
n_speeds = len(all_speeds_sorted)

all_age_atr_sorted = sorted(court_frac_active['age_ATR'].unique())
markers_cycle = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
age_atr_markers = {
    atr: markers_cycle[i % len(markers_cycle)]
    for i, atr in enumerate(all_age_atr_sorted)
}

fig, axn = plt.subplots(n_species, n_speeds,
                        figsize=(3.5 * n_speeds, 3.5 * n_species),
                        sharex=True, sharey=True, squeeze=False)

for si, sp in enumerate(species_list):
    sp_data = court_frac_active[court_frac_active['species'] == sp]
    for spi, spd in enumerate(all_speeds_sorted):
        ax = axn[si, spi]
        spd_data = sp_data[sp_data['speed_hz'] == spd]
        for atr in all_age_atr_sorted:
            atr_data = spd_data[spd_data['age_ATR'] == atr]
            if len(atr_data) == 0:
                continue
            sns.lineplot(data=atr_data, ax=ax,
                         x='led_intensity', y='courtship_frac',
                         marker=age_atr_markers[atr],
                         color=species_palette.get(sp, 'lightgray'),
                         errorbar='se', label=atr)
        if si == 0:
            ax.set_title(f'{int(spd)} Hz', fontsize=MIN_FONTSIZE)
        if spi == 0:
            ax.set_ylabel(f'{sp}\nCourtship fraction')
        else:
            ax.set_ylabel('')
        if si == n_species - 1:
            ax.set_xlabel('LED intensity')
            ax.set_xticks(sorted(spd_data['led_intensity'].dropna().unique()))
        else:
            ax.set_xlabel('')
        ax.get_legend().remove() if ax.get_legend() else None

# Single shared legend using all age_ATR conditions
import matplotlib.lines as mlines
legend_handles = [
    mlines.Line2D([], [], color=bg_color, marker=age_atr_markers[atr],
                  markersize=8, linewidth=2, label=atr)
    for atr in all_age_atr_sorted
]
fig.legend(handles=legend_handles, title='age-ATR',
           loc='upper right', bbox_to_anchor=(1.0, 0.98),
           frameon=False, fontsize=9, title_fontsize=9)

fig.suptitle('Courtship fraction by LED intensity per speed block — all data',
             fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'courtship_frac_by_led_intensity_species_x_speed'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# STAGE 1 DIAGNOSTICS: Age-ATR selection
# ============================================================

# ============================================================
# PLOT 4: Mean courtship fraction per age_ATR per species
#         Highlights the selected condition(s)
# ============================================================
if age_atr_fly_means is not None:
    fly_means = age_atr_fly_means.copy()
else:
    fly_means_grouper = ['species', 'acquisition', 'age_ATR', 'led_level',
                         'speed_hz']
    _cf = compute_courtship_fraction(df_all, grouper=fly_means_grouper)
    _cf = _cf[(_cf['led_level'] > 0) & (_cf['speed_hz'] > 0)]
    fly_means = _cf.groupby(
        ['species', 'acquisition', 'age_ATR']
    )['courtship_frac'].mean().reset_index()

all_age_atr = sorted(fly_means['age_ATR'].unique())

fig, ax = plt.subplots(figsize=(max(6, 2 * len(all_age_atr)), 4.5))

sns.barplot(data=fly_means, ax=ax,
            x='age_ATR', y='courtship_frac', hue='species',
            palette=species_palette, errorbar='se',
            order=all_age_atr, alpha=0.8)
sns.stripplot(data=fly_means, ax=ax,
              x='age_ATR', y='courtship_frac', hue='species',
              palette=species_palette, dodge=True, alpha=0.5,
              size=5, order=all_age_atr, legend=False)

for i, atr in enumerate(all_age_atr):
    if atr in selected_age_atr:
        ax.axvspan(i - 0.45, i + 0.45, color='lime', alpha=0.12, zorder=0)

ax.set_xlabel('age-ATR condition')
ax.set_ylabel('Mean courtship fraction\n(per fly, avg. across blocks)')
ax.set_title(f'Selected: {selected_age_atr}', fontsize=MIN_FONTSIZE)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

fig.suptitle('Stage 1: Age-ATR selection — courtship by condition',
             fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'age_atr_selection_courtship_by_condition'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# PLOT 5: Age-ATR candidates ranked by species difference
# ============================================================
if age_atr_candidates is not None and len(age_atr_candidates) > 0:
    cand_rows = []
    for c in age_atr_candidates:
        top_spds = c.get('top_speeds', {})
        row = {
            'condition': ' + '.join(c['age_atr_set']),
            'species_diff': c['species_diff'],
            'n_viable_speeds': c.get('n_viable_speeds', 0),
            **{f'frac_{sp}': c['mean_frac'].get(sp, np.nan)
               for sp in species_list},
            **{f'n_{sp}': c['n_flies'].get(sp, 0)
               for sp in species_list},
        }
        for sp in species_list:
            spds = top_spds.get(sp, []) if isinstance(top_spds, dict) \
                else top_spds
            row[f'speeds_{sp}'] = spds
        cand_rows.append(row)
    cand_df = pd.DataFrame(cand_rows)
    best_label = ' + '.join(selected_age_atr)

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(cand_df)), 5))
    bars = ax.barh(range(len(cand_df)), cand_df['species_diff'],
                   color=['lime' if c == best_label else 'lightgray'
                          for c in cand_df['condition']],
                   edgecolor='k', linewidth=0.5)
    ax.set_yticks(range(len(cand_df)))
    ax.set_yticklabels(cand_df['condition'], fontsize=MIN_FONTSIZE - 1)
    ax.set_xlabel('|species diff| (each at own best speeds)')
    ax.invert_yaxis()

    for i, row in cand_df.iterrows():
        label_parts = []
        for sp in species_list:
            spd_str = ', '.join(
                [str(int(s)) for s in row[f'speeds_{sp}']])
            label_parts.append(
                f'{sp}: {row[f"frac_{sp}"]:.2f} '
                f'(n={int(row[f"n_{sp}"])}, '
                f'speeds=[{spd_str}])')
        ax.text(row['species_diff'] + 0.003, i,
                '  |  '.join(label_parts),
                va='center', fontsize=MIN_FONTSIZE - 2)

    fig.suptitle(
        'Stage 1: Age-ATR candidates '
        '(each species scored at its own best speeds)',
        fontsize=MIN_FONTSIZE + 2)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    putil.label_figure(fig, figid)
    figname = 'age_atr_candidates_ranked'
    plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150,
                bbox_inches='tight')

#%%
# ============================================================
# STAGE 2 DIAGNOSTICS: Block-level courtship matching
# (operates on df_filtered — selected age_ATR only)
# ============================================================
match_grouper = ['species', 'acquisition', 'age_ATR', 'led_type',
                 'led_level', 'led_intensity', 'speed_hz']
match_frac = compute_courtship_fraction(df_filtered, grouper=match_grouper)
match_frac_active = match_frac[
    (match_frac['led_level'] > 0) & (match_frac['speed_hz'] > 0)
].copy()

match_frac_active['above_threshold'] = (
    match_frac_active['courtship_frac'] >= min_courtship_frac
)

# Each fly's block is matched independently — no requirement for both
# species to share the same (led_level, speed_hz).
match_frac_active['matched'] = match_frac_active['above_threshold']

#%%
# ============================================================
# PLOT 6: Distribution of per-block courtship fractions by species
#         (selected age_ATR only) with threshold line
# ============================================================
fig, axn = plt.subplots(1, n_species, figsize=(5 * n_species, 4),
                        sharex=True, sharey=True)
if n_species == 1:
    axn = [axn]

for si, sp in enumerate(species_list):
    ax = axn[si]
    sp_data = match_frac_active[match_frac_active['species'] == sp]

    ax.hist(sp_data['courtship_frac'], bins=25, range=(0, 1),
            color=species_palette.get(sp, 'gray'), alpha=0.7,
            edgecolor='k', linewidth=0.5)
    ax.axvline(x=min_courtship_frac, color='r', linestyle='--', lw=1.5,
               label=f'threshold = {min_courtship_frac}')

    n_total = len(sp_data)
    n_above = sp_data['above_threshold'].sum()
    n_matched = sp_data['matched'].sum()
    ax.set_title(f'{sp}\n{n_above}/{n_total} above thr, '
                 f'{n_matched} matched', fontsize=MIN_FONTSIZE)
    ax.set_xlabel('Courtship fraction')
    ax.set_ylabel('# blocks' if si == 0 else '')
    ax.legend(fontsize=8, frameon=False)

fig.suptitle(
    f'Per-block courtship fraction (age_ATR: {selected_age_atr})',
    fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'courtship_frac_distribution_by_species'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# PLOT 7: Heatmap of fraction of flies passing threshold
#         per (led_level, speed_hz) for each species
# ============================================================
fig, axn = plt.subplots(1, n_species, figsize=(5 * n_species, 4))
if n_species == 1:
    axn = [axn]

for si, sp in enumerate(species_list):
    ax = axn[si]
    sp_data = match_frac_active[match_frac_active['species'] == sp]

    pass_rate = sp_data.groupby(['led_intensity', 'speed_hz']).agg(
        n_pass=('above_threshold', 'sum'),
        n_total=('above_threshold', 'count'),
    ).reset_index()
    pass_rate['frac_pass'] = pass_rate['n_pass'] / pass_rate['n_total']

    pivot = pass_rate.pivot(index='led_intensity', columns='speed_hz',
                            values='frac_pass')
    pivot = pivot.sort_index(ascending=True)

    sns.heatmap(pivot, ax=ax, annot=True, fmt='.2f', vmin=0, vmax=1,
                cmap='YlGn', cbar_kws={'label': 'Frac. passing'},
                linewidths=0.5, linecolor='gray')
    ax.set_title(f'{sp}', fontsize=MIN_FONTSIZE)
    ax.set_xlabel('Speed (Hz)')
    ax.set_ylabel('LED intensity' if si == 0 else '')
    ax.invert_yaxis()

fig.suptitle(
    f'Fraction of flies with courtship >= {min_courtship_frac} '
    f'(age_ATR: {selected_age_atr})',
    fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'courtship_matching_heatmap_by_species'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# PLOT 8: Before vs after matching — courtship fractions
#         Strips + box for matched vs excluded blocks
# ============================================================
match_frac_active['status'] = match_frac_active['matched'].map(
    {True: 'matched', False: 'excluded'})

fig, axn = plt.subplots(1, n_species, figsize=(4.5 * n_species, 4.5),
                        sharey=True)
if n_species == 1:
    axn = [axn]

status_palette = {'matched': 'forestgreen', 'excluded': 'salmon'}
for si, sp in enumerate(species_list):
    ax = axn[si]
    sp_data = match_frac_active[match_frac_active['species'] == sp]

    sns.boxplot(data=sp_data, ax=ax,
                x='status', y='courtship_frac', order=['excluded', 'matched'],
                hue='status', palette=status_palette,
                width=0.5, fliersize=0, linewidth=1)
    sns.stripplot(data=sp_data, ax=ax,
                  x='status', y='courtship_frac', order=['excluded', 'matched'],
                  hue='status', palette=status_palette,
                  alpha=0.4, size=4, dodge=False, legend=False)
    ax.axhline(y=min_courtship_frac, color='r', linestyle='--', lw=1,
               label=f'threshold = {min_courtship_frac}')

    n_exc = (sp_data['status'] == 'excluded').sum()
    n_mat = (sp_data['status'] == 'matched').sum()
    ax.set_title(f'{sp}\n{n_mat} matched, {n_exc} excluded',
                 fontsize=MIN_FONTSIZE)
    ax.set_xlabel('')
    ax.set_ylabel('Courtship fraction' if si == 0 else '')
    ax.legend(fontsize=8, frameon=False, loc='lower right')

fig.suptitle(f'Stage 2: Block matching (age_ATR: {selected_age_atr})',
             fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'courtship_matching_before_vs_after'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# PLOT 9: Summary — N acquisitions, blocks, frames per species
#         Three stages: all data → age_ATR selected → matched
# ============================================================
f1_all = df_all[(df_all['id'] == 0) & (df_all['courtship'] == 1)
                & (df_all['led_level'] > 0) & (df_all['speed_hz'] > 0)].copy()
f1_filt = df_filtered[(df_filtered['id'] == 0) & (df_filtered['courtship'] == 1)
                      & (df_filtered['led_level'] > 0)
                      & (df_filtered['speed_hz'] > 0)].copy()

summary_rows = []
for sp in species_list:
    for stage, src in [('all data', f1_all),
                       ('age_ATR selected', f1_filt),
                       ('matched', matched_df)]:
        sp_src = src[src['species'] == sp]
        summary_rows.append({
            'species': sp,
            'stage': stage,
            'n_acquisitions': sp_src['acquisition'].nunique(),
            'n_blocks': sp_src.groupby(
                ['acquisition', 'led_intensity', 'speed_hz']).ngroups,
            'n_frames': len(sp_src),
        })
summary = pd.DataFrame(summary_rows)

stage_order = ['all data', 'age_ATR selected', 'matched']
stage_palette = {'all data': 'gray', 'age_ATR selected': 'steelblue',
                 'matched': 'forestgreen'}

fig, axn = plt.subplots(1, 3, figsize=(14, 4.5))
for mi, (metric, ylabel) in enumerate([
    ('n_acquisitions', '# acquisitions'),
    ('n_blocks', '# blocks'),
    ('n_frames', '# courtship frames'),
]):
    ax = axn[mi]
    x_positions = np.arange(n_species)
    n_stages = len(stage_order)
    bar_width = 0.8 / n_stages

    for sti, stage in enumerate(stage_order):
        vals = summary[summary['stage'] == stage][metric].values
        offset = (sti - (n_stages - 1) / 2) * bar_width
        bars = ax.bar(x_positions + offset, vals, bar_width,
                      color=stage_palette[stage], edgecolor='k',
                      linewidth=0.5,
                      label=stage if mi == 0 else None)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{int(val)}', ha='center', va='bottom',
                    fontsize=MIN_FONTSIZE - 2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(species_list)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax, offset=4, trim=True, bottom=True)

axn[0].legend(fontsize=9, frameon=False)

fig.suptitle(
    f'Matching pipeline summary (age_ATR: {selected_age_atr}, '
    f'threshold: {min_courtship_frac})',
    fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'courtship_matching_summary'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

# %%
