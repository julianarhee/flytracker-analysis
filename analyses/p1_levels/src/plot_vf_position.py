#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual field position analysis for the calibration (P1 levels) dataset.

Plots:
  - Species comparison: all conditions vs. courtship-matched (frontal/lateral).
  - VF fraction vs LED level (courtship-matched).
  - Delta VF fraction from LED-1 baseline.
  - VF fraction by LED level, averaged across speeds.
  - VF fraction split by speed per species.
"""
#%%
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as spstats

import libs.plotting as putil

from analyses.p1_levels.src.load_calibration_data import (
    load_and_prepare_dataset, PLOT_STYLE, MIN_FONTSIZE, SPECIES_PALETTE,
)

# ============================================================
# SETTINGS
# ============================================================
#%%
putil.set_sns_style(PLOT_STYLE, min_fontsize=MIN_FONTSIZE)
bg_color = [0.7] * 3 if PLOT_STYLE == 'dark' else 'w'
species_palette = SPECIES_PALETTE.copy()

frontal_deg = 25
lateral_deg = 45

# ============================================================
# LOAD DATA
# ============================================================
#%%
data = load_and_prepare_dataset()
# df_all: full dataset, all age-ATR, all frames
# df_filtered: age-ATR selected, but all frames (including led 0, courtship 0/1, all speeds)
# matched_df: courtship frames only (id=0, courtship=1) from blocks with >=min_courtship_frac (default, 0.2), excludes LED 0 and speed 0
df_unfiltered = data['df_all'] 
df_all = data['df_filtered']   
matched_df = data['matched_df'] 
basedir = data['basedir']
min_courtship_frac = data['min_courtship_frac']
selected_age_atr = data['selected_age_atr']

figdir = os.path.join(basedir, 'p1_levels', 'vf_position')
os.makedirs(figdir, exist_ok=True)
print(f"Saving figures to: {figdir}")

figid = 'p1_levels_vf_position'

# ============================================================
# PREPARE: male fly, courtship only
# ============================================================
#%%
# "All conditions" reference from full unfiltered dataset
f1_unfilt = df_unfiltered[df_unfiltered['id'] == 0].copy()
crt_unfilt = f1_unfilt[f1_unfilt['courtship'] == True].copy()
crt_unfilt['vf_position'] = np.nan
crt_unfilt.loc[crt_unfilt['targ_pos_theta'] < np.deg2rad(frontal_deg), 'vf_position'] = 'frontal'
crt_unfilt.loc[crt_unfilt['targ_pos_theta'] >= np.deg2rad(lateral_deg), 'vf_position'] = 'lateral'
plotdf_all = crt_unfilt[(crt_unfilt['led_level'] > 0) & (crt_unfilt['speed_hz'] > 0)].copy()

# Matched data with VF position
matched_df['vf_position'] = np.nan
matched_df.loc[matched_df['targ_pos_theta'] < np.deg2rad(frontal_deg), 'vf_position'] = 'frontal'
matched_df.loc[matched_df['targ_pos_theta'] >= np.deg2rad(lateral_deg), 'vf_position'] = 'lateral'

#%%
# ============================================================
# 1. Overall species VF comparison: ALL conditions vs COURTSHIP-MATCHED
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(8, 7))

for row_idx, (label, plotdf) in enumerate([
    ('All data (LED>0, speed>0)', plotdf_all),
    (f'Matched (age_ATR: {selected_age_atr}, frac >= {min_courtship_frac})',
     matched_df),
]):
    acq_zone = plotdf.groupby(
        ['species', 'acquisition', 'vf_position']
    )['frame'].count().reset_index(name='zone_frames')
    acq_total = plotdf.groupby(
        ['species', 'acquisition']
    )['frame'].count().reset_index(name='total_frames')
    vf_frac = acq_zone.merge(acq_total, on=['species', 'acquisition'])
    vf_frac['fraction'] = vf_frac['zone_frames'] / vf_frac['total_frames']
    vf_frac = vf_frac.dropna(subset=['vf_position'])

    for vi, vf_zone in enumerate(['frontal', 'lateral']):
        ax = axes[row_idx, vi]
        vf_data = vf_frac[vf_frac['vf_position'] == vf_zone]

        sns.barplot(data=vf_data, ax=ax,
                    x='species', y='fraction',
                    hue='species', palette=species_palette,
                    fill=False, errorbar='se')
        sns.stripplot(data=vf_data, ax=ax,
                      x='species', y='fraction',
                      hue='species', palette=species_palette,
                      dodge=False, alpha=0.6, legend=False)
        ax.set_xlabel('')
        ax.set_ylabel('Fraction' if vi == 0 else '')
        zone_label = (f'Frontal (< {frontal_deg}°)' if vf_zone == 'frontal'
                      else f'Lateral (>= {lateral_deg}°)')

        mel_vals = vf_data[vf_data['species'] == 'Dmel']['fraction'].dropna()
        yak_vals = vf_data[vf_data['species'] == 'Dyak']['fraction'].dropna()
        if len(mel_vals) > 0 and len(yak_vals) > 0:
            stat, pval = spstats.mannwhitneyu(
                mel_vals, yak_vals, alternative='two-sided')
            stars = ('***' if pval < 0.001 else '**' if pval < 0.01
                     else '*' if pval < 0.05 else 'n.s.')
            ax.set_title(f'{zone_label}\np={pval:.3g} ({stars})',
                         fontsize=MIN_FONTSIZE)
        else:
            ax.set_title(zone_label, fontsize=MIN_FONTSIZE)
        sns.despine(ax=ax, offset=4, trim=True, bottom=True)

    axes[row_idx, 0].set_ylabel(f'{label}\nFraction')

plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'vf_species_comparison_all_vs_matched'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# Helper: compute VF fraction (courtship in zone / total frames)
# including LED 0 and non-courtship frames.
# ============================================================
def _compute_vf_fractions(src_df, frontal_deg, lateral_deg):
    """Compute VF zone fractions from all frames in src_df (id=0)."""
    f1 = src_df[src_df['id'] == 0].copy()
    f1['vf_position'] = np.nan
    f1.loc[f1['targ_pos_theta'] < np.deg2rad(frontal_deg), 'vf_position'] = 'frontal'
    f1.loc[f1['targ_pos_theta'] >= np.deg2rad(lateral_deg), 'vf_position'] = 'lateral'

    grp_cols = ['species', 'acquisition', 'led_type', 'led_level']

    court_zone = f1[f1['courtship'] == True].groupby(
        grp_cols + ['vf_position']
    )['frame'].count().reset_index(name='zone_court_frames')

    total = f1.groupby(grp_cols)['frame'].count().reset_index(
        name='total_frames')

    vf_zones = pd.DataFrame({'vf_position': ['frontal', 'lateral']})
    grid = total.merge(vf_zones, how='cross')
    result = grid.merge(
        court_zone, on=grp_cols + ['vf_position'], how='left')
    result['zone_court_frames'] = result['zone_court_frames'].fillna(0)
    result['fraction'] = result['zone_court_frames'] / result['total_frames']
    return result

vf_all_data = _compute_vf_fractions(df_unfiltered, frontal_deg, lateral_deg)
vf_filtered_ageATR = _compute_vf_fractions(df_all, frontal_deg, lateral_deg)


#%%
# ============================================================
# 2. VF FRACTION vs LED LEVEL — all data vs matched
# ============================================================
all_led_levels = sorted(vf_all_data['led_level'].dropna().unique())

fig, axn = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

for row_idx, (label, vf_src) in enumerate([
    ('All data', vf_all_data),
    (f'Filtered (age_ATR: {selected_age_atr})', vf_filtered_ageATR),
]):
    for vi, vf_zone in enumerate(['frontal', 'lateral']):
        ax = axn[row_idx, vi]
        zone_data = vf_src[vf_src['vf_position'] == vf_zone]
        sns.lineplot(data=zone_data, ax=ax,
                     x='led_level', y='fraction',
                     hue='species', style='led_type',
                     palette=species_palette,
                     errorbar='se', markers=True, markersize=5)
        ax.set_xticks(all_led_levels)
        ax.set_xlabel('LED level' if row_idx == 1 else '')
        if vi == 0:
            ax.set_ylabel(f'{label}\nFraction')
        else:
            ax.set_ylabel('')
        if row_idx == 0:
            zone_label = (f'Frontal (< {frontal_deg}°)'
                          if vf_zone == 'frontal'
                          else f'Lateral (>= {lateral_deg}°)')
            ax.set_title(zone_label)

        n_lines = []
        for sp in sorted(zone_data['species'].unique()):
            sp_d = zone_data[zone_data['species'] == sp]
            for lt in sorted(sp_d['led_type'].dropna().unique()):
                n = sp_d[sp_d['led_type'] == lt]['acquisition'].nunique()
                n_lines.append(f'{sp}/{lt}: n={n}')
        ax.text(0.02, 0.97, '\n'.join(n_lines),
                transform=ax.transAxes,
                fontsize=MIN_FONTSIZE - 3, va='top', ha='left')

        if ax.get_legend():
            sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1),
                            frameon=False, title='', fontsize=8)
        else:
            ax.legend_.remove() if ax.legend_ else None

fig.suptitle('VF fraction vs LED level (courtship in zone / total frames)',
             fontsize=MIN_FONTSIZE + 2)

led_map = (df_all.groupby(['led_type', 'led_level'])['led_intensity']
           .apply(lambda x: sorted(x.dropna().unique())).reset_index())

led_text_parts = []
for led_t in sorted(led_map['led_type'].unique()):
    t_map = led_map[led_map['led_type'] == led_t].sort_values('led_level')
    mapping = ', '.join([
        f'{int(row.led_level)}→{int(row.led_intensity[0])}'
        for row in t_map.itertuples()
        if len(row.led_intensity) > 0])
    led_text_parts.append(f'{led_t}: {mapping}')
led_text = 'LED level → intensity:   ' + '     |     '.join(led_text_parts)
fig.text(0.5, -0.02, led_text, ha='center', va='top',
         fontsize=MIN_FONTSIZE - 2, style='italic')

plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'vf_fraction_vs_led_level_all_vs_matched'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# 3. DELTA from LED level 0 baseline (no LED)
# Reuses vf_filtered_ageATR (courtship in zone / total frames,
# including LED 0). Subtract LED 0 per fly, per led_type.
# led_type is kept so low_led and full_led are plotted separately.
# ============================================================
split_led_type = True  # set False to collapse low/full into one line per species

baseline = vf_filtered_ageATR[vf_filtered_ageATR['led_level'] == 0][
    ['species', 'acquisition', 'led_type', 'vf_position', 'fraction']
].rename(columns={'fraction': 'baseline_fraction'})

vf_norm = vf_filtered_ageATR.merge(
    baseline, on=['species', 'acquisition', 'led_type', 'vf_position'],
    how='left')
vf_norm['delta_fraction'] = vf_norm['fraction'] - vf_norm['baseline_fraction']
vf_norm = vf_norm.dropna(subset=['delta_fraction'])

all_led_levels = sorted(vf_norm['led_level'].dropna().unique())

fig, axn = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for vi, vf_zone in enumerate(['frontal', 'lateral']):
    ax = axn[vi]
    zone_data = vf_norm[vf_norm['vf_position'] == vf_zone]
    sns.lineplot(data=zone_data, ax=ax,
                 x='led_level', y='delta_fraction',
                 hue='species',
                 style='led_type' if split_led_type else None,
                 palette=species_palette,
                 errorbar='se', markers=True, markersize=5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_xlabel('LED level')
    ax.set_ylabel('Δ fraction (from LED 0)')
    ax.set_xticks(all_led_levels)
    if vf_zone == 'frontal':
        ax.set_title(f'Frontal (< {frontal_deg}°)')
    else:
        ax.set_title(f'Lateral (>= {lateral_deg}°)')
    if ax.get_legend():
        sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1),
                        frameon=False, title='', fontsize=8)
    else:
        ax.legend_.remove() if ax.legend_ else None
fig.suptitle(
    f'Δ VF fraction from LED 0 (age_ATR: {selected_age_atr})',
    fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)
figname = 'vf_fraction_delta_from_baseline_courtship_matched'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

#%%
# ============================================================
# 4: MATCHED DATA: Fraction of courtship in frontal vs. lateral VF
#    Averaged across speeds per fly, then mean +/- SEM.
# ============================================================
# PLOT_BY_INTENSITY : True  → x-axis is led_intensity (physical units, default)
#                    False → x-axis is led_level (ordinal 1-5)
# SPECIES_LED_FILTER: True  → restrict to low_led for Dmel, full_led for Dyak
#                    False → use all matched blocks (default)
PLOT_BY_INTENSITY  = False
SPECIES_LED_FILTER = True
SPECIES_LED_TYPE   = {'Dmel': 'low_led', 'Dyak': 'full_led'}

# Optionally restrict to species-specific led_type
if SPECIES_LED_FILTER:
    _led_mask = matched_df.apply(
        lambda r: r['led_type'] == SPECIES_LED_TYPE.get(r['species'], r['led_type']),
        axis=1,
    )
    _matched_plot = matched_df[_led_mask].copy()
    _led_filter_label = '   |   '.join(
        f'{sp}: {lt}' for sp, lt in SPECIES_LED_TYPE.items())
else:
    _matched_plot = matched_df
    _led_filter_label = None

# Choose x-axis variable
_xvar   = 'led_intensity' if PLOT_BY_INTENSITY else 'led_level'
_xlabel = 'LED intensity'  if PLOT_BY_INTENSITY else 'LED level'
_figname_suffix = 'intensity' if PLOT_BY_INTENSITY else 'level'

grouper = ['species', 'acquisition', _xvar, 'speed_hz', 'vf_position']
vf_counts = _matched_plot.groupby(grouper)['frame'].count().reset_index(
    name='n_frames')

total_grouper = ['species', 'acquisition', _xvar, 'speed_hz']
totals = _matched_plot.groupby(total_grouper)['frame'].count().reset_index(
    name='total_frames')

vf_frac = vf_counts.merge(totals, on=total_grouper)
vf_frac['fraction'] = vf_frac['n_frames'] / vf_frac['total_frames']
vf_frac = vf_frac.dropna(subset=['vf_position'])

fly_mean = vf_frac.groupby(
    ['species', 'acquisition', _xvar, 'vf_position']
)['fraction'].mean().reset_index()

#%%
_xticks = sorted(vf_frac[_xvar].dropna().unique())

_title = f'Courtship-matched (frac >= {min_courtship_frac})'
if _led_filter_label:
    _title += f' — {_led_filter_label}'

fig, axn = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=False)

for vi, vf_zone in enumerate(['frontal', 'lateral']):
    ax = axn[vi]
    plotd = fly_mean[fly_mean['vf_position'] == vf_zone]
    sns.lineplot(data=plotd, ax=ax,
                 x=_xvar, y='fraction',
                 hue='species', palette=species_palette,
                 errorbar='se', marker='o', markersize=5,
                 legend=(vi == 1))
    ax.set_xlabel(_xlabel)
    ax.set_ylabel('Fraction of courtship frames')
    ax.set_box_aspect(1)
    ax.set_xticks(_xticks)
    if vf_zone == 'frontal':
        ax.set_title(f'Frontal VF (< {frontal_deg}°)')
    else:
        ax.set_title(f'Lateral VF (>= {lateral_deg}°)')

sns.move_legend(axn[-1], loc='upper left', bbox_to_anchor=(1, 1),
                frameon=False, title='')
fig.suptitle(_title, fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)

figname = f'vf_fraction_frontal_lateral_by_led_{_figname_suffix}_matched'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150, bbox_inches='tight')

# ============================================================
# PLOT A2: Split by speed (hue) per species panel
# ============================================================
#%%
all_speeds = sorted(_matched_plot['speed_hz'].dropna().unique())
speed_palette = dict(zip(all_speeds,
                         sns.color_palette('viridis', n_colors=len(all_speeds))))

for vf_zone in ['frontal', 'lateral']:
    plotd = vf_frac[vf_frac['vf_position'] == vf_zone].copy()
    n_species = plotd['species'].nunique()
    species_list_plot = sorted(plotd['species'].unique())

    fig, axn = plt.subplots(1, n_species, figsize=(5 * n_species, 4),
                            sharex=True, sharey=True)
    if n_species == 1:
        axn = [axn]

    for si, sp in enumerate(species_list_plot):
        ax = axn[si]
        sp_data = plotd[plotd['species'] == sp]
        sns.lineplot(data=sp_data, ax=ax,
                     x=_xvar, y='fraction',
                     hue='speed_hz', palette=speed_palette,
                     legend=(si == n_species - 1))
        ax.set_title(f'{sp}')
        ax.set_xlabel(_xlabel)
        ax.set_ylabel(f'Fraction {vf_zone} VF')
        ax.set_box_aspect(1)
        ax.set_xticks(_xticks)

    if n_species > 1:
        sns.move_legend(axn[-1], loc='upper left', bbox_to_anchor=(1, 1),
                        frameon=False, title='Speed (Hz)')

    _sp_title = (f'{vf_zone.capitalize()} VF — courtship-matched '
                 f'(frac >= {min_courtship_frac})')
    if _led_filter_label:
        _sp_title += f' — {_led_filter_label}'
    fig.suptitle(_sp_title, fontsize=MIN_FONTSIZE + 2)
    plt.tight_layout()
    putil.label_figure(fig, figid)

    figname = f'vf_fraction_{vf_zone}_by_led_{_figname_suffix}_speed_matched'
    plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150,
                bbox_inches='tight')

# %%
