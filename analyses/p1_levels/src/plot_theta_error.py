#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egocentric target position (theta error) analysis for the calibration
(P1 levels) dataset.

Plots:
  1) Polar egocentric target position with center-of-mass (COM) markers
     for each speed, organized as rows (species) x columns (LED levels).
  2) Summary: COM theta vs speed, faceted by LED level.
"""
#%%
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# ============================================================
# LOAD DATA
# ============================================================
#%%
data = load_and_prepare_dataset()
matched_df = data['matched_df']
basedir = data['basedir']
min_courtship_frac = data['min_courtship_frac']

figdir = os.path.join(basedir, 'p1_levels', 'theta_error')
os.makedirs(figdir, exist_ok=True)
print(f"Saving figures to: {figdir}")

figid = 'p1_levels_theta_error'

# ============================================================
# PREPARE
# ============================================================
#%%
# SPECIES_LED_FILTER: True  → restrict to low_led for Dmel, full_led for Dyak
#                    False → use all matched blocks (default)
SPECIES_LED_FILTER = True
SPECIES_LED_TYPE   = {'Dmel': 'low_led', 'Dyak': 'full_led'}

if SPECIES_LED_FILTER:
    _led_mask = matched_df.apply(
        lambda r: r['led_type'] == SPECIES_LED_TYPE.get(r['species'], r['led_type']),
        axis=1,
    )
    crt = matched_df[_led_mask].copy()
else:
    crt = matched_df.copy()

# When filtering by species led_type, use led_level (ordinal, same scale per species).
# Otherwise use led_intensity (physical units, comparable across species).
_xvar   = 'led_level'    if SPECIES_LED_FILTER else 'led_intensity'
_xlabel = 'LED level'    if SPECIES_LED_FILTER else 'LED intensity'

led_vals = sorted(crt[_xvar].dropna().unique())
species_list = sorted(crt['species'].unique())
all_speeds = sorted(crt['speed_hz'].dropna().unique())

n_led = len(led_vals)
n_species = len(species_list)

speed_palette = dict(zip(all_speeds,
                         sns.color_palette('viridis', n_colors=len(all_speeds))))

# ============================================================
# PLOT 1: Polar egocentric target position
#         Rows = species, Columns = LED levels, Hue = speed
#         Show scatter (low alpha) + COM markers per speed
# ============================================================
#%%
# Scale COM markers by the number of frames contributing to each (species,
# led, speed) group. Sparse conditions produce unreliable COMs that would
# otherwise appear with the same visual weight as well-sampled ones.
# SCALE_COM_ALPHA: fades out sparse COMs (alpha range: _COM_ALPHA_MIN–1.0)
# SCALE_COM_SIZE : shrinks sparse COMs (size range: _COM_SIZE_MIN–_COM_SIZE_MAX)
SCALE_COM_ALPHA   = False
SCALE_COM_SIZE    = True
_COM_ALPHA_MIN    = 0.15
_COM_SIZE_MIN     = 20
_COM_SIZE_MAX     = 120
_COM_SIZE_DEFAULT = 50   # used when SCALE_COM_SIZE is False

fig, axn = plt.subplots(n_species, n_led,
                        figsize=(3 * n_led, 3 * n_species),
                        subplot_kw={'projection': 'polar'})
if n_species == 1:
    axn = axn[np.newaxis, :]
if n_led == 1:
    axn = axn[:, np.newaxis]

# Pre-compute global max frame count for normalisation
_counts = crt.groupby(['species', _xvar, 'speed_hz']).size()
_max_n = _counts.max() if len(_counts) > 0 else 1

for si, sp in enumerate(species_list):
    sp_data = crt[crt['species'] == sp]
    for li, led_val in enumerate(led_vals):
        ax = axn[si, li]
        led_data = sp_data[sp_data[_xvar] == led_val]

        for speed_hz in all_speeds:
            sdf = led_data[led_data['speed_hz'] == speed_hz]
            if len(sdf) == 0:
                continue

            ax.scatter(sdf['targ_pos_theta'], sdf['targ_pos_radius'],
                       s=2, alpha=0.2, color=speed_palette[speed_hz],
                       edgecolor='none')

            # Circular mean: arctan2(mean(sin), mean(cos)) keeps the result
            # within [-π, π], matching the raw targ_pos_theta range.
            # Using np.unwrap().mean() was incorrect — it can shift the mean
            # by multiples of 2π when data spans the ±π boundary, placing
            # the COM marker far from the actual point cloud.
            _angles = sdf['targ_pos_theta'].dropna().values
            cm_theta = np.arctan2(np.mean(np.sin(_angles)),
                                  np.mean(np.cos(_angles)))
            cm_radius = sdf['targ_pos_radius'].dropna().mean()

            frac_n   = len(sdf) / _max_n
            cm_alpha = (_COM_ALPHA_MIN + frac_n * (1.0 - _COM_ALPHA_MIN)
                        if SCALE_COM_ALPHA else 1.0)
            cm_size  = (_COM_SIZE_MIN + frac_n * (_COM_SIZE_MAX - _COM_SIZE_MIN)
                        if SCALE_COM_SIZE else _COM_SIZE_DEFAULT)
            ax.scatter(cm_theta, cm_radius, s=cm_size, alpha=cm_alpha,
                       color=speed_palette[speed_hz],
                       edgecolor='k', linewidths=0.5, zorder=10)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        if si == 0:
            ax.set_title(f'{_xlabel} {int(led_val)}',
                         fontsize=MIN_FONTSIZE, pad=10)
        if li == 0:
            ax.set_ylabel(sp, fontsize=MIN_FONTSIZE, labelpad=20)

        ax.set_yticklabels([])
        ax.set_xticklabels([])

legend_handles = [mpl.lines.Line2D(
    [0], [0], marker='o', color='w',
    markerfacecolor=speed_palette[s], markersize=8,
    markeredgecolor='k', markeredgewidth=0.5)
    for s in all_speeds]
legend_labels = [f'{int(s)} Hz' for s in all_speeds]
fig.legend(legend_handles, legend_labels, loc='lower center',
           ncol=len(all_speeds), frameon=False, fontsize=MIN_FONTSIZE - 1,
           title='Target speed', title_fontsize=MIN_FONTSIZE,
           bbox_to_anchor=(0.5, -0.04))

_led_filter_label = (
    '   |   '.join(f'{sp}: {lt}' for sp, lt in SPECIES_LED_TYPE.items())
    if SPECIES_LED_FILTER else None
)
_base_title = f'Egocentric target position (courtship-matched >= {min_courtship_frac})'
if _led_filter_label:
    _base_title += f'\n{_led_filter_label}'
fig.suptitle(_base_title + '\nCOM per speed', fontsize=MIN_FONTSIZE + 2, y=1.02)
plt.tight_layout()
putil.label_figure(fig, figid)

figname = 'egocentric_target_position_com_by_led_speed'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150,
            bbox_inches='tight')

# ============================================================
# PLOT 2: Summary — COM theta vs speed, faceted by LED level
# ============================================================
#%%
com_rows = []
for (sp, led_val, speed_hz), grp in crt.groupby(
        ['species', _xvar, 'speed_hz']):
    if grp['targ_pos_theta'].dropna().empty:
        continue
    _angles = grp['targ_pos_theta'].dropna().values
    cm_theta = np.arctan2(np.mean(np.sin(_angles)), np.mean(np.cos(_angles)))
    cm_radius = grp['targ_pos_radius'].dropna().mean()
    com_rows.append({
        'species': sp,
        _xvar: led_val,
        'speed_hz': speed_hz,
        'com_theta_deg': np.rad2deg(cm_theta),
        'com_radius': cm_radius,
    })
com_df = pd.DataFrame(com_rows)

fig, axn = plt.subplots(1, n_led, figsize=(3 * n_led, 3.5),
                        sharex=True, sharey=True)
if n_led == 1:
    axn = [axn]

for li, led_val in enumerate(led_vals):
    ax = axn[li]
    plotd = com_df[com_df[_xvar] == led_val]
    sns.lineplot(data=plotd, ax=ax,
                 x='speed_hz', y='com_theta_deg',
                 hue='species', palette=species_palette,
                 marker='o', markersize=6,
                 legend=(li == n_led - 1))
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_title(f'{_xlabel} {int(led_val)}', fontsize=MIN_FONTSIZE)
    ax.set_xlabel('Speed (Hz)')
    if li == 0:
        ax.set_ylabel('COM theta (°)')

if n_led > 1:
    sns.move_legend(axn[-1], loc='upper left', bbox_to_anchor=(1, 1),
                    frameon=False, title='')

_com_title = f'COM of target angle vs. speed (courtship-matched >= {min_courtship_frac})'
if _led_filter_label:
    _com_title += f'\n{_led_filter_label}'
fig.suptitle(_com_title, fontsize=MIN_FONTSIZE + 2)
plt.tight_layout()
putil.label_figure(fig, figid)

figname = 'com_theta_vs_speed_by_led'
plt.savefig(os.path.join(figdir, f'{figname}.png'), dpi=150,
            bbox_inches='tight')

# %%
