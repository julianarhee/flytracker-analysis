#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spatial_maps.py

Egocentric spatial-occupancy maps for the `strain_variation` analysis: where one
fly sits relative to the other (centered at the origin) during courtship. Two
perspectives, both plotted from the `targ_rel_pos_x` / `targ_rel_pos_y` columns:

  - female-centered ("where is the male"): female rows during the frames where
    the paired male is courting.
  - male-centered ("where does the male keep the female"): the male's own rows
    during courtship.

Ported from the spatial-occupancy cells of
`analyses/multichamber/src/multichamber_strains.py` (`plot_occupancy`).
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import libs.plotting as putil


PAIR_KEYS = ['species', 'strain', 'acquisition', 'fly_pair']


def courting_frames(df, mask_col='is_courting', sex='m'):
    """Male rows flagged as courting (the `court_` set in the old script)."""
    return df[(df['sex'] == sex) & (df[mask_col] == 1)].copy()


def male_position_from_female_view(df, court_):
    """Female rows during the paired male's courting frames.

    On those female rows, `targ_rel_pos_*` is the *male's* position relative to
    the female (origin), i.e. "where is the male" from the female's view.
    """
    out = []
    for (sp, strain, acq, fp), pair_court in court_.groupby(PAIR_KEYS):
        fem = df[(df['species'] == sp) & (df['strain'] == strain)
                 & (df['acquisition'] == acq) & (df['fly_pair'] == fp)
                 & (df['sex'] == 'f')
                 & (df['frame'].isin(pair_court['frame']))]
        out.append(fem)
    if not out:
        return df.iloc[0:0].copy()
    return pd.concat(out).reset_index(drop=True)


def female_position_from_male_view(court_):
    """Male courting rows; `targ_rel_pos_*` is the female relative to the male."""
    return court_.copy()


def plot_occupancy(data, ax=None, cmap='magma', vmin=0, vmax=0.001, bins=100,
                   stat='probability', marker_color='w', lims=(-300, 300)):
    """2D histogram of egocentric target position, with an origin marker.

    `data` must have `targ_rel_pos_x` / `targ_rel_pos_y`.
    """
    if ax is None:
        _, ax = plt.subplots()
    sns.histplot(data=data, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax,
                 cmap=cmap, stat=stat, vmin=vmin, vmax=vmax, bins=bins)
    ax.plot(0, 0, color=marker_color, markersize=5, marker='>')
    ax.set_aspect(1)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if lims is not None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.axis('off')
    return ax


def plot_occupancy_grid(data, grouper='strain', ncols=4, cmap='magma',
                        vmax=0.001, bins=100, stat='probability',
                        marker_color='w', lims=(-300, 300), figid=None,
                        suptitle=None):
    """Grid of occupancy maps, one panel per level of `grouper`, + colorbar.

    Returns (fig, axn). Mirrors the old "all-pairs per species" occupancy figure.
    """
    levels = list(data.groupby(grouper).groups.keys())
    nrows = int(np.ceil(len(levels) / ncols))
    fig, axn = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                            figsize=(ncols * 3, nrows * 3), squeeze=False)
    ai = -1
    for ai, (level, sub) in enumerate(data.groupby(grouper)):
        ax = axn.flat[ai]
        plot_occupancy(sub, ax=ax, cmap=cmap, vmax=vmax, bins=bins, stat=stat,
                       marker_color=marker_color, lims=lims)
        ax.set_title(level, fontsize=6)
    for ax in axn.flat[ai + 1:]:
        ax.axis('off')

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    putil.colorbar_from_mappable(axn.flat[ai], norm=norm, cmap=cmap,
                                 axes=[0.92, 0.3, 0.01, 0.4], hue_title=stat)
    if suptitle:
        fig.text(0.1, 0.95, suptitle, fontsize=8)
    if figid:
        putil.label_figure(fig, figid)
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)
    return fig, axn
