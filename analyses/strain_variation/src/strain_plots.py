#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strain_plots.py

Shared plotting + stats primitives for the `strain_variation` comparison
figures. These consume the *tidy* summary DataFrames produced by
`strain_metrics` and draw the recurring "grouped boxplot, species groups,
strain-colored boxes, Mann-Whitney yak-vs-mel annotation" panel.

Factored out of the ~6 copy-pasted cells in
`analyses/multichamber/src/multichamber_strains.py`
(`plot_grouped_boxplots`, `annotate_p_value_two_groups`).

Local to this analysis for now; promote to `libs.plotting` if another analysis
needs the grouped-boxplot-with-gaps primitive.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as spstats

import libs.plotting as putil


DEFAULT_PALETTE = 'PRGn'


def mannwhitney_annotation(ax, group_a, group_b, xy=(0.5, 1.0),
                           fontsize=8, color='k', alternative='two-sided'):
    """Two-sided Mann-Whitney U between two groups; annotate n.s./*/** on `ax`.

    Returns the scipy result so the caller can log the p-value.
    """
    res = spstats.mannwhitneyu(group_a, group_b, alternative=alternative)
    if res.pvalue >= 0.05:
        label = 'n.s.'
    elif res.pvalue < 0.01:
        label = '**'
    else:
        label = '*'
    ax.annotate(label, xy=xy, xycoords='axes fraction', ha='center',
                va='center', fontsize=fontsize, color=color)
    return res


def grouped_boxplots(data, x='strain_name', y='vel', grouper='species',
                     ax=None, palette=DEFAULT_PALETTE,
                     between_group_spacing=10, within_group_spacing=1.1,
                     box_width=0.5, lw=0.5, edgecolor='black',
                     show_legend=False):
    """Boxplots grouped by `grouper`, boxes colored by `x`, with custom spacing.

    Seaborn's boxplot has no gap/spacing control between groups, so positions
    are placed manually (ported from the old `plot_grouped_boxplots`; the stray
    `ai==2` legend gate is replaced by an explicit `show_legend` argument).
    """
    if ax is None:
        _, ax = plt.subplots()

    group_order = sorted(data[grouper].unique())
    box_order = sorted(data[x].unique())
    colors = dict(zip(box_order, sns.color_palette(palette, n_colors=len(box_order))))

    positions = {}
    x_ticks, x_tick_labels = [], []
    x_pos = 0
    for group in group_order:
        boxes = data[data[grouper] == group][x].unique()
        n = len(boxes)
        offsets = [(i - (n - 1) / 2) * within_group_spacing for i in range(n)]
        for i, box in enumerate(boxes):
            positions[(group, box)] = x_pos + offsets[i]
        x_ticks.append(x_pos)
        x_tick_labels.append(group)
        x_pos += between_group_spacing

    for (group, box), pos in positions.items():
        vals = data[(data[grouper] == group) & (data[x] == box)][y].dropna()
        if len(vals) == 0:
            continue
        ax.boxplot(vals, positions=[pos], widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor=colors[box], edgecolor=edgecolor, linewidth=lw),
                   whiskerprops=dict(color=edgecolor, linewidth=lw),
                   capprops=dict(color=edgecolor, linewidth=lw),
                   medianprops=dict(color=edgecolor),
                   flierprops=dict(marker='o', markersize=0, color=edgecolor))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    if show_legend:
        for box in box_order:
            ax.plot([], [], color=colors[box], label=box, linewidth=5)
        ax.legend(title='Strain', bbox_to_anchor=(1.05, 1), loc='upper left',
                  frameon=False, fontsize=4)
    return ax


def compare_metric(data, y, ax=None, x='strain_name', grouper='species',
                   species_pair=('Dyak', 'Dmel'), palette=DEFAULT_PALETTE,
                   annotate=True, edgecolor='black', show_legend=False,
                   **box_kws):
    """One comparison panel: grouped boxplots of `y` + Mann-Whitney annotation.

    The two-group test compares the per-pair `y` values of `species_pair[0]`
    vs `species_pair[1]`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    grouped_boxplots(data, x=x, y=y, grouper=grouper, ax=ax, palette=palette,
                     edgecolor=edgecolor, show_legend=show_legend, **box_kws)
    if annotate:
        a = data[data[grouper] == species_pair[0]][y]
        b = data[data[grouper] == species_pair[1]][y]
        if len(a) and len(b):
            mannwhitney_annotation(ax, a, b, color=edgecolor)
    return ax


def compare_metrics_row(data, metrics, labels=None, figsize=None, x='strain_name',
                        palette=DEFAULT_PALETTE, edgecolor='black', **box_kws):
    """A row of `compare_metric` panels, one per column in `metrics`.

    `metrics` is a list of y-columns; `labels` optionally overrides the y-axis
    label per panel. Legend is shown on the last panel only.
    """
    n = len(metrics)
    labels = labels or metrics
    fig, axn = plt.subplots(1, n, figsize=figsize or (4 * n, 4), squeeze=False)
    axn = axn[0]
    for ai, (metric, label) in enumerate(zip(metrics, labels)):
        compare_metric(data, metric, ax=axn[ai], x=x, palette=palette,
                       edgecolor=edgecolor, show_legend=(ai == n - 1), **box_kws)
        axn[ai].set_xlabel('')
        axn[ai].set_ylabel(label)
    sns.despine(offset=2, bottom=True)
    plt.subplots_adjust(wspace=0.5, top=0.85)
    return fig, axn


# Re-export commonly used helpers so analysis scripts import from one place.
label_figure = putil.label_figure
set_sns_style = putil.set_sns_style
