#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strain_metrics.py

Per-group summary metrics for the `strain_variation` analysis. These functions
read the processed aggregate parquet (after `strain_funcs.derive_courtship_labels`
has added the canonical `is_<behavior>` columns) and return *tidy* DataFrames,
one row per group, ready to hand to `strain_plots`. No plotting happens here.

Compute / plot are deliberately separated: the same per-(species, strain,
fly_pair) summary shape feeds every comparison plot, so we build the summary
once and let `strain_plots` draw it.

Ported / refactored from
`analyses/multichamber/src/multichamber_strains.py` (the repeated
"groupby -> mean -> grouped boxplot" cells).
"""
import numpy as np
import pandas as pd

try:
    import analyses.strain_variation.src.strain_funcs as sf
except ModuleNotFoundError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import strain_funcs as sf


# Default grouping for a per-pair summary (one row per fly pair).
PAIR_GROUPER = ['species', 'strain', 'strain_name', 'acquisition', 'fly_pair']
# Canonical label columns produced by derive_courtship_labels().
BEHAVIOR_COLS = ['is_{}'.format(b) for b in sf.CANONICAL_BEHAVIORS]


# ---------------------------------------------------------------------------
# Grouping / labeling helpers (add columns used for plotting)
# ---------------------------------------------------------------------------
def add_strain_name(df):
    """Add `strain_name` = "<species> <strain>" (groups colormaps cleanly)."""
    df = df.copy()
    df['strain_name'] = ['{} {}'.format(sp, st)
                         for sp, st in zip(df['species'], df['strain'])]
    return df


def add_legend_column_with_n(df, key='strain_name',
                             grouper=('species', 'strain', 'strain_name',
                                      'acquisition', 'fly_pair')):
    """Add `<key>_legend` annotating each level with its fly-pair count `n`.

    Replaces the duplicated `add_legend_column_with_n` / `_with_N` pair in the
    old script.
    """
    df = df.copy()
    grouper = list(grouper)
    conds = df[grouper].drop_duplicates()
    counts = conds.groupby(key)['fly_pair'].count()
    df['{}_legend'.format(key)] = df[key].map(
        lambda x: '{} (n={})'.format(x, counts[x]))
    return df, counts


# ---------------------------------------------------------------------------
# Generic per-group summary
# ---------------------------------------------------------------------------
def group_means(df, value_cols, grouper=PAIR_GROUPER, sex='m',
                restrict_courting=False, extra_query=None):
    """Mean of `value_cols` per group (one row per group).

    Args:
        df (pd.DataFrame): processed df with canonical labels.
        value_cols (str or list): column(s) to average.
        grouper (list): grouping columns.
        sex (str or None): restrict to this sex first ('m' by default; None=all).
        restrict_courting (bool): keep only `is_courting == 1` frames first.
        extra_query (str): optional pandas `query` applied before grouping.

    Returns:
        pd.DataFrame: grouper columns + mean value_cols, NaN groups dropped.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    sub = df
    if sex is not None:
        sub = sub[sub['sex'] == sex]
    if restrict_courting:
        sub = sub[sub['is_courting'] == 1]
    if extra_query:
        sub = sub.query(extra_query)
    return (sub.groupby(list(grouper))[value_cols]
            .mean().reset_index().dropna())


# ---------------------------------------------------------------------------
# Specific analyses (thin wrappers around group_means / filter)
# ---------------------------------------------------------------------------
def behavior_probabilities(df, behaviors=BEHAVIOR_COLS, grouper=PAIR_GROUPER,
                           restrict_courting=True):
    """p(behavior) per group: mean of each `is_<behavior>` column.

    With `restrict_courting=True` this is p(behavior | courtship).
    """
    df = df.copy()
    df[behaviors] = df[behaviors].astype(float)
    return group_means(df, behaviors, grouper=grouper,
                       restrict_courting=restrict_courting)


def mean_velocity(df, grouper=PAIR_GROUPER):
    """Mean male velocity per group (all frames)."""
    return group_means(df, 'vel', grouper=grouper)


def velocity_by_behavior(df, behaviors=('chasing', 'singing'),
                         grouper=PAIR_GROUPER):
    """Long-form mean velocity per group for 'all' frames + during each behavior.

    Returns columns: grouper..., 'behavior', 'vel'. 'behavior' is one of
    'all' / the supplied behaviors. Mirrors the old "velocity overall vs during
    chasing/singing" 3-panel figure (feed straight to a faceted comparison).
    """
    out = []
    allv = group_means(df, 'vel', grouper=grouper)
    allv['behavior'] = 'all'
    out.append(allv)
    for beh in behaviors:
        col = 'is_{}'.format(beh)
        beh_frames = df[df[col] == 1]
        v = group_means(beh_frames, 'vel', grouper=grouper)
        v['behavior'] = beh
        out.append(v)
    return pd.concat(out, ignore_index=True)


def interfly_distance(df, grouper=PAIR_GROUPER, restrict_courting=True):
    """Mean interfly distance (`dist_to_other`) per group."""
    return group_means(df, 'dist_to_other', grouper=grouper,
                       restrict_courting=restrict_courting)


# ---------------------------------------------------------------------------
# Distance-resolved behavior
# ---------------------------------------------------------------------------
def bin_distance(df, bin_size=5, max_dist=30, col='dist_to_other'):
    """Add a `binned_dist_to_other` column (left bin edge, float).

    Returns (df_copy, dist_bins) so callers can format axis ticks.
    """
    df = df.copy()
    dist_bins = np.arange(0, max_dist + bin_size, bin_size)
    df['binned_dist_to_other'] = pd.cut(df[col], bins=dist_bins,
                                        labels=dist_bins[:-1]).astype(float)
    return df, dist_bins


def behavior_by_distance(df, behaviors=BEHAVIOR_COLS, grouper=PAIR_GROUPER,
                         restrict_courting=True, bin_size=5, max_dist=30):
    """p(behavior) per group within each interfly-distance bin.

    Returns (tidy_df, dist_bins). tidy_df has grouper + 'binned_dist_to_other' +
    one mean column per behavior. Mirrors the old "p(singing/chasing) vs binned
    dist_to_other" analysis.
    """
    df = df.copy()
    df[behaviors] = df[behaviors].astype(float)
    df, dist_bins = bin_distance(df, bin_size=bin_size, max_dist=max_dist)
    grouper = list(grouper) + ['binned_dist_to_other']
    means = group_means(df, behaviors, grouper=grouper,
                        restrict_courting=restrict_courting)
    return means, dist_bins
