#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_jaaba_to_heuristics.py

Sub-analysis: for a single acquisition, compare the two ways of labeling
chasing / singing — the **JAABA** classifier (`source='jaaba'`) vs. the
**kinematic heuristics** (`source='kinematic'`, the gates in
`strain_funcs.derive_courtship_labels`). Each acquisition has 4 fly-pairs, so
one acquisition already gives 4 paired examples.

Orienting is excluded from the comparison: it has no JAABA classifier (it's the
same `facing_angle` heuristic under both sources). Courting is reported for
context but differs only through chasing/singing.

Per focal male, per behavior, we compute:
  - p(beh) under each method (fraction of frames),
  - frame-wise agreement, Cohen's kappa, Jaccard (intersection / union),
  - treating JAABA as the reference: precision / recall / F1 of the heuristic,
  - the confusion breakdown (both / jaaba-only / kin-only / neither).
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    import analyses.strain_variation.src.strain_funcs as sf
    import analyses.strain_variation.src.process_multichamber as pm
    import analyses.strain_variation.src.strain_metrics as sm
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    import strain_funcs as sf
    import process_multichamber as pm
    import strain_metrics as sm


COMPARE_BEHAVIORS = ['chasing', 'singing']
PAIR_GROUPER = ['species', 'strain', 'acquisition', 'fly_pair']

# Kinematic gate thresholds for each behavior.
# Format: {column_name: (direction_str, value)} where direction_str is the
# inequality applied in `filter_chasing` / `derive_courtship_labels`.
# Column names match what the processed parquet holds (degree columns where
# the raw radian column has a _deg companion computed by the transform).
CHASING_KIN_THRESHOLDS = {
    'vel':              ('>=', 10),    # mm/s
    'facing_angle_deg': ('<=', 60),   # degrees
    'dist_to_other':    ('<=', 20),   # mm
}
SINGING_KIN_THRESHOLDS = {
    'max_wing_ang':     ('>=', np.deg2rad(30)),  # rad (~0.52)
    'facing_angle_deg': ('<=', 90),              # degrees
    'dist_to_other':    ('<=', 35),              # mm
}

# Variables computed by the transform pipeline that are *absent* from the
# kinematic chasing gate and that JAABA may implicitly weight.  The names
# correspond to the processed-parquet column names; a column not present in a
# given df is silently skipped by the diagnostic functions.
CHASING_UNCONSIDERED_COLS = [
    'rel_vel',           # rate of distance change (neg = approaching)
    'ang_vel_fly',       # focal fly turn rate (rad/s)
    'targ_ang_vel',      # angular speed of target in egocentric frame
    'target_vel',        # target (female) speed (mm/s)
    'max_wing_ang',      # wing extension (min=0 gate = no gate for chasing)
    'targ_pos_theta_deg',  # target bearing from focal fly (degrees)
]
CHASING_UNCONSIDERED_XLABELS = [
    'approach speed (rel_vel, mm/s)',
    'turn rate ang_vel_fly (rad/s)',
    'target ang vel in ego frame',
    'target velocity (mm/s)',
    'max wing angle (rad)',
    'target bearing targ_pos_theta (°)',
]

# Agreement-category palette shared across all diagnostic plots.
AGREEMENT_COLORS = {
    'both':           '#55a868',
    'JAABA only':     '#4c72b0',
    'kinematic only': '#dd8452',
    'neither':        '#aaaaaa',
}
AGREEMENT_CAT_ORDER = ['both', 'JAABA only', 'kinematic only', 'neither']


# ---------------------------------------------------------------------------
# Dual labeling
# ---------------------------------------------------------------------------
def add_both_labelings(df, orienting_angle_deg=10, chasing_kws=None,
                       singing_kws=None):
    """Add `is_<beh>_jaaba` and `is_<beh>_kin` columns from both label sources.

    Both labelings are derived on the same rows, so the columns align. Orienting
    is identical under both (kept once as `is_orienting`).
    """
    dfj = sf.derive_courtship_labels(df, source='jaaba',
                                     orienting_angle_deg=orienting_angle_deg)
    dfk = sf.derive_courtship_labels(df, source='kinematic',
                                     orienting_angle_deg=orienting_angle_deg,
                                     chasing_kws=chasing_kws, singing_kws=singing_kws)
    out = df.copy()
    out['is_orienting'] = dfj['is_orienting'].to_numpy()
    for beh in ['chasing', 'singing', 'courting']:
        out['is_{}_jaaba'.format(beh)] = dfj['is_{}'.format(beh)].to_numpy()
        out['is_{}_kin'.format(beh)] = dfk['is_{}'.format(beh)].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------
def _cohen_kappa(a, b):
    """Cohen's kappa for two boolean arrays (NaN if undefined)."""
    n = len(a)
    if n == 0:
        return np.nan
    po = np.mean(a == b)
    pa, pb = np.mean(a), np.mean(b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        # Both labelings constant and identical: perfect-but-degenerate.
        return np.nan
    return (po - pe) / (1 - pe)


def _pair_metrics(j, k):
    """Agreement metrics between two boolean arrays j (JAABA) and k (kinematic)."""
    j = np.asarray(j, dtype=bool)
    k = np.asarray(k, dtype=bool)
    n = len(j)
    both = int(np.sum(j & k))
    jaaba_only = int(np.sum(j & ~k))
    kin_only = int(np.sum(~j & k))
    neither = int(np.sum(~j & ~k))
    union = both + jaaba_only + kin_only
    return {
        'n': n,
        'p_jaaba': float(np.mean(j)) if n else np.nan,
        'p_kin': float(np.mean(k)) if n else np.nan,
        'agreement': float(np.mean(j == k)) if n else np.nan,
        'kappa': _cohen_kappa(j, k),
        'jaccard': both / union if union else np.nan,
        # JAABA as reference:
        'recall': both / (both + jaaba_only) if (both + jaaba_only) else np.nan,
        'precision': both / (both + kin_only) if (both + kin_only) else np.nan,
        'both': both, 'jaaba_only': jaaba_only,
        'kin_only': kin_only, 'neither': neither,
    }


def compare_labelings(df_both, behaviors=COMPARE_BEHAVIORS, grouper=PAIR_GROUPER,
                      sex='m'):
    """Per-(group, behavior) JAABA-vs-kinematic agreement summary (tidy)."""
    rows = []
    sub = df_both[df_both['sex'] == sex] if sex is not None else df_both
    for keys, g in sub.groupby(grouper):
        keyd = dict(zip(grouper, keys if isinstance(keys, tuple) else (keys,)))
        for beh in behaviors:
            m = _pair_metrics(g['is_{}_jaaba'.format(beh)],
                              g['is_{}_kin'.format(beh)])
            rec = dict(keyd); rec['behavior'] = beh; rec.update(m)
            # F1 from precision/recall.
            p, r = rec['precision'], rec['recall']
            rec['f1'] = (2 * p * r / (p + r)) if (p and r and (p + r) > 0) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_pbeh_paired(summary, behavior, ax=None):
    """Paired bars: p(behavior) under JAABA vs kinematic, one pair-group per x."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sub = summary[summary['behavior'] == behavior].copy()
    labels = ['{}\npair{}'.format(s.split()[0], int(fp))
              for s, fp in zip(sub['strain'], sub['fly_pair'])]
    x = np.arange(len(sub))
    w = 0.4
    ax.bar(x - w / 2, sub['p_jaaba'], w, label='JAABA', color='#4c72b0')
    ax.bar(x + w / 2, sub['p_kin'], w, label='kinematic', color='#dd8452')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.set_ylabel('p({})'.format(behavior))
    ax.set_title('p({}) by labeling'.format(behavior))
    ax.legend(frameon=False, fontsize=6)
    return ax


def plot_confusion_breakdown(summary, behavior, ax=None):
    """Stacked frame-fraction bars: both / JAABA-only / kin-only per pair."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sub = summary[summary['behavior'] == behavior].copy()
    n = sub['n'].to_numpy()
    both = sub['both'] / n
    jonly = sub['jaaba_only'] / n
    konly = sub['kin_only'] / n
    labels = ['{}\npair{}'.format(s.split()[0], int(fp))
              for s, fp in zip(sub['strain'], sub['fly_pair'])]
    x = np.arange(len(sub))
    ax.bar(x, both, label='both', color='#55a868')
    ax.bar(x, jonly, bottom=both, label='JAABA only', color='#4c72b0')
    ax.bar(x, konly, bottom=both + jonly, label='kinematic only', color='#dd8452')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('frame fraction')
    ax.set_title('{}: agreement / disagreement'.format(behavior))
    ax.legend(frameon=False, fontsize=6)
    return ax


def plot_comparison(summary, figid=None):
    """Full comparison figure: rows = behaviors, cols = [p(beh) paired, confusion]."""
    import matplotlib.pyplot as plt
    import libs.plotting as putil
    behaviors = list(summary['behavior'].unique())
    fig, axn = plt.subplots(len(behaviors), 2, figsize=(12, 4 * len(behaviors)),
                            squeeze=False)
    for ri, beh in enumerate(behaviors):
        plot_pbeh_paired(summary, beh, ax=axn[ri, 0])
        plot_confusion_breakdown(summary, beh, ax=axn[ri, 1])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    if figid:
        putil.label_figure(fig, figid)
    return fig, axn


def plot_pbeh_scatter(summary, behaviors=None, palette=None, figsize=None):
    """Scatter p(behavior) under JAABA vs kinematic across all pairs.

    One panel per behavior. Each dot is one fly-pair; species-colored when
    'species' is in the summary. An identity line (y = x) is drawn for
    reference so systematic over- or under-detection is visible at a glance.

    Args:
        summary (pd.DataFrame): output of `compare_labelings`.
        behaviors (list or None): behaviors to plot; defaults to all unique.
        palette (dict or None): {species: color}; auto-assigned if None.
        figsize (tuple or None): figure size; auto-sized if None.

    Returns:
        (fig, axn): axn is a 1-D array of Axes, one per behavior.
    """
    import matplotlib.pyplot as plt
    behaviors = behaviors or sorted(summary['behavior'].unique())
    n = len(behaviors)
    figsize = figsize or (4.5 * n, 4)
    fig, axn = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axn = axn[0]

    has_species = 'species' in summary.columns
    if has_species:
        species_list = sorted(summary['species'].unique())
        _default_colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
        palette = palette or dict(zip(species_list, _default_colors))

    for ai, beh in enumerate(behaviors):
        ax = axn[ai]
        sub = summary[summary['behavior'] == beh]
        if has_species:
            for sp, grp in sub.groupby('species'):
                ax.scatter(grp['p_jaaba'], grp['p_kin'],
                           label=sp, color=palette.get(sp, 'gray'),
                           alpha=0.75, s=40, linewidths=0)
        else:
            ax.scatter(sub['p_jaaba'], sub['p_kin'],
                       alpha=0.75, s=40, color='gray', linewidths=0)
        lim = max(sub['p_jaaba'].max(), sub['p_kin'].max()) * 1.15
        lim = max(lim, 0.05)
        ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlim([0, lim])
        ax.set_ylim([0, lim])
        ax.set_xlabel('p({}) JAABA'.format(beh), fontsize=8)
        ax.set_ylabel('p({}) kinematic'.format(beh), fontsize=8)
        ax.set_title(beh)
        if ai == n - 1 and has_species:
            ax.legend(frameon=False, fontsize=7)
    plt.subplots_adjust(wspace=0.4)
    return fig, axn


def plot_agreement_metrics(summary, metrics=('kappa', 'jaccard', 'f1'),
                           behavior_order=None, figsize=None):
    """Boxplots of per-pair agreement between JAABA and kinematic labels.

    Layout: rows = metrics (kappa / Jaccard / F1), columns = behaviors.
    When 'species' is in the summary, one box per species is drawn per panel;
    individual pair dots are overlaid.

    Args:
        summary (pd.DataFrame): output of `compare_labelings`.
        metrics (tuple): agreement columns to plot; default (kappa, jaccard, f1).
        behavior_order (list or None): behaviors to include and their order.
        figsize (tuple or None): auto-sized if None.

    Returns:
        (fig, axn): 2-D Axes array (n_metrics x n_behaviors).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    behaviors = behavior_order or sorted(summary['behavior'].unique())
    n_metrics, n_behs = len(metrics), len(behaviors)
    figsize = figsize or (4 * n_behs, 3 * n_metrics)
    fig, axn = plt.subplots(n_metrics, n_behs, figsize=figsize, squeeze=False)
    has_species = 'species' in summary.columns
    x_col = 'species' if has_species else None

    for ri, metric in enumerate(metrics):
        for ci, beh in enumerate(behaviors):
            ax = axn[ri, ci]
            sub = summary[summary['behavior'] == beh].copy()
            if x_col:
                sns.boxplot(data=sub, x=x_col, y=metric, ax=ax,
                            width=0.5, palette='Set2', linewidth=0.8)
                sns.stripplot(data=sub, x=x_col, y=metric, ax=ax,
                              color='k', size=3, alpha=0.6, jitter=True)
            else:
                sns.boxplot(data=sub, y=metric, ax=ax, width=0.5, linewidth=0.8)
                sns.stripplot(data=sub, y=metric, ax=ax,
                              color='k', size=3, alpha=0.6, jitter=True)
            ax.set_title(beh if ri == 0 else '')
            ax.set_xlabel('')
            ax.set_ylabel(metric if ci == 0 else '')
            ax.set_ylim([-0.15, 1.1])
            ax.axhline(0, color='gray', lw=0.5, ls='--')
    plt.subplots_adjust(hspace=0.4, wspace=0.35)
    return fig, axn


# ---------------------------------------------------------------------------
# Agreement-category helper
# ---------------------------------------------------------------------------
def add_agreement_category(df_both, behavior='chasing'):
    """Add an ``agreement_cat`` column classifying each frame by label agreement.

    Categories (string):
        'both'           — JAABA and kinematic both positive
        'JAABA only'     — JAABA positive, kinematic negative
        'kinematic only' — kinematic positive, JAABA negative
        'neither'        — both negative

    Args:
        df_both (pd.DataFrame): output of `add_both_labelings`.
        behavior (str): 'chasing' or 'singing'.

    Returns:
        pd.DataFrame: copy of df_both with 'agreement_cat' column added.
    """
    df_both = df_both.copy()
    j = df_both['is_{}_jaaba'.format(behavior)].astype(bool)
    k = df_both['is_{}_kin'.format(behavior)].astype(bool)
    cats = pd.Series('neither', index=df_both.index, dtype=object)
    cats[j & k]  = 'both'
    cats[j & ~k] = 'JAABA only'
    cats[~j & k] = 'kinematic only'
    df_both['agreement_cat'] = cats
    return df_both


# ---------------------------------------------------------------------------
# Kinematic threshold diagnostic plots
# ---------------------------------------------------------------------------
def plot_kinematic_threshold_diagnostics(df_both, behavior='chasing',
                                         thresholds=None, sex='m',
                                         figsize=None):
    """KDE per agreement category for each variable in the kinematic gate.

    For each kinematic variable, draws overlapping KDEs colored by agreement
    category and marks the threshold with a dashed vertical line. This makes
    it immediately clear how much of the JAABA-only / kinematic-only
    disagreement lies on the *wrong* side of each individual threshold.

    Args:
        df_both (pd.DataFrame): output of `add_both_labelings`.
        behavior (str): 'chasing' (default) or 'singing'.
        thresholds (dict or None): ``{col: (direction_str, value)}``, e.g.
            ``{'vel': ('>=', 10)}``. Defaults to ``CHASING_KIN_THRESHOLDS``
            or ``SINGING_KIN_THRESHOLDS`` depending on `behavior`.
        sex (str or None): restrict to this sex ('m' by default; None = all).
        figsize (tuple or None): auto-sized if None.

    Returns:
        (fig, axn): 1-D Axes array, one panel per variable.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if thresholds is None:
        thresholds = (CHASING_KIN_THRESHOLDS if behavior == 'chasing'
                      else SINGING_KIN_THRESHOLDS)

    sub = df_both[df_both['sex'] == sex].copy() if sex else df_both.copy()
    if 'facing_angle_deg' not in sub.columns and 'facing_angle' in sub.columns:
        sub['facing_angle_deg'] = np.rad2deg(sub['facing_angle'])
    if 'targ_pos_theta_deg' not in sub.columns and 'targ_pos_theta' in sub.columns:
        sub['targ_pos_theta_deg'] = np.rad2deg(sub['targ_pos_theta'])

    sub = add_agreement_category(sub, behavior)

    cols = list(thresholds.keys())
    n = len(cols)
    figsize = figsize or (4.5 * n, 4)
    fig, axn = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axn = axn[0]

    for ai, col in enumerate(cols):
        if col not in sub.columns:
            continue
        ax = axn[ai]
        direction, thresh = thresholds[col]
        for cat in AGREEMENT_CAT_ORDER:
            grp = sub[sub['agreement_cat'] == cat][col].dropna()
            if len(grp) < 5:
                continue
            sns.kdeplot(grp, ax=ax, label=cat,
                        color=AGREEMENT_COLORS[cat],
                        fill=True, alpha=0.25, linewidth=1.2)
        ax.axvline(thresh, color='k', lw=1.2, ls='--',
                   label='threshold ({}{})'.format(direction, thresh))
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel('density')
        ax.set_title('{} {} {}'.format(col, direction, thresh))
        if ai == n - 1:
            ax.legend(frameon=False, fontsize=6)
        elif ax.get_legend():
            ax.get_legend().remove()

    plt.subplots_adjust(wspace=0.4)
    return fig, axn


def plot_unconsidered_features(df_both, extra_cols=None, behavior='chasing',
                               sex='m', xlabels=None, figsize=None):
    """KDE distributions of variables absent from the kinematic gate.

    These are features the transform computes (or that JAABA implicitly uses)
    but that the kinematic heuristic does not gate on. Comparing the
    'JAABA only' and 'kinematic only' distributions highlights where the two
    methods diverge.

    Args:
        df_both (pd.DataFrame): output of `add_both_labelings`.
        extra_cols (list or None): columns to plot; defaults to
            ``CHASING_UNCONSIDERED_COLS``. Columns absent from df_both are
            silently skipped.
        behavior (str): 'chasing' or 'singing'.
        sex (str or None): restrict to this sex ('m'); None = all.
        xlabels (list or None): axis labels; defaults to column names. Must
            match the final (after skipping absent columns) list length.
        figsize (tuple or None): auto-sized if None.

    Returns:
        (fig, axn): 1-D Axes array, one panel per present column.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if extra_cols is None:
        extra_cols = CHASING_UNCONSIDERED_COLS
    if xlabels is None:
        if extra_cols is CHASING_UNCONSIDERED_COLS:
            xlabels = CHASING_UNCONSIDERED_XLABELS
        else:
            xlabels = list(extra_cols)

    sub = df_both[df_both['sex'] == sex].copy() if sex else df_both.copy()
    if 'facing_angle_deg' not in sub.columns and 'facing_angle' in sub.columns:
        sub['facing_angle_deg'] = np.rad2deg(sub['facing_angle'])
    if 'targ_pos_theta_deg' not in sub.columns and 'targ_pos_theta' in sub.columns:
        sub['targ_pos_theta_deg'] = np.rad2deg(sub['targ_pos_theta'])

    sub = add_agreement_category(sub, behavior)

    # Filter to columns that exist and build the matching label list.
    present = [(c, lbl) for c, lbl in zip(extra_cols, xlabels)
               if c in sub.columns]
    if not present:
        raise ValueError(
            "None of the requested extra_cols are in df_both.\n"
            "Requested: {}\nAvailable: {}".format(extra_cols, list(sub.columns)))
    cols_ok, xlabels_ok = zip(*present)

    n = len(cols_ok)
    figsize = figsize or (4.5 * n, 4)
    fig, axn = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axn = axn[0]

    for ai, (col, xlabel) in enumerate(zip(cols_ok, xlabels_ok)):
        ax = axn[ai]
        for cat in AGREEMENT_CAT_ORDER:
            grp = sub[sub['agreement_cat'] == cat][col].dropna()
            if len(grp) < 5:
                continue
            sns.kdeplot(grp, ax=ax, label=cat,
                        color=AGREEMENT_COLORS[cat],
                        fill=True, alpha=0.25, linewidth=1.2)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('density')
        ax.set_title(col)
        if ai == n - 1:
            ax.legend(frameon=False, fontsize=6)
        elif ax.get_legend():
            ax.get_legend().remove()

    plt.subplots_adjust(wspace=0.4)
    return fig, axn


# ---------------------------------------------------------------------------
# Loader for one acquisition (robust to the multi-row-group schema quirk)
# ---------------------------------------------------------------------------
def load_acquisition(acq, rootdir=None, columns=None):
    """Load one processed-acquisition parquet via the file-level reader.

    pd.read_parquet's dataset scanner can drop the jaaba_unilateral_extension*
    columns on these multi-row-group files; ParquetFile.read() uses the footer
    schema.
    """
    import pyarrow.parquet as pq
    rootdir = rootdir or sf.ROOTDIR
    _, aggdir = pm.get_output_dirs(rootdir, make=False)
    fpath = os.path.join(aggdir, 'processed', '{}.parquet'.format(acq))
    pf = pq.ParquetFile(fpath)
    return pf.read(columns=columns).to_pandas()


#%%
# ---------------------------------------------------------------------------
# Runnable: compare JAABA vs heuristic for one acquisition (4 pairs)
# ---------------------------------------------------------------------------
if __name__ == '__main__' and not hasattr(sys, 'ps1'):
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import libs.plotting as putil

    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', default=sf.ROOTDIR)
    parser.add_argument('--acq', required=True,
                        help='acquisition name (processed/<acq>.parquet)')
    parser.add_argument('--savedir', default=None)
    args = parser.parse_args()

    putil.set_sns_style('white', min_fontsize=7)
    _, aggdir = pm.get_output_dirs(args.rootdir, make=False)
    savedir = args.savedir or os.path.join(aggdir, 'figures', 'jaaba_vs_heuristic')
    os.makedirs(savedir, exist_ok=True)

    need = ['species', 'strain', 'acquisition', 'fly_pair', 'sex', 'id', 'frame',
            'facing_angle', 'vel', 'dist_to_other', 'targ_pos_theta', 'max_wing_ang',
            'jaaba_chasing_binary', 'jaaba_unilateral_extension_binary']
    df = load_acquisition(args.acq, rootdir=args.rootdir, columns=need)
    df = add_both_labelings(df)
    summary = compare_labelings(df)

    cols = ['strain', 'fly_pair', 'behavior', 'n', 'p_jaaba', 'p_kin',
            'agreement', 'kappa', 'jaccard', 'precision', 'recall', 'f1']
    print(summary[cols].to_string(index=False))

    fig, _ = plot_comparison(summary, figid=aggdir)
    fig.suptitle('JAABA vs kinematic: {}'.format(args.acq), fontsize=9)
    out = os.path.join(savedir, '{}_jaaba_vs_heuristic.png'.format(args.acq))
    plt.savefig(out, bbox_inches='tight')
    summary.to_csv(os.path.join(savedir, '{}_jaaba_vs_heuristic.csv'.format(args.acq)),
                   index=False)
    print('\nSaved: {}'.format(out))
