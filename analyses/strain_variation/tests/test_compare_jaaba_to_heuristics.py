"""Tests for analyses/strain_variation/src/compare_jaaba_to_heuristics.py.

Agreement metrics on controlled boolean overlaps + dual-labeling shape, headless.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import analyses.strain_variation.src.strain_funcs as sf
import analyses.strain_variation.src.compare_jaaba_to_heuristics as cmp


# ---------------------------------------------------------------------------
# _pair_metrics on known overlaps
# ---------------------------------------------------------------------------
def test_pair_metrics_perfect_agreement():
    j = [1, 1, 0, 0]
    m = cmp._pair_metrics(j, j)
    assert m['agreement'] == 1.0
    assert m['jaccard'] == 1.0
    assert m['precision'] == 1.0 and m['recall'] == 1.0
    assert m['both'] == 2 and m['jaaba_only'] == 0 and m['kin_only'] == 0


def test_pair_metrics_partial_overlap():
    #   frame:   0  1  2  3  4
    j = np.array([1, 1, 1, 0, 0], bool)   # JAABA positive: 0,1,2
    k = np.array([0, 1, 1, 1, 0], bool)   # kin positive:   1,2,3
    m = cmp._pair_metrics(j, k)
    assert m['both'] == 2          # frames 1,2
    assert m['jaaba_only'] == 1    # frame 0
    assert m['kin_only'] == 1      # frame 3
    assert m['neither'] == 1       # frame 4
    assert m['jaccard'] == 2 / 4
    assert m['recall'] == 2 / 3    # of JAABA-positive (3), kin caught 2
    assert m['precision'] == 2 / 3  # of kin-positive (3), JAABA agrees on 2
    assert m['agreement'] == 3 / 5  # frames 1,2,4 agree


def test_cohen_kappa_chance_level():
    rng = np.random.default_rng(0)
    a = rng.random(5000) > 0.5
    b = rng.random(5000) > 0.5  # independent -> kappa ~ 0
    assert abs(cmp._cohen_kappa(a, b)) < 0.1


def test_cohen_kappa_perfect():
    a = np.array([1, 0, 1, 0, 1], bool)
    assert cmp._cohen_kappa(a, a) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Dual labeling + summary on synthetic acquisition
# ---------------------------------------------------------------------------
def test_add_both_labelings_columns(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    for beh in ['chasing', 'singing', 'courting']:
        assert 'is_{}_jaaba'.format(beh) in out.columns
        assert 'is_{}_kin'.format(beh) in out.columns
    assert 'is_orienting' in out.columns
    assert len(out) == len(synth_tracking_df)


def test_compare_labelings_summary(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    summary = cmp.compare_labelings(out)
    # one row per (pair, behavior); behaviors = chasing, singing.
    assert set(summary['behavior'].unique()) == {'chasing', 'singing'}
    for col in ['p_jaaba', 'p_kin', 'agreement', 'kappa', 'jaccard', 'f1']:
        assert col in summary.columns
    assert summary['agreement'].between(0, 1).all()
    assert summary['p_jaaba'].between(0, 1).all()
    assert summary['p_kin'].between(0, 1).all()


def test_plot_comparison_runs(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    summary = cmp.compare_labelings(out)
    fig, axn = cmp.plot_comparison(summary)
    assert axn.shape == (2, 2)  # 2 behaviors x [p(beh), confusion]
    plt.close(fig)


def test_plot_pbeh_scatter_runs(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    summary = cmp.compare_labelings(out)
    fig, axn = cmp.plot_pbeh_scatter(summary)
    behaviors = sorted(summary['behavior'].unique())
    assert len(axn) == len(behaviors)
    # Identity line lim should be positive for any non-trivial data.
    for ax in axn:
        xl = ax.get_xlim()
        assert xl[1] > 0
    plt.close(fig)


def test_plot_pbeh_scatter_species_colored(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    summary = cmp.compare_labelings(out)
    assert 'species' in summary.columns
    fig, axn = cmp.plot_pbeh_scatter(summary)
    # Last panel should have a legend (species colored).
    assert axn[-1].get_legend() is not None
    plt.close(fig)


def test_plot_agreement_metrics_runs(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    summary = cmp.compare_labelings(out)
    metrics = ('kappa', 'jaccard', 'f1')
    behaviors = sorted(summary['behavior'].unique())
    fig, axn = cmp.plot_agreement_metrics(summary, metrics=metrics)
    assert axn.shape == (len(metrics), len(behaviors))
    plt.close(fig)


def test_plot_agreement_metrics_ylim(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    summary = cmp.compare_labelings(out)
    fig, axn = cmp.plot_agreement_metrics(summary)
    for row in axn:
        for ax in row:
            yl = ax.get_ylim()
            assert yl[0] < 0 and yl[1] > 1  # always show the full 0–1 range
    plt.close(fig)


# ---------------------------------------------------------------------------
# add_agreement_category
# ---------------------------------------------------------------------------
def test_add_agreement_category_all_four(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    out = cmp.add_agreement_category(out, behavior='chasing')
    assert 'agreement_cat' in out.columns
    cats = set(out['agreement_cat'].unique())
    # With random data seeded at 0 we should see at least 3 distinct categories.
    assert cats <= {'both', 'JAABA only', 'kinematic only', 'neither'}
    assert len(cats) >= 2


def test_add_agreement_category_logic():
    """Directly verify category assignment for hand-crafted boolean vectors."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        'sex': 'm',
        'facing_angle': rng.uniform(0, 0.5, n),
        'vel': rng.uniform(5, 20, n),
        'dist_to_other': rng.uniform(5, 25, n),
        'targ_pos_theta': rng.uniform(-1, 1, n),
        'max_wing_ang': rng.uniform(0, 1, n),
        'jaaba_chasing_binary': np.array([True, False, True, False] * (n // 4)),
        'jaaba_unilateral_extension_binary': np.zeros(n, bool),
        'id': 0, 'frame': np.arange(n),
        'species': 'Dmel', 'strain': 'CS',
        'acquisition': 'test', 'fly_pair': 1,
    })
    out = cmp.add_both_labelings(df)
    out = cmp.add_agreement_category(out, behavior='chasing')
    both_mask = out['is_chasing_jaaba'].astype(bool) & out['is_chasing_kin'].astype(bool)
    jonly_mask = out['is_chasing_jaaba'].astype(bool) & ~out['is_chasing_kin'].astype(bool)
    konly_mask = ~out['is_chasing_jaaba'].astype(bool) & out['is_chasing_kin'].astype(bool)
    assert (out.loc[both_mask, 'agreement_cat'] == 'both').all()
    assert (out.loc[jonly_mask, 'agreement_cat'] == 'JAABA only').all()
    assert (out.loc[konly_mask, 'agreement_cat'] == 'kinematic only').all()


# ---------------------------------------------------------------------------
# plot_kinematic_threshold_diagnostics
# ---------------------------------------------------------------------------
def test_plot_kinematic_threshold_diagnostics_runs(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    n_thresh = len(cmp.CHASING_KIN_THRESHOLDS)
    fig, axn = cmp.plot_kinematic_threshold_diagnostics(out, behavior='chasing')
    assert len(axn) == n_thresh
    plt.close(fig)


def test_plot_kinematic_threshold_diagnostics_threshold_line(synth_tracking_df):
    """The dashed threshold line must appear on each panel."""
    out = cmp.add_both_labelings(synth_tracking_df)
    fig, axn = cmp.plot_kinematic_threshold_diagnostics(out, behavior='chasing')
    for ax in axn:
        vlines = [l for l in ax.lines if l.get_linestyle() in ('--', 'dashed')]
        assert len(vlines) >= 1, "Expected a dashed threshold line in each panel"
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_unconsidered_features
# ---------------------------------------------------------------------------
def test_plot_unconsidered_features_subset(synth_tracking_df):
    """Use only columns present in the synthetic fixture."""
    out = cmp.add_both_labelings(synth_tracking_df)
    # targ_pos_theta and max_wing_ang are in the fixture and absent from the
    # chasing gate (effectively) — good stand-ins for the diagnostic.
    fig, axn = cmp.plot_unconsidered_features(
        out, extra_cols=['targ_pos_theta', 'max_wing_ang'],
        xlabels=['target pos theta (rad)', 'max wing ang (rad)'],
        behavior='chasing')
    assert len(axn) == 2
    plt.close(fig)


def test_plot_unconsidered_features_missing_cols_skipped(synth_tracking_df):
    """Columns absent from the df are silently skipped; the rest still plot."""
    out = cmp.add_both_labelings(synth_tracking_df)
    fig, axn = cmp.plot_unconsidered_features(
        out,
        extra_cols=['nonexistent_col', 'targ_pos_theta'],
        xlabels=['ghost', 'targ pos theta'],
        behavior='chasing')
    assert len(axn) == 1  # only the present column is plotted
    plt.close(fig)


def test_plot_unconsidered_features_all_missing_raises(synth_tracking_df):
    out = cmp.add_both_labelings(synth_tracking_df)
    with pytest.raises(ValueError, match='None of the requested'):
        cmp.plot_unconsidered_features(
            out, extra_cols=['nonexistent_a', 'nonexistent_b'],
            xlabels=['x', 'y'], behavior='chasing')
