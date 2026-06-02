"""Tests for analyses/strain_variation/src/strain_plots.py.

Figure-producing functions are tested headless (Agg). We assert structure
(axes, ticks, returned stats) rather than pixels.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import analyses.strain_variation.src.strain_funcs as sf
import analyses.strain_variation.src.strain_metrics as sm
import analyses.strain_variation.src.strain_plots as splot


@pytest.fixture
def summary(synth_tracking_df):
    df = sf.derive_courtship_labels(synth_tracking_df, source='jaaba')
    df = sm.add_strain_name(df)
    return sm.behavior_probabilities(df, restrict_courting=True)


def test_mannwhitney_annotation_returns_result_and_annotates():
    fig, ax = plt.subplots()
    a = np.arange(10)
    b = np.arange(10) + 100  # clearly separated -> significant
    res = splot.mannwhitney_annotation(ax, a, b)
    assert hasattr(res, 'pvalue')
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() in {'n.s.', '*', '**'}
    plt.close(fig)


def test_grouped_boxplots_ticks_match_groups(summary):
    fig, ax = plt.subplots()
    splot.grouped_boxplots(summary, x='strain_name', y='is_chasing', ax=ax)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert set(labels) == set(summary['species'].unique())
    plt.close(fig)


def test_compare_metric_runs(summary):
    fig, ax = plt.subplots()
    out = splot.compare_metric(summary, 'is_chasing', ax=ax)
    assert out is ax
    plt.close(fig)


def test_compare_metrics_row_panel_count(summary):
    fig, axn = splot.compare_metrics_row(
        summary, metrics=['is_chasing', 'is_singing', 'is_orienting'])
    assert len(axn) == 3
    plt.close(fig)


def test_compare_metrics_row_with_legend_column(summary):
    summary2, _ = sm.add_legend_column_with_n(summary, key='strain_name')
    fig, axn = splot.compare_metrics_row(
        summary2, metrics=['is_chasing', 'is_singing'], x='strain_name_legend')
    assert len(axn) == 2
    plt.close(fig)
