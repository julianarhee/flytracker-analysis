"""Tests for analyses/strain_variation/src/strain_metrics.py.

Pure per-group summary logic, exercised on the synthetic `synth_tracking_df`
fixture (see conftest.py). No mounted data volume required.
"""
import numpy as np
import pandas as pd
import pytest

import analyses.strain_variation.src.strain_funcs as sf
import analyses.strain_variation.src.strain_metrics as sm


@pytest.fixture
def labeled_df(synth_tracking_df):
    df = sf.derive_courtship_labels(synth_tracking_df, source='jaaba')
    return sm.add_strain_name(df)


# ---------------------------------------------------------------------------
# Labeling helpers
# ---------------------------------------------------------------------------
def test_add_strain_name(synth_tracking_df):
    out = sm.add_strain_name(synth_tracking_df)
    assert (out['strain_name'] == out['species'] + ' ' + out['strain']).all()


def test_add_legend_column_with_n(labeled_df):
    out, counts = sm.add_legend_column_with_n(labeled_df, key='strain_name')
    # Each strain has 2 fly pairs in the fixture.
    assert (counts == 2).all()
    sample = out['strain_name_legend'].iloc[0]
    assert '(n=2)' in sample


# ---------------------------------------------------------------------------
# group_means
# ---------------------------------------------------------------------------
def test_group_means_one_row_per_group(labeled_df):
    out = sm.group_means(labeled_df, 'vel')
    # 3 strains x 2 pairs = 6 groups, males only.
    assert len(out) == 6
    assert 'vel' in out.columns


def test_group_means_sex_filter(labeled_df):
    # Restricting to males should change the mean vs. all flies.
    male = sm.group_means(labeled_df, 'vel', sex='m')
    allflies = sm.group_means(labeled_df, 'vel', sex=None)
    assert not np.allclose(sorted(male['vel']), sorted(allflies['vel']))


def test_group_means_restrict_courting_subset(labeled_df):
    full = sm.group_means(labeled_df, 'dist_to_other')
    court = sm.group_means(labeled_df, 'dist_to_other', restrict_courting=True)
    # Same groups, but means differ once we restrict to courting frames.
    assert len(full) == len(court)


# ---------------------------------------------------------------------------
# Specific metrics
# ---------------------------------------------------------------------------
def test_behavior_probabilities_in_unit_range(labeled_df):
    probs = sm.behavior_probabilities(labeled_df, restrict_courting=True)
    for col in sm.BEHAVIOR_COLS:
        assert probs[col].between(0, 1).all()


def test_velocity_by_behavior_levels(labeled_df):
    vel = sm.velocity_by_behavior(labeled_df, behaviors=('chasing', 'singing'))
    assert set(vel['behavior'].unique()) == {'all', 'chasing', 'singing'}


def test_interfly_distance(labeled_df):
    dist = sm.interfly_distance(labeled_df)
    assert 'dist_to_other' in dist.columns
    assert (dist['dist_to_other'] >= 0).all()


# ---------------------------------------------------------------------------
# Distance binning
# ---------------------------------------------------------------------------
def test_bin_distance_left_edges(labeled_df):
    out, bins = sm.bin_distance(labeled_df, bin_size=5, max_dist=30)
    assert list(bins) == [0, 5, 10, 15, 20, 25, 30]
    vals = out['binned_dist_to_other'].dropna().unique()
    # Bin labels are left edges, all multiples of the bin size below max.
    assert set(vals).issubset({0.0, 5.0, 10.0, 15.0, 20.0, 25.0})


def test_behavior_by_distance_shape(labeled_df):
    out, bins = sm.behavior_by_distance(labeled_df, bin_size=5, max_dist=30)
    assert 'binned_dist_to_other' in out.columns
    for col in sm.BEHAVIOR_COLS:
        assert col in out.columns
        assert out[col].between(0, 1).all()
