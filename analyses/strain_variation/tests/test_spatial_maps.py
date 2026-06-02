"""Tests for analyses/strain_variation/src/spatial_maps.py.

Egocentric occupancy helpers + headless (Agg) plotting structure.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

import analyses.strain_variation.src.strain_funcs as sf
import analyses.strain_variation.src.spatial_maps as smaps


@pytest.fixture
def labeled_df(synth_tracking_df):
    return sf.derive_courtship_labels(synth_tracking_df, source='jaaba')


def test_courting_frames_males_only(labeled_df):
    court_ = smaps.courting_frames(labeled_df, mask_col='is_courting')
    assert court_['sex'].eq('m').all()
    assert court_['is_courting'].eq(1).all()


def test_male_position_from_female_view_is_female_rows(labeled_df):
    court_ = smaps.courting_frames(labeled_df)
    mff = smaps.male_position_from_female_view(labeled_df, court_)
    assert mff['sex'].eq('f').all()
    # Each female row aligns to a frame where its paired male was courting.
    for (sp, strain, acq, fp), grp in mff.groupby(smaps.PAIR_KEYS):
        male_frames = court_[(court_['species'] == sp) & (court_['strain'] == strain)
                             & (court_['acquisition'] == acq)
                             & (court_['fly_pair'] == fp)]['frame']
        assert set(grp['frame']).issubset(set(male_frames))


def test_female_position_from_male_view_is_court(labeled_df):
    court_ = smaps.courting_frames(labeled_df)
    fmv = smaps.female_position_from_male_view(court_)
    assert len(fmv) == len(court_)
    assert fmv['sex'].eq('m').all()


def test_plot_occupancy_returns_ax(labeled_df):
    court_ = smaps.courting_frames(labeled_df)
    fig, ax = plt.subplots()
    out = smaps.plot_occupancy(court_, ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_occupancy_grid_returns_fig(labeled_df):
    court_ = smaps.courting_frames(labeled_df)
    dmel = court_[court_['species'] == 'Dmel']
    fig, axn = smaps.plot_occupancy_grid(dmel, grouper='strain', ncols=4)
    assert axn.size >= dmel['strain'].nunique()
    plt.close(fig)
