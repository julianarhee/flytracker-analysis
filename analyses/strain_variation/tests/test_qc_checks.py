"""Tests for the head-tail flip detector in analyses/strain_variation/src/qc_checks.py.

A head-tail flip is a ~180-degree frame-to-frame jump in orientation. The
detector works on the `ori` column and is invariant to a global sign flip
(FlyTracker raw ori vs our negated ori).
"""
import os

import matplotlib
matplotlib.use('Agg')  # headless, for the plotting helper

import numpy as np
import pandas as pd
import pytest

import analyses.strain_variation.src.qc_checks as qc


# ---------------------------------------------------------------------------
# wrapped_ori_jump_deg
# ---------------------------------------------------------------------------
def test_wrapped_ori_jump_smooth():
    # Smoothly rotating: small per-frame jumps, no flips.
    ori = np.deg2rad(np.arange(0, 60, 3))
    jumps = qc.wrapped_ori_jump_deg(ori)
    assert jumps[0] == 0
    assert np.all(jumps[1:] < 10)


def test_wrapped_ori_jump_detects_180():
    ori = np.deg2rad([10, 10, 10, -170, -170])  # ~180 deg flip at index 3
    jumps = qc.wrapped_ori_jump_deg(ori)
    assert jumps[3] > 150
    assert jumps[1] < 1


def test_wrapped_ori_jump_sign_flip_invariant():
    ori = np.deg2rad([10, 50, -120, 30])
    assert np.allclose(qc.wrapped_ori_jump_deg(ori),
                       qc.wrapped_ori_jump_deg(-ori))


# ---------------------------------------------------------------------------
# detect_headtail_flips
# ---------------------------------------------------------------------------
@pytest.fixture
def flip_df():
    """One fly steady at 140 deg, flips ~180 deg to -40 at frame 7, then flips
    back at frame 18 (mirrors the real example, fly 1)."""
    n = 20
    ori_deg = np.full(n, 140.0)
    ori_deg[7:18] = -40.0
    ori_deg[18:] = 137.0
    return pd.DataFrame({'id': 0, 'frame': np.arange(n), 'sex': 'm', 'strain': 'X',
                         'ori': np.deg2rad(ori_deg)})


def test_detect_headtail_flips_marks_jumps(flip_df):
    out = qc.detect_headtail_flips(flip_df, flip_threshold_deg=150.0)
    flip = out.sort_values('frame')['headtail_flip'].values
    # Exactly the two flip frames (into -40 at 7, back to 137 at 18).
    assert list(np.where(flip)[0]) == [7, 18]


def test_detect_headtail_flips_ignores_smooth_rotation():
    n = 30
    df = pd.DataFrame({'id': 0, 'frame': np.arange(n), 'sex': 'm', 'strain': 'X',
                       'ori': np.deg2rad(np.linspace(-41, 83, n))})  # smooth turn
    out = qc.detect_headtail_flips(df)
    assert not out['headtail_flip'].any()


def test_detect_headtail_flips_per_id_independent():
    n = 10
    rows = []
    for fid in [0, 1]:
        ori_deg = np.full(n, 20.0)
        if fid == 1:
            ori_deg[5:] = -160.0  # flip only for id 1
        rows.append(pd.DataFrame({'id': fid, 'frame': np.arange(n), 'sex': 'm',
                                  'strain': 'X', 'ori': np.deg2rad(ori_deg)}))
    df = pd.concat(rows, ignore_index=True)
    out = qc.detect_headtail_flips(df)
    assert not out.loc[out['id'] == 0, 'headtail_flip'].any()
    assert out.loc[out['id'] == 1, 'headtail_flip'].sum() == 1


# ---------------------------------------------------------------------------
# find_flip_window + plot
# ---------------------------------------------------------------------------
def test_find_flip_window(flip_df):
    f_start, n_flips = qc.find_flip_window(flip_df, n_frames=20)
    assert f_start == 0
    assert n_flips == 2


def test_plot_headtail_flips_returns_summary(flip_df):
    fig, summary = qc.plot_headtail_flips(flip_df, acq='test')
    assert set(summary.columns) >= {'id', 'sex', 'pct_flip_frames', 'n_flip_frames'}
    assert summary.loc[summary['id'] == 0, 'n_flip_frames'].iloc[0] == 2


# ---------------------------------------------------------------------------
# Per-flypair output directory layout
# ---------------------------------------------------------------------------
def test_pair_qc_dir_builds_per_pair_path(tmp_path):
    d = qc.pair_qc_dir(str(tmp_path), 'myacq', 3)
    assert d == os.path.join(str(tmp_path), 'myacq', 'pair3')
    assert os.path.isdir(d)


def test_acq_qc_dir_builds_acq_path(tmp_path):
    d = qc.acq_qc_dir(str(tmp_path), 'myacq')
    assert d == os.path.join(str(tmp_path), 'myacq')
    assert os.path.isdir(d)
