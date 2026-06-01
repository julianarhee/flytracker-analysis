"""Tests for the shared QC primitives in libs/qc.py.

Covers the head-tail flip resolution policy (forward-velocity alignment with a
start-of-video fallback), orientation exclusion, and a headless smoke test of
the bout-video writer.
"""
import os

import matplotlib
matplotlib.use('Agg')  # headless

import numpy as np
import pandas as pd
import pytest

import libs.qc as qc


# ---------------------------------------------------------------------------
# resolve_flip_chunks
# ---------------------------------------------------------------------------
def _walking_fly(ori_by_chunk, step_px=5.0, fps=60, n_per_chunk=10):
    """Fly walking steadily in +x; ori is constant within each listed chunk.

    Position is continuous (the fly really walks forward); only `ori` flips, so
    chunks alternate between aligned and anti-aligned with the heading.
    """
    n = len(ori_by_chunk) * n_per_chunk
    pos_x = np.arange(n, dtype=float) * step_px
    pos_y = np.zeros(n)
    ori = np.concatenate([np.full(n_per_chunk, o) for o in ori_by_chunk])
    return pd.DataFrame({'id': 0, 'frame': np.arange(n),
                         'pos_x': pos_x, 'pos_y': pos_y, 'ori': ori})


def test_resolve_flips_velocity_flags_backward_chunk():
    # Walking +x (heading 0). Middle chunk faces backward (pi) -> flipped.
    df = _walking_fly([0.0, np.pi, 0.0])
    out, summary = qc.resolve_flip_chunks(df, fps=60)

    assert summary['parity_source'] == 'velocity'
    assert summary['n_chunks'] == 3
    assert summary['n_flipped_chunks'] == 1
    # Only the middle chunk (chunk_id == 1) is flipped.
    assert out.loc[out['chunk_id'] == 1, 'ori_flipped'].all()
    assert not out.loc[out['chunk_id'] == 0, 'ori_flipped'].any()
    assert not out.loc[out['chunk_id'] == 2, 'ori_flipped'].any()


def test_resolve_flips_velocity_per_chunk_independent():
    # Each chunk is classified by its OWN motion alignment, not by an
    # alternating parity from an anchor: the two pi-facing (backward) end chunks
    # are flipped and the forward middle chunk is kept, even though that breaks
    # strict alternation. (min_moving_frames=5 so the short first chunk, whose
    # first frame has zero displacement, is still decidable.)
    df = _walking_fly([np.pi, 0.0, np.pi])
    out, summary = qc.resolve_flip_chunks(df, fps=60, min_moving_frames=5)

    assert summary['parity_source'] == 'velocity'
    assert not out.loc[out['chunk_id'] == 1, 'ori_flipped'].any()
    assert out.loc[out['chunk_id'] == 0, 'ori_flipped'].all()
    assert out.loc[out['chunk_id'] == 2, 'ori_flipped'].all()


def test_resolve_flips_two_aligned_chunks_not_flipped():
    # Regression: a spurious flip-and-back splits a forward-walking track into
    # two chunks that are BOTH aligned with motion. Neither should be flipped
    # (the old alternating-parity rule wrongly flagged the second chunk).
    df = _walking_fly([0.0, 0.0])           # two chunks, but...
    df.loc[20, 'ori'] = np.pi               # single-frame spurious flip at 20
    df.loc[20, 'frame'] = 20
    out, summary = qc.resolve_flip_chunks(df, fps=60, min_moving_frames=5)
    # The lone frame 20 is its own tiny chunk; the two long chunks stay aligned.
    long_chunks = out.groupby('chunk_id').filter(lambda g: len(g) >= 5)
    assert not long_chunks['ori_flipped'].any()


def test_resolve_flips_undecidable_chunk_is_kept():
    # A stationary (undecidable) chunk between two decidable forward chunks is
    # KEPT (not flipped), even though a flip bounds it — we never NaN without
    # positive backward evidence.
    npc = 12
    # chunk 0: walking +x forward; chunk 1: stationary (flip in, flip out)
    px = np.concatenate([np.arange(npc) * 5.0, np.full(npc, npc * 5.0)])
    ori = np.concatenate([np.zeros(npc), np.full(npc, np.pi)])  # flip at boundary
    df = pd.DataFrame({'id': 0, 'frame': np.arange(2 * npc),
                       'pos_x': px, 'pos_y': np.zeros(2 * npc), 'ori': ori})
    out, summary = qc.resolve_flip_chunks(df, fps=60, min_moving_frames=5)
    assert summary['parity_source'] == 'velocity'
    assert not out['ori_flipped'].any()  # forward chunk decided; stationary kept


def test_resolve_flips_no_flip_single_chunk():
    df = _walking_fly([0.0])
    out, summary = qc.resolve_flip_chunks(df, fps=60)
    assert summary['n_chunks'] == 1
    assert summary['n_flipped_chunks'] == 0
    assert not out['ori_flipped'].any()


def test_resolve_flips_start_anchor_when_stationary():
    # Stationary fly: no frame exceeds the speed threshold, so velocity can't
    # decide -> fall back to anchoring the first chunk as correct.
    n = 20
    ori = np.concatenate([np.zeros(10), np.full(10, np.pi)])  # one flip at 10
    df = pd.DataFrame({'id': 0, 'frame': np.arange(n),
                       'pos_x': np.zeros(n), 'pos_y': np.zeros(n), 'ori': ori})
    out, summary = qc.resolve_flip_chunks(df, fps=60, speed_thresh_px_s=20.0)

    assert summary['parity_source'] == 'start_anchor'
    assert summary['n_flipped_chunks'] == 1
    assert not out.loc[out['chunk_id'] == 0, 'ori_flipped'].any()  # anchor correct
    assert out.loc[out['chunk_id'] == 1, 'ori_flipped'].all()


# ---------------------------------------------------------------------------
# find_flip_window (courtship restriction)
# ---------------------------------------------------------------------------
def test_find_flip_window_restricts_to_courtship():
    # A dense flip cluster OUTSIDE courtship (frames 2-5) and a single flip
    # INSIDE courtship (frame 24). With courtship_col set, the courtship window
    # must be chosen even though it has fewer flips.
    n = 40
    ori = np.zeros(n)
    ori[2:6] = [np.pi, 0.0, np.pi, 0.0]  # 4 flips near frame 2-5 (non-courtship)
    ori[24] = np.pi                       # 1 flip at frame 24 (courtship)
    ori[25] = np.pi
    courtship = np.zeros(n, int)
    courtship[20:35] = 1
    df = pd.DataFrame({'id': 0, 'frame': np.arange(n), 'ori': ori,
                       'courtship': courtship})

    # Unrestricted: densest window is the frame 2-5 cluster.
    f_start_all, n_all = qc.find_flip_window(df, n_frames=10)
    assert f_start_all <= 5

    # Restricted to courtship: window must sit inside frames 20-34.
    f_start_c, n_c = qc.find_flip_window(df, n_frames=10, courtship_col='courtship')
    assert 20 <= f_start_c <= 25
    assert n_c >= 1


def test_find_flip_window_courtship_fallback_when_none():
    # No fully-courtship window exists -> fall back to the unrestricted densest
    # window (still finds the real flip rather than reporting nothing).
    n = 20
    ori = np.zeros(n)
    ori[5] = np.pi
    df = pd.DataFrame({'id': 0, 'frame': np.arange(n), 'ori': ori,
                       'courtship': np.zeros(n, int)})
    f_start, n_flips = qc.find_flip_window(df, n_frames=8, courtship_col='courtship')
    assert n_flips >= 1  # found the real flip via fallback (frame 5 in/out = 2)
    assert f_start <= 5


# ---------------------------------------------------------------------------
# exclude_flipped_orientation
# ---------------------------------------------------------------------------
def test_exclude_flipped_orientation_nans_flipped_chunk():
    df = _walking_fly([0.0, np.pi, 0.0])
    df['ang_vel_fly'] = 1.0
    out, summary = qc.exclude_flipped_orientation(
        df, cols=('ori', 'ang_vel_fly'), fps=60)

    flipped = out['chunk_id'] == 1
    assert out.loc[flipped, 'ori'].isna().all()
    assert out.loc[flipped, 'ang_vel_fly'].isna().all()
    assert out.loc[~flipped, 'ori'].notna().all()


def test_nan_flipped_orientation_per_id_preserves_index():
    # Two flies: id 0 has a flipped middle chunk; id 1 walks forward throughout.
    f0 = _walking_fly([0.0, np.pi, 0.0]); f0['id'] = 0
    f1 = _walking_fly([0.0]); f1['id'] = 1
    df = pd.concat([f0, f1], ignore_index=True)
    df.index = df.index + 100  # non-default index to confirm preservation

    out, summaries = qc.nan_flipped_orientation_per_id(df, fps=60)

    # Index/order preserved, no chunk_id/ori_flipped columns leaked in.
    assert out.index.equals(df.index)
    assert 'chunk_id' not in out.columns and 'ori_flipped' not in out.columns
    # id 0: only the backward middle third is NaN'd.
    o0 = out[out['id'] == 0].sort_values('frame')['ori'].values
    assert np.isnan(o0[10:20]).all()
    assert np.isfinite(o0[:10]).all() and np.isfinite(o0[20:]).all()
    # id 1: nothing flipped.
    assert out[out['id'] == 1]['ori'].notna().all()
    assert summaries[1]['n_flipped_chunks'] == 0


# ---------------------------------------------------------------------------
# resolve_orientation (method dispatch)
# ---------------------------------------------------------------------------
def test_resolve_orientation_velocity_nans_flipped_chunk():
    df = _walking_fly([0.0, np.pi, 0.0])
    out, info = qc.resolve_orientation(df, method='velocity', fps=60, min_moving_frames=5)
    assert info['method'] == 'velocity'
    assert info['per_fly'] is not None
    # The backward middle third is NaN'd; ends kept.
    o = out.sort_values('frame')['ori'].values
    assert np.isnan(o[10:20]).all()
    assert np.isfinite(o[:10]).all() and np.isfinite(o[20:]).all()


def test_resolve_orientation_wing_nans_wingless_frames():
    n = 10
    df = pd.DataFrame({
        'id': 0, 'frame': np.arange(n), 'pos_x': np.arange(n) * 5.0,
        'pos_y': np.zeros(n), 'ori': np.zeros(n),
        'wing_l_x': np.r_[np.nan * np.ones(4), np.arange(6)],
        'wing_l_y': np.r_[np.nan * np.ones(4), np.arange(6)],
        'wing_r_x': np.r_[np.nan * np.ones(4), np.arange(6)],
        'wing_r_y': np.r_[np.nan * np.ones(4), np.arange(6)],
    })
    out, info = qc.resolve_orientation(df, method='wing')
    assert info['method'] == 'wing'
    assert out['ori'].iloc[:4].isna().all()   # no wing info -> NaN
    assert out['ori'].iloc[4:].notna().all()


def test_resolve_orientation_none_keeps_everything():
    df = _walking_fly([0.0, np.pi, 0.0])
    out, info = qc.resolve_orientation(df, method='none')
    assert info['method'] == 'none'
    assert info['frac_ori_nan'] == 0.0
    assert out['ori'].notna().all()


def test_resolve_orientation_bad_method_raises():
    df = _walking_fly([0.0])
    with pytest.raises(ValueError):
        qc.resolve_orientation(df, method='magnets')


# ---------------------------------------------------------------------------
# plot_theta_error_ori_vs_heading
# ---------------------------------------------------------------------------
def test_plot_theta_error_ori_vs_heading_returns_r():
    rng = np.random.default_rng(0)
    ori = rng.uniform(-np.pi, np.pi, 500)
    df = pd.DataFrame({'theta_error': ori,
                       'theta_error_heading': ori + rng.normal(0, 0.05, 500)})
    fig, r = qc.plot_theta_error_ori_vs_heading(df)
    assert fig is not None
    assert r > 0.9  # strongly correlated by construction


def test_plot_theta_error_ori_vs_heading_missing_col():
    df = pd.DataFrame({'theta_error': [0.1, 0.2, 0.3]})
    fig, r = qc.plot_theta_error_ori_vs_heading(df)
    assert fig is None and np.isnan(r)


def test_recompute_theta_error_heading_matches_circular_distance():
    import libs.utils as util
    df = pd.DataFrame({'abs_ang_between': [0.5, -2.0, 3.0, 0.0],
                       'heading': [0.1, 1.0, -3.0, np.pi]})
    out = qc.recompute_theta_error_heading(df)
    expected = util.circular_distance(df['abs_ang_between'], df['heading'])
    assert np.allclose(out, expected)
    assert np.all(np.abs(out) <= np.pi + 1e-9)  # wrapped to [-pi, pi]


# ---------------------------------------------------------------------------
# save_bout_video (headless smoke test)
# ---------------------------------------------------------------------------
def test_save_bout_video_writes_file(tmp_path):
    cv2 = pytest.importorskip('cv2')

    # Write a tiny synthetic video.
    vid_path = os.path.join(str(tmp_path), 'movie.avi')
    w, h, nframes = 64, 64, 6
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter(vid_path, fourcc, 30, (w, h))
    assert writer.isOpened()
    for _ in range(nframes):
        writer.write(np.full((h, w, 3), 128, dtype=np.uint8))
    writer.release()

    flydf = pd.DataFrame({
        'frame': np.arange(nframes),
        'pos_x': np.full(nframes, 32.0),
        'pos_y': np.full(nframes, 32.0),
        'ori': np.zeros(nframes),
        'ang_vel_fly': np.linspace(-1, 1, nframes),
    })
    targdf = flydf.assign(pos_x=20.0, pos_y=40.0)

    out_path = os.path.join(str(tmp_path), 'sub', 'bout.avi')
    ret = qc.save_bout_video(str(tmp_path), flydf, 1, 4, out_path,
                             targdf=targdf, color_col='ang_vel_fly',
                             ori_col='ori', fps=30, output_fps=10)
    assert ret == out_path
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_save_bout_video_missing_video_returns_none(tmp_path):
    flydf = pd.DataFrame({'frame': [0, 1], 'pos_x': [1.0, 1.0],
                          'pos_y': [1.0, 1.0], 'ori': [0.0, 0.0]})
    out_path = os.path.join(str(tmp_path), 'bout.avi')
    # An empty dir has no video -> None, no file written.
    empty = os.path.join(str(tmp_path), 'empty')
    os.makedirs(empty)
    assert qc.save_bout_video(empty, flydf, 0, 1, out_path) is None
