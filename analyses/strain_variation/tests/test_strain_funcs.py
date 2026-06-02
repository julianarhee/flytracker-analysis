"""Tests for analyses/strain_variation/src/strain_funcs.py.

These cover the pure preprocessing logic (no mounted data volume required):
arena ID mapping, sex/strain assignment, multi-fly -actions.mat parsing,
copulation-frame pairing, action-frame assignment, and JAABA stacking. A
synthetic `-actions.mat` is built with scipy.io.savemat so it round-trips
through the same access pattern as the real FlyTracker files.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import scipy.io
import pytest

import analyses.strain_variation.src.strain_funcs as sf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synth_actions_mat(tmp_path):
    """A 4-fly synthetic -actions.mat matching the real FlyTracker structure.

    Annotations are on male id 0: chasing 10-20, courtship 5-50, copulation @100.
    """
    behs_names = ['chasing', 'copulation ', 'courtship', 'unilateral extension']
    n_flies, n_behs = 4, len(behs_names)

    behs = np.empty((n_behs, 1), dtype=object)
    for i, n in enumerate(behs_names):
        behs[i, 0] = np.array([n])

    bouts = np.empty((n_flies, n_behs), dtype=object)
    for i in range(n_flies):
        for j in range(n_behs):
            bouts[i, j] = np.zeros((0, 3))
    bouts[0, 0] = np.array([[10, 20, 2]])    # chasing
    bouts[0, 2] = np.array([[5, 50, 2]])     # courtship
    bouts[0, 1] = np.array([[100, 100, 2]])  # copulation onset

    fpath = os.path.join(str(tmp_path), 'synth-actions.mat')
    scipy.io.savemat(fpath, {'behs': behs, 'bouts': bouts})
    return fpath


@pytest.fixture
def two_pair_trk():
    """Minimal tracking df: 2 fly-pairs (ids 0,1 and 4,5 per the 2x2 LUT)."""
    n = 10
    frames = np.arange(n)
    rows = []
    for fly_id in [0, 1, 4, 5]:
        rows.append(pd.DataFrame({'id': fly_id, 'frame': frames,
                                  'pos_x': fly_id * 100.0 + frames,
                                  'pos_y': fly_id * 50.0}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Arena / fly-ID assignment
# ---------------------------------------------------------------------------
def test_meta_flynum_to_ft_id_2x2():
    lut = sf.meta_flynum_to_ft_id('2x2')
    assert lut == {1: 0, 2: 4, 3: 2, 4: 6}
    # All male ids are even (female = male + 1).
    assert all(v % 2 == 0 for v in lut.values())


def test_meta_flynum_to_ft_id_bad():
    with pytest.raises(ValueError):
        sf.meta_flynum_to_ft_id('5x5')


def test_assign_sex_even_male_odd_female(two_pair_trk):
    out = sf.assign_sex(two_pair_trk.copy())
    assert set(out.loc[out['id'] == 0, 'sex']) == {'m'}
    assert set(out.loc[out['id'] == 1, 'sex']) == {'f'}
    assert set(out.loc[out['id'] == 4, 'sex']) == {'m'}
    assert set(out.loc[out['id'] == 5, 'sex']) == {'f'}


def test_assign_frame_number(two_pair_trk):
    out = sf.assign_frame_number(two_pair_trk.copy())
    for fly_id, df_ in out.groupby('id'):
        assert list(df_['frame']) == list(range(len(df_)))


def test_assign_strain_to_multichamber(two_pair_trk):
    meta = pd.DataFrame({
        'fly_num': [1, 2],
        'strain_male': ['SD105N', 'RG38N'],
        'strain_female': ['SD105N', 'RG38N'],
        'species_male': ['Dmel', 'Dmel'],
    })
    out = sf.assign_strain_to_multichamber(two_pair_trk.copy(), meta, array_size='2x2')
    # fly_num 1 -> male id 0, female id 1 -> SD105N
    assert set(out.loc[out['id'].isin([0, 1]), 'fly_pair']) == {1}
    assert set(out.loc[out['id'].isin([0, 1]), 'strain']) == {'SD105N'}
    # fly_num 2 -> male id 4, female id 5 -> RG38N
    assert set(out.loc[out['id'].isin([4, 5]), 'fly_pair']) == {2}
    assert set(out.loc[out['id'].isin([4, 5]), 'strain']) == {'RG38N'}


# ---------------------------------------------------------------------------
# Behavior-name normalization
# ---------------------------------------------------------------------------
def test_normalize_behavior_name():
    assert sf.normalize_behavior_name('copulation ') == 'copulation'
    assert sf.normalize_behavior_name('unilateral extension') == 'unilateral_extension'
    assert sf.normalize_behavior_name('Chasing') == 'chasing'


# ---------------------------------------------------------------------------
# -actions.mat parsing (multi-fly)
# ---------------------------------------------------------------------------
def test_ft_actions_to_bout_df_courtship_set(synth_actions_mat):
    bouts = sf.ft_actions_to_bout_df(synth_actions_mat)
    # Only courtship-set behaviors that are present, all on fly id 0.
    assert set(bouts['action']) == {'chasing', 'courtship', 'copulation'}
    assert set(bouts['id']) == {0}
    chase = bouts[bouts['action'] == 'chasing'].iloc[0]
    assert (chase['start'], chase['end']) == (10, 20)
    assert bouts['boutnum'].is_unique


def test_ft_actions_to_bout_df_filter(synth_actions_mat):
    only_chase = sf.ft_actions_to_bout_df(synth_actions_mat, behaviors=['chasing'])
    assert set(only_chase['action']) == {'chasing'}

    all_beh = sf.ft_actions_to_bout_df(synth_actions_mat, behaviors='all')
    # 'unilateral extension' has no bouts here, so it won't appear, but the
    # other three should still be present when keeping all behaviors.
    assert {'chasing', 'courtship', 'copulation'}.issubset(set(all_beh['action']))


def test_get_copulation_frames_pairs_partner(synth_actions_mat):
    cop = sf.get_copulation_frames(synth_actions_mat)
    # Male id 0 copulates at frame 100; female id 1 inherits the same frame.
    assert cop[0] == 100
    assert cop[1] == 100
    # Untouched flies are -1.
    assert cop[2] == -1 and cop[3] == -1


def test_assign_action_frames_to_df_per_id(synth_actions_mat):
    # df with two flies; annotations only on id 0.
    frames = np.arange(60)
    df = pd.concat([
        pd.DataFrame({'id': 0, 'frame': frames}),
        pd.DataFrame({'id': 1, 'frame': frames}),
    ], ignore_index=True)
    actions = sf.ft_actions_to_bout_df(synth_actions_mat)
    out = sf.assign_action_frames_to_df(df, actions)

    # chasing 10-20 marked on id 0 only.
    m0 = out[out['id'] == 0]
    assert m0.loc[m0['frame'].between(10, 20), 'chasing'].eq(1).all()
    assert m0.loc[m0['frame'] == 25, 'chasing'].eq(0).all()
    # id 1 (female) never flagged for the male's annotation.
    assert out.loc[out['id'] == 1, 'chasing'].eq(0).all()
    # boutnum recorded inside the bout.
    assert m0.loc[m0['frame'] == 15, 'chasing_boutnum'].notna().all()


# ---------------------------------------------------------------------------
# JAABA stacking
# ---------------------------------------------------------------------------
def test_stack_jaaba_scores():
    scores = pd.DataFrame(np.arange(6).reshape(3, 2), columns=[0, 1])
    stacked = sf.stack_jaaba_scores(scores)
    assert list(stacked.columns) == ['frame', 'id', 'score']
    assert len(stacked) == 6
    row = stacked[(stacked['frame'] == 2) & (stacked['id'] == 1)]
    assert row['score'].iloc[0] == 5


def _make_scores_mat(tmp_path, lengths, tstart=None):
    """Write a synthetic JAABA scores_*.mat with the given per-fly lengths.

    `lengths` are the per-fly score-vector lengths (ragged allowed). Fields are
    written in the exact positional order load_jaaba_from_mat expects.
    """
    import os
    n = len(lengths)
    tstart = tstart or [1] * n
    scores = np.empty((1, n), dtype=object)
    post = np.empty((1, n), dtype=object)
    t0s = np.empty((1, n), dtype=object)
    t1s = np.empty((1, n), dtype=object)
    for i, L in enumerate(lengths):
        scores[0, i] = np.arange(L, dtype=float).reshape(1, L)
        post[0, i] = np.zeros((1, L))
        t0s[0, i] = np.zeros((1, 0))
        t1s[0, i] = np.zeros((1, 0))
    allscores = {
        'scores': scores,
        'tStart': np.array(tstart, dtype=float).reshape(1, n),
        'tEnd': np.array([s + L - 1 for s, L in zip(tstart, lengths)],
                         dtype=float).reshape(1, n),
        'postprocessed': post,
        'postprocessparams': np.zeros((1, 1)),
        't0s': t0s,
        't1s': t1s,
        'scoreNorm': np.ones((1, 1)),
    }
    fpath = os.path.join(str(tmp_path), 'scores_test.mat')
    scipy.io.savemat(fpath, {'allScores': allscores})
    return fpath


def test_load_jaaba_equal_lengths(tmp_path):
    fpath = _make_scores_mat(tmp_path, lengths=[5, 5, 5])
    df = sf.load_jaaba_from_mat(fpath)
    assert df.shape == (5, 3)
    assert df.notna().all().all()
    # Column i holds 0..L-1 by construction.
    assert list(df[0]) == [0, 1, 2, 3, 4]


def test_load_jaaba_ragged_lengths_nan_padded(tmp_path):
    # Fly 2 ends early (len 3); the common length is the max (5).
    fpath = _make_scores_mat(tmp_path, lengths=[5, 5, 3])
    df = sf.load_jaaba_from_mat(fpath)
    assert df.shape == (5, 3)
    # Short fly: real values at frames 0-2, NaN at 3-4.
    assert list(df[2].iloc[:3]) == [0, 1, 2]
    assert df[2].iloc[3:].isna().all()
    # stack() drops the NaN-padded trailing frames.
    stacked = sf.stack_jaaba_scores(df)
    assert len(stacked) == 5 + 5 + 3


def test_load_jaaba_respects_tstart_offset(tmp_path):
    # Fly 1 starts at frame 3 (tStart=3), length 2 -> occupies frames 2,3.
    fpath = _make_scores_mat(tmp_path, lengths=[5, 2], tstart=[1, 3])
    df = sf.load_jaaba_from_mat(fpath)
    assert df.shape == (5, 2)
    assert df[1].iloc[:2].isna().all()          # frames 0,1 empty
    assert list(df[1].iloc[2:4]) == [0, 1]       # scores placed at offset
    assert df[1].iloc[4:].isna().all()


# ---------------------------------------------------------------------------
# Derived courtship labels (analysis-time)
# ---------------------------------------------------------------------------
def _label_input():
    """Small df with controlled facing_angle + JAABA/manual label columns."""
    return pd.DataFrame({
        'sex': ['m', 'm', 'm', 'f'],
        'facing_angle': [np.deg2rad(5), np.deg2rad(45), np.deg2rad(5), np.deg2rad(5)],
        'vel': [12, 12, 0, 0],
        'targ_pos_theta': [0.0, 0.0, 0.0, 0.0],
        'max_wing_ang': [np.deg2rad(50)] * 4,
        'dist_to_other': [10, 10, 10, 10],
        'jaaba_chasing_binary': [True, False, False, False],
        'jaaba_unilateral_extension_binary': [False, False, True, False],
        'chasing': [1, 0, 0, 0],
        'unilateral_extension': [0, 0, 1, 0],
    })


def test_derive_courtship_labels_jaaba_source():
    out = sf.derive_courtship_labels(_label_input(), source='jaaba',
                                     orienting_angle_deg=10)
    # facing_angle <= 10deg -> orienting (rows 0, 2, 3).
    assert list(out['is_orienting']) == [1, 0, 1, 1]
    # chasing/singing from the jaaba_*_binary columns.
    assert list(out['is_chasing']) == [1, 0, 0, 0]
    assert list(out['is_singing']) == [0, 0, 1, 0]
    # courting = any of the three.
    assert list(out['is_courting']) == [1, 0, 1, 1]
    assert list(out['courtship_sum']) == [2, 0, 2, 1]


def test_derive_courtship_labels_manual_source():
    out = sf.derive_courtship_labels(_label_input(), source='manual')
    # manual columns back chasing/singing instead of jaaba.
    assert list(out['is_chasing']) == [1, 0, 0, 0]
    assert list(out['is_singing']) == [0, 0, 1, 0]


def test_derive_courtship_labels_kinematic_source():
    out = sf.derive_courtship_labels(
        _label_input(), source='kinematic',
        chasing_kws=dict(min_vel=10, max_facing_angle=np.deg2rad(30),
                         min_wing_ang=0, max_dist_to_other=20))
    # Only the male row with vel>=10 AND facing<=30deg qualifies (row 0).
    assert list(out['is_chasing']) == [1, 0, 0, 0]
    assert set(out['is_chasing'].unique()).issubset({0, 1})


def test_derive_courtship_labels_missing_column_raises():
    df = _label_input().drop(columns=['jaaba_chasing_binary'])
    with pytest.raises(KeyError):
        sf.derive_courtship_labels(df, source='jaaba')


def test_derive_courtship_labels_bad_source():
    with pytest.raises(ValueError):
        sf.derive_courtship_labels(_label_input(), source='nope')


def test_derive_courtship_labels_does_not_mutate_input():
    df = _label_input()
    sf.derive_courtship_labels(df, source='jaaba')
    assert 'is_courting' not in df.columns


def test_filter_chasing_jaaba_mode():
    df = _label_input()
    out = sf.filter_chasing(df, use_jaaba=True, beh_type='jaaba_chasing_binary',
                            max_facing_angle=np.deg2rad(90))
    # Only male rows with the binary set and facing within range (row 0).
    assert len(out) == 1
    assert out['sex'].eq('m').all()


def test_filter_chasing_singing_adds_wing_gate():
    df = _label_input()
    # Wing gate above the data's wing angle -> nothing passes.
    out = sf.filter_chasing(df, use_jaaba=True,
                            beh_type='jaaba_unilateral_extension_binary',
                            max_facing_angle=np.deg2rad(90),
                            min_wing_ang=np.deg2rad(80))
    assert len(out) == 0


def test_filter_chasing_kinematic_mode():
    df = _label_input()
    out = sf.filter_chasing(df, use_jaaba=False, min_vel=10,
                            max_facing_angle=np.deg2rad(30), min_wing_ang=0,
                            max_dist_to_other=20)
    assert len(out) == 1
    assert out['sex'].eq('m').all()
