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
