"""Shared fixtures for the strain_variation analysis-layer tests.

These build a synthetic processed-parquet-style DataFrame (the kind
`process_multichamber.py` writes) so the analysis modules can be tested with no
mounted data volume. Plotting tests force the Agg backend.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_tracking_df():
    """Synthetic multichamber df: 3 strains x 2 pairs x (male, female).

    Includes the relative-metric and JAABA-binary columns the analysis layer
    reads. Values are seeded so each behavior column has a mix of 0/1.
    """
    rng = np.random.default_rng(0)
    n = 200
    rows = []
    for sp, strain in [('Dmel', 'CS'), ('Dmel', 'SD'), ('Dyak', 'RL')]:
        for fp in [1, 2]:
            for sex, fid in [('m', 0), ('f', 1)]:
                rows.append(pd.DataFrame({
                    'species': sp, 'strain': strain,
                    'acquisition': '{}_acq'.format(sp), 'fly_pair': fp,
                    'sex': sex, 'id': fid, 'frame': np.arange(n),
                    'facing_angle': np.abs(rng.normal(0.2, 0.3, n)),
                    'vel': np.abs(rng.normal(8, 3, n)),
                    'dist_to_other': np.abs(rng.normal(10, 5, n)),
                    'targ_pos_theta': rng.uniform(-1, 1, n),
                    'max_wing_ang': np.abs(rng.normal(0.5, 0.3, n)),
                    'targ_rel_pos_x': rng.normal(0, 100, n),
                    'targ_rel_pos_y': rng.normal(0, 100, n),
                    'jaaba_chasing_binary': rng.random(n) > 0.6,
                    'jaaba_unilateral_extension_binary': rng.random(n) > 0.7,
                }))
    return pd.concat(rows, ignore_index=True)
