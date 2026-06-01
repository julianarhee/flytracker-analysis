# strain_variation

Processing + QC for Caitlin's 2×2 (8-fly) multichamber *Drosophila* courtship
assays (Dmel and Dyak strains). Four male/female pairs are recorded per
acquisition; each pair is transformed into egocentric / relative metrics and
annotated with manual (`-actions.mat`) and JAABA behaviors.

```
src/process_multichamber.py   load FlyTracker -> relative metrics -> annotations -> parquet
src/qc_checks.py              per-acquisition + per-pair QC (layout, example bout, flips)
src/strain_funcs.py           dataset helpers (arena/sex/strain LUTs, actions, JAABA)
```

## Pipeline

```bash
# one acquisition (writes processed/<acq>.parquet)
python analyses/strain_variation/src/process_multichamber.py --species Dmel --single <acq>

# a species / both (rebuilds the aggregate parquets); --new recomputes cached acqs
python analyses/strain_variation/src/process_multichamber.py --species both --new
```

Outputs live under `<rootdir>/2x2_strains_processed/` (per-acq parquet in
`processed/`, aggregates at the top level). QC figures go under
`qc/<acq>/` (acq-level `arena_layout.png`, `headtail_flips_summary.png`) and
`qc/<acq>/pair<N>/` (per-pair `*_traj.png`, `*_timecourse.png`, `*_bout.avi`,
`headtail_flip_montage.png`).

## Orientation & head-tail flip handling (`--ori_method`)

How FlyTracker orientation / head-tail flips are handled is selectable with
`--ori_method` (default `velocity`). This affects all orientation-derived
metrics (`theta_error`, `targ_pos_theta`, egocentric target position,
`ang_vel_fly`); `feat`-derived `vel`/`ang_vel`/`pos` are unaffected.

| `--ori_method` | Summary | Use when |
|---|---|---|
| **`velocity`** (default) | keep body-axis `ori`; NaN only chunks moving *opposite* their orientation (a real flip) | forward-locomotion assays (courtship/pursuit) |
| **`wing`** | NaN `ori` where wings undetected (legacy `filter_ori=True`) | substantial sideways/backward motion |
| **`none`** | raw FlyTracker `ori`, unchanged | when handling polarity downstream |

`process_single_acquisition` is a thin caller of the shared, dataset-agnostic
**`libs.qc.resolve_orientation`**. The full rationale — why `velocity` is the
default (data retention + the behaviour-correlation bias of `wing`), exactly how
each chunk is classified, when to switch, and the deliberately **deferred
sideways-aware "middle-ground" option** — is documented once in
[`libs/README.md`](../../libs/README.md#orientation-heading--head-tail-flips-qcresolve_orientation),
alongside the rest of the shared FlyTracker preprocessing.
