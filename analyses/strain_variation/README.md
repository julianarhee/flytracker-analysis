# strain_variation

Processing, QC, and analysis for Caitlin's 2×2 (8-fly) multichamber
*Drosophila* courtship assays (Dmel and Dyak strains). Four male/female pairs
are recorded per acquisition; each pair is transformed into egocentric /
relative metrics and annotated with manual (`-actions.mat`) and automated JAABA
behaviors.

## Source files

| File | Role |
|---|---|
| `src/strain_funcs.py` | Dataset constants, arena/sex/strain LUTs, multi-fly `-actions.mat` parser, JAABA loader, courtship label derivation, subset-selection helpers |
| `src/process_multichamber.py` | Preprocessing CLI: FlyTracker → relative metrics → actions/JAABA → per-acq parquet |
| `src/qc_checks.py` | Per-acquisition + per-pair QC (arena layout, example bout, head-tail flip montage) |
| `src/strain_metrics.py` | Tidy per-pair summary metrics (p(behavior), velocity, interfly distance, distance-resolved behavior) |
| `src/strain_plots.py` | Grouped boxplots + Mann-Whitney annotation; re-exports `label_figure`/`set_sns_style` |
| `src/spatial_maps.py` | Egocentric occupancy maps (male-from-female, female-from-male views) |
| `src/compare_jaaba_to_heuristics.py` | JAABA vs kinematic label agreement: per-pair metrics (kappa, Jaccard, F1), diagnostic KDE plots, threshold visualizations |
| `src/run_strain_analysis.py` | Interactive `#%%` orchestrator for the full dataset: loads aggregate parquet, derives labels, produces all strain-comparison + JAABA figures |
| `src/run_subset_analysis.py` | Same pipeline but for a configurable **subset** of acquisitions (pilot / testset); dataset selection lives at the top of this file |

## Typical pipeline

### 1. Preprocessing

```bash
# One acquisition (saves processed/<acq>.parquet)
python analyses/strain_variation/src/process_multichamber.py --species Dmel --single <acq>

# All acquisitions, serial (--new forces recomputing cached parquets)
python analyses/strain_variation/src/process_multichamber.py --species both --new

# Parallel — 4 workers (recommended for external/USB drives)
python analyses/strain_variation/src/process_multichamber.py --species both --new --workers 4

# Parallel — all CPU cores (recommended for local SSD)
python analyses/strain_variation/src/process_multichamber.py --species both --new --workers 0
```

`--workers` behaviour:

| Value | Effect |
|---|---|
| `1` (default) | Serial — identical to original behaviour |
| `N > 1` | N worker processes, each handling one acquisition at a time |
| `0` | One worker per CPU core (`multiprocessing.cpu_count()`) |

Workers run under the macOS `spawn` context (safe with numpy/OpenCV/matplotlib in
the parent). Each worker saves its per-acquisition parquet and exits; the main
process loads them all at the end. For a spinning-disk or USB-attached drive,
4–6 workers is usually optimal; a local NVMe SSD can use all cores.

Outputs under `<rootdir>/2x2_strains_processed/`:

```
processed/<acq>.parquet        per-acquisition (cached; one file per acquisition)
Dmel_2x2_strains.parquet       per-species aggregate
Dyak_2x2_strains.parquet
mel_yak_2x2_strains.parquet    combined Dmel+Dyak
```

### 2. QC

```bash
python analyses/strain_variation/src/qc_checks.py --species Dmel --acq <acq>
```

Figures under `<rootdir>/2x2_strains_processed/qc/<acq>/` (arena layout, flip
summary) and `qc/<acq>/pair<N>/` (trajectory, timecourse, bout clip, flip
montage).

### 3a. Full-dataset analysis (`run_strain_analysis.py`)

Open in VS Code and run cells interactively (`#%%` style), or execute as a
script.

**Required input** — the per-species parquets (produced by step 1):

```
<rootdir>/2x2_strains_processed/Dmel_2x2_strains.parquet
<rootdir>/2x2_strains_processed/Dyak_2x2_strains.parquet
```

The script loads each file separately through Arrow, filters to male rows before
converting to pandas, and then concatenates.  This avoids the OOM kill that
occurs when the full combined dataset (both sexes, 30–40 M rows) is materialised
all at once:

- Arrow columnar format is 3–5× smaller than pandas in RAM
- Male-only filter halves the row count before the pandas conversion
- One species is in Arrow at a time; the table is freed immediately after filtering
- A slim female table (spatial-map columns only) is retained for the
  female-centered occupancy map

The parquets are read with `pyarrow.parquet.ParquetFile` (footer schema) rather
than `pd.read_parquet` to avoid a schema-fragmentation issue where the JAABA
`unilateral_extension` columns are silently dropped by the dataset scanner when
row groups were written at different times.

**CLI flags:**

```bash
python analyses/strain_variation/src/run_strain_analysis.py \
    --labels kinematic          # jaaba (default) | kinematic | manual
    --orienting-angle 15        # facing-angle threshold in degrees (default: 10)
    --all-frames                # p(behavior) over all frames, not just courting
    --rootdir /path/to/data     # override default mount point
```

**Config variables** (also editable directly at the top of the config cell):

| Variable | CLI flag | Default | Effect |
|---|---|---|---|
| `LABEL_SOURCE` | `--labels` | `'jaaba'` | `'jaaba'` / `'kinematic'` / `'manual'` |
| `ORIENTING_ANGLE_DEG` | `--orienting-angle` | `10` | facing-angle threshold (degrees) |
| `COURTING_FRAMES_ONLY` | `--all-frames` (negates) | `True` | if True, p(behavior\|courtship) |
| `rootdir` | `--rootdir` | `sf.ROOTDIR` | data root; update if mount differs |

**Output figures** written to `<rootdir>/2x2_strains_processed/figures/`:

```
figures/strain_comparison/
    mean_vel_by_behavior_per_strain.png     velocity during all / chasing / singing
    p-behaviors_courtframes.png             p(chasing), p(singing), p(orienting) per strain
    dist_to_other_courtframes.png           interfly distance during courtship
    p-behaviors_v_binned_dist_Dmel.png      p(behavior) vs. distance bin, per species
    p-behaviors_v_binned_dist_Dyak.png
    male-rel-pos_all-pairs_Dmel.png         egocentric male position (female-centered)
    male-rel-pos_all-pairs_Dyak.png
    female-rel-pos_all-pairs_Dmel.png       egocentric female position (male-centered)
    female-rel-pos_all-pairs_Dyak.png

figures/jaaba_vs_heuristic/
    <acq>_jaaba_vs_heuristic.png            per-acquisition p(beh) bars + confusion breakdown
    pbeh_scatter_jaaba_vs_kin.png           scatter: p(beh) JAABA vs kinematic, all pairs
    agreement_metrics_by_species.png        kappa / Jaccard / F1 boxplots by species × behavior
    jaaba_vs_heuristic_summary.csv          tidy summary table (one row per pair × behavior)
    chasing_kin_threshold_diagnostics.png   KDE per agreement category for each gated variable
    chasing_unconsidered_features.png       KDE for variables absent from the kinematic gate
```

### 3b. Subset / testset analysis (`run_subset_analysis.py`)

**Where to edit which data are included:** the `SUBSET_ACQUISITIONS` dict near
the top of the file (search for the `Dataset selection` cell). Two modes:

- **Manual** — fill in `SUBSET_ACQUISITIONS` explicitly; this is the
  reproducible, version-controlled record of the testset.
- **Auto** — leave `SUBSET_ACQUISITIONS = None`; the script calls
  `sf.select_subset_acquisitions()` using `N_STRAINS` / `N_ACQS_PER_STRAIN`,
  prints the result, and you can paste it back in to lock it.

```bash
python analyses/strain_variation/src/run_subset_analysis.py \
    --labels kinematic          # jaaba (default) | kinematic | manual
    --orienting-angle 15        # facing-angle threshold (default: 10)
    --all-frames                # p(behavior) over all frames, not just courting
    --tag testset_kin           # output tag (default: testset)
    --new                       # force reprocessing of cached parquets
    --rootdir /path/to/data     # override default mount point
```

**Key config variables** (also editable at the top of the file):

| Variable | CLI flag | Default | Effect |
|---|---|---|---|
| `LABEL_SOURCE` | `--labels` | `'jaaba'` | `'jaaba'` / `'kinematic'` / `'manual'` |
| `SUBSET_TAG` | `--tag` | `'testset'` | Tag on every output path; change to version runs |
| `CREATE_NEW` | `--new` | `False` | Force reprocessing of cached per-acquisition parquets |
| `ORIENTING_ANGLE_DEG` | `--orienting-angle` | `10` | facing-angle threshold (degrees) |
| `COURTING_FRAMES_ONLY` | `--all-frames` (negates) | `True` | p(behavior\|courtship) |
| `SUBSET_ACQUISITIONS` | — (edit in file) | (dict) | Which acquisitions; `None` = auto-select |

**Memory management** — multi-chamber acquisitions contain data for all fly pairs
in all chambers, not just the subset strains.  The script filters the loaded data
in two stages before any heavy computation:

1. **Sex filter** — only male (focal fly) rows are kept immediately after loading
   each species' data.  All downstream metrics (`strain_metrics`, `compare_jaaba`,
   spatial maps) operate on the male's perspective; female rows are dropped to
   halve the in-memory footprint.  The one female-centered spatial map
   (`male_position_from_female_view`) is skipped; re-enable it by removing the
   sex filter if that view is needed.
2. **Strain filter** — after concatenating across species, rows belonging to
   strains outside `SUBSET_ACQUISITIONS` are removed.

All testset outputs are tagged (default `testset`) so they never overwrite
full-pipeline files:

```
mel_yak_2x2_strains_testset.parquet
figures/strain_comparison_testset/
figures/jaaba_vs_heuristic_testset/
    ├── pbeh_scatter_jaaba_vs_kin.png
    ├── agreement_metrics_by_species.png
    ├── chasing_kin_threshold_diagnostics.png   ← gate variable distributions
    ├── chasing_unconsidered_features.png       ← variables JAABA uses, kin ignores
    ├── jaaba_vs_heuristic_summary.csv
    └── <acq>_jaaba_vs_heuristic.png (per-acq)
```

Change `SUBSET_TAG` to version parallel runs (e.g. `'testset_v2'`).

To discover available strains and acquisitions before selecting:

```python
import analyses.strain_variation.src.strain_funcs as sf
available = sf.list_acquisitions_by_strain('/Volumes/Juliana/Caitlin_RA_data')
sf.print_subset_summary(available)
```

## JAABA vs kinematic label comparison

The `compare_jaaba_to_heuristics` module compares two ways of labeling
chasing and singing on the same data:

| Method | Source |
|---|---|
| **JAABA** | `jaaba_chasing_binary`, `jaaba_unilateral_extension_binary` columns |
| **kinematic** | gates on `vel`, `facing_angle`, `dist_to_other` (and `max_wing_ang` for singing) |

Per-pair agreement is summarised by Cohen's kappa, Jaccard index, and F1
(JAABA as reference). The **threshold diagnostic** plots show where each
kinematic gate falls relative to the JAABA-positive distribution; the
**unconsidered features** plots show variables the transform computes but the
kinematic gate ignores — key candidates for explaining the chasing discrepancy:

| Variable | Why it matters for chasing |
|---|---|
| `rel_vel` | Closing speed (negative = approaching); not gated at all |
| `ang_vel_fly` | Active steering / turn rate during pursuit |
| `targ_ang_vel` | How fast the target drifts across the male's visual field |
| `target_vel` | Target (female) speed |
| `max_wing_ang` | Gate is `min=0` (effectively off) for chasing |
| `targ_pos_theta_deg` | Target bearing; gate bounds are ±270° (no constraint) |

## Orientation & head-tail flip handling (`--ori_method`)

`--ori_method` (default `velocity`) controls how FlyTracker head-tail
ambiguity is resolved. Affects all orientation-derived metrics
(`theta_error`, `targ_pos_theta`, `ang_vel_fly`); `feat`-derived
`vel`/`ang_vel`/`pos` are unaffected.

| `--ori_method` | Summary | Use when |
|---|---|---|
| **`velocity`** (default) | keep body-axis `ori`; NaN only chunks moving opposite their orientation | forward-locomotion assays (courtship/pursuit) |
| **`wing`** | NaN `ori` where wings undetected (legacy `filter_ori=True`) | substantial sideways/backward motion |
| **`none`** | raw FlyTracker `ori`, unchanged | when handling polarity downstream |

Full rationale in
[`libs/README.md`](../../libs/README.md#orientation-heading--head-tail-flips-qcresolve_orientation).
