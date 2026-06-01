# libs/

Shared utilities used across the analyses. Add code here only when more than
one analysis needs it; otherwise keep it in that analysis's `src/`. The package
is `pip install -e .`'d, so import via the package path (e.g.
`import libs.utils as util`) rather than relative imports.

## Modules

| Module | Alias | What it's for |
|---|---|---|
| `utils.py` | `util` | Core helpers: FlyTracker I/O (`load_flytracker_data`, `find_video`), frame/coordinate transforms, `find_action_snippet`, `recompute_ang_vel`, `shift_variables_by_lag`, species/acquisition parsing. |
| `plotting.py` | `putil` | Shared plotting: `set_sns_style`, diagnostic plotters (`diagnostics_plot_2d_traj_and_rel`, `diagnostics_plot_timecourses`), `plot_video_overlay`, color wheels / colorbars, ethograms. |
| `qc.py` | `qc` | Quality-control primitives: orientation / head-tail-flip handling (`resolve_orientation`, `resolve_flip_chunks`, `nan_flipped_orientation_per_id`, `detect_headtail_flips`), `plot_flip_montage`, `save_bout_video`. See decision points below. |
| `stats.py` | `lstats` | Stats helpers: OLS R², mixed ANOVA, multiple-comparison correction, p-value stars. |
| `regplot.py` | `rpl` | Regression/scatter plotting with polynomial fits (`regplot`, `scatter`, `fitplot`, `polyfit`). |
| `dlc.py` | `dlc` | DeepLabCut helpers: filtering, unit conversion, fly/dot/interfly parameter extraction. |
| `utils_2p.py` | `util2p` | 2-photon / ViRMEn loaders (MATLAB `.mat`, XML), tuning-index computation. |
| `basic_units.py` | — | Vendored matplotlib `basic_units` example (unit-aware plotting helper). |
| `SeabornFig2Grid.py` | — | Vendored StackOverflow helper to place seaborn figure-level plots into a `GridSpec`. |

Tests live in `libs/tests/` (run with `pytest`; figure-producing code uses
`matplotlib.use('Agg')`).

---

## Decision points

Key, non-obvious choices baked into the shared code — documented here so they're
discoverable from any analysis, not buried in one analysis folder.

### FlyTracker `ori` and the `-1*ori` negation (coordinate frame)

FlyTracker reports orientation (`ori`) in **image coordinates**: origin
top-left, y-axis pointing **down**, which makes angles **clockwise-positive**.
The relative-metrics transform (rotation to the focal fly, `cart2pol`,
`calculate_theta_error`) instead assumes the **standard math frame**: y-axis
**up**, angles **CCW-positive** (0 = East, increasing CCW). To reconcile them,
FlyTracker `ori` is negated **once** at the start of processing:

```python
trk['ori'] = -1 * trk['ori']   # FlyTracker image-frame (CW+) -> math-frame (CCW+)
```

(see `transform_data/relative_metrics.py` — `do_transformations_on_df` and the
per-fly transform comments: *"Assumes fly1 ORI goes from 0 to pi CCW, y-axis
NOT-inverted; if using FlyTracker, trk['ori'] = -1*trk['ori']"*). After this,
every orientation-derived metric (egocentric target position, `theta_error`,
`ang_vel_fly`) is in the math frame.

- **DLC output is already in the math/bottom-left frame, so it is NOT negated.**
  The negation is FlyTracker-specific.
- **Ordering gotcha.** Anything that reasons about raw FlyTracker `ori` against
  the *video image* must run on the RAW `ori`, **before** this negation —
  notably `qc.resolve_orientation` / `resolve_flip_chunks` (`heading =
  arctan2(-dy, dx)`) and `qc.plot_flip_montage` (on-image heading `= -ori`). The
  processing pipeline calls `resolve_orientation` *before* negating for exactly
  this reason.

### Angular velocity: `ang_vel` (unsigned) vs `ang_vel_fly` (signed)

Two angular-velocity columns are easy to confuse — they differ in **sign**:

| Column | Source | Sign | Use for |
|---|---|---|---|
| `ang_vel` | FlyTracker `feat.mat`, as-is | **unsigned** angular *speed* (\|dθ/dt\|, always ≥ 0) | magnitude only |
| `ang_vel_fly` | recomputed from (negated) `ori` via circular differencing (`util.smooth_and_calculate_velocity_circvar`; cf. `util.recompute_ang_vel`, which unwraps to correct flips) | **signed** (= dθ/dt; + = CCW in the math frame) | turn **direction** |

Verified on real data: `ang_vel` min 0.00 / 0% negative; `ang_vel_fly` ranges
±~80 with ~49% negative. **Do not compare the sign of `ang_vel` to
`ang_vel_fly`** (one has no sign) — compare magnitudes, or use `ang_vel_fly` for
anything directional.

- `libs.plotting.diagnostics_plot_timecourses` (and the zoom variant) plot
  `ang_vel` **as-is** (unsigned magnitude, no sign flip) against the signed
  `ang_vel_fly`. (They previously negated `ang_vel` by default — a historical
  convention that wrongly treated an unsigned speed as signed; removed.)
- **Display sign of `ang_vel_fly` / `theta_error`** (positive = fly's right vs.
  left) is an **analysis-specific display convention**, not a property of the
  data — the `gain/` and `p1_levels/` analyses document their chosen convention
  (positive = fly's right) in their script headers. Don't assume a global
  left/right sign.

### Orientation, heading & head-tail flips (`qc.resolve_orientation`)

FlyTracker reports a body-axis **orientation** (`ori`) per fly, but two things
make it unreliable in places:

1. **Head-tail flips** — FlyTracker occasionally swaps which end is the head, a
   ~180° frame-to-frame jump in `ori`. These split a track into contiguous
   *chunks* that alternate between correct and flipped orientation.
2. **Missing wings** — FlyTracker sometimes tracks the body fine but fails to
   detect the wings. The body axis is still valid, but wings help disambiguate
   head from tail, so orientation is *less certain* on those frames.

Everything orientation-derived depends on how we handle this —
`theta_error`, `targ_pos_theta`, egocentric target position, and the recomputed
signed `ang_vel_fly`. Velocity/position-derived columns (`vel`, FlyTracker
`ang_vel`, `pos_x/y`) do **not**.

`qc.resolve_orientation(trk, method=...)` is the single, dataset-agnostic entry
point. Call it on **raw** tracking (`load_flytracker_data(..., filter_ori=False)`,
`ori` not yet negated), **before** the FlyTracker `ori = -1*ori` convention flip
and the relative-metrics transform.

| `method` | What it does | Motion assumption | Data kept |
|---|---|---|---|
| **`velocity`** (default) | keep body-axis `ori`; NaN only chunks pointing *opposite* the direction of motion (a real flip), per fly | flies move **forward**, rarely sideways/backward | most frames |
| **`wing`** | NaN `ori` wherever wings were undetected (legacy `load_flytracker_data(filter_ori=True)`) | **none** | fewer — discards all wingless frames |
| **`none`** | trust FlyTracker's raw `ori` unchanged | **none** | all |

**Why `velocity` is the default.** The error we actually care about is head/tail
*polarity*. `wing` uses wing-presence as a *proxy* for "can we trust polarity,"
which is poorly correlated with the real error: it (a) discards large stretches
of valid courtship where the body axis was fine but wings weren't (e.g. one Dyak
bout lost 121/181 frames under `wing`, fully recovered under `velocity`), and
(b) is **behaviour-correlated** — wing detection tracks wing extension / song, so
its data loss biases orientation-based metrics. `velocity` tests the thing
directly: a walking/chasing fly moves forward, so on moving frames the body axis
should align with heading; chunks that are anti-aligned are genuine flips and
are excluded. Compute cost is negligible (vectorized; dwarfed by I/O and the
transform).

**How `velocity` decides (and what it does *not* assume).** Per fly, it splits
the track at detected ~180° jumps and classifies **each chunk independently** by
the mean of `cos(ori − heading)` over its moving frames (`heading = arctan2(-dy,
dx)`; image y is down). A chunk is flipped iff that mean is negative. It does
**not** assume each detected jump is a real persistent toggle and alternate
parity from an anchor — spurious flip-and-back jumps are common and that
assumption mislabels correctly-oriented chunks (observed: a fly with two chunks
both aligned at +0.65 / +0.85 had its long correct chunk wrongly flagged by the
old global-parity rule). Chunks with too little motion to decide are **kept**
(never NaN'd without positive backward evidence); if *no* chunk is decidable
(a fly that never moves enough), it falls back to anchoring the first chunk as
correct and alternating at each flip.

**When to switch.** `velocity` assumes forward motion. For assays with
substantial **sideways or backward** motion, prefer `wing` or `none`. Note
`velocity` also cannot catch a flip on a fly that never moves (those chunks are
conservatively kept — but they're downstream-filtered by velocity thresholds
anyway).

**Deferred middle-ground option (on purpose).** A natural refinement is a
`velocity`-style method that *ignores near-perpendicular (sideways) frames* when
scoring each chunk, so occasional sideways steps don't influence the call while
real flips are still caught. It is intentionally **not** implemented: the
per-chunk decision already uses the *mean* alignment, and sideways frames
contribute ≈ 0 (cos ≈ 0), so they only matter when a chunk is *predominantly*
sideways. We'll add the knob if/when that case actually shows up in data rather
than adding speculative complexity now.

**Adoption.** `analyses/strain_variation/src/process_multichamber.py` is a thin
caller of `resolve_orientation` (exposed as `--ori_method`). Most other
FlyTracker analyses still inline the old `filter_ori=True` + `ori = -1*ori`
pattern; they can adopt `resolve_orientation` to get the same selectable
behavior.

### `theta_error`: from orientation (default) vs from heading

`theta_error` is the angular position of the target relative to the focal fly —
the circular distance between the **line of sight** to the target
(`abs_ang_between = arctan2(Δy, Δx)`) and the fly's facing direction. The
transform computes **both** versions (`do_transformations_on_df`):

| Column | Facing direction used | Function |
|---|---|---|
| **`theta_error`** (default) | **orientation** — the body axis (`ori`) | `calculate_theta_error` |
| `theta_error_heading` | **heading** — direction of motion (`arctan2(Δpos_y, Δpos_x)`) | `calculate_theta_error_from_heading` |

The default, and the one all pursuit / steering-gain analyses use, is the
**orientation-based** `theta_error`.

Use **orientation**. A fly's visual
system is body-/head-fixed, so the target's position *on the retina* — the error
signal that drives the steering (turning) response in visual pursuit — is set by
**body orientation**, not by the direction of travel. Heading and orientation
diverge whenever the fly sideslips, and heading is not what the eye senses
directly, so orientation is the appropriate regressor for steering gain and
theta-error analyses. In practice the two are highly correlated when sideslip is
small (the `p1_levels` gain QC plots their correlation); reserve the
heading-based version for questions specifically about control of travel
direction. Visualize the two against each other for any dataset with
`qc.plot_theta_error_ori_vs_heading(df)`.

> `theta_error_heading` is **QC-only** — not maintained or used in any
> downstream analysis (one exploratory pairplot aside). The value stored in
> older parquets predates transform fixes, and `heading` (smoothed position
> differencing) hasn't been validated for analysis use. **If you want to use the
> column, recompute it** rather than trusting the stored value — and consider
> revisiting how `heading` is derived first:
> ```python
> df['theta_error_heading'] = qc.recompute_theta_error_heading(df)
> # == util.circular_distance(df['abs_ang_between'], df['heading'])
> ```

### Egocentric target position & the left/right display convention

The transform stores the target's position **in the focal fly's frame** (fly at
origin, facing along its body axis):

- `targ_rel_pos_x` — along the body axis; **positive = in front** of the fly.
- `targ_rel_pos_y` — lateral; **positive = the fly's LEFT** (raw CCW / math
  frame, consistent with the `-1*ori` negation above).
- `targ_pos_radius`, `targ_pos_theta` — the polar form of the same.

To plot this the way you'd *look at it* — fly facing **up**, the fly's right on
the right of the page — analyses (e.g. `gain/`, `p1_levels`, `qc_checks`) map:

```python
plot_x = -targ_rel_pos_y   # negate so the fly's RIGHT is +x (right of page)
plot_y =  targ_rel_pos_x   # in front of fly -> top of page
```

This is the same "positive = fly's right" display convention used for
`theta_error` / `ang_vel_fly` in those analyses (see the ang_vel note above):
it's a display choice applied at plot time, not a property of the stored data.

### QC outputs (`qc.plot_flip_montage`, `qc.save_bout_video`)

- Flip QC is run on **raw** tracking (`filter_ori=False`) and **truncated at
  copulation** (post-copulation flies are stationary/mounted, with unreliable
  orientation that would inflate flip rates).
- Flip-example montages are selected from **courtship bouts** (see
  `find_flip_window(..., courtship_col=...)`), and we keep the montage rather
  than per-frame PNGs.
