#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check transformations on a single acquisition.

Loads one processed datafile, recomputes ang_vel_fly with head-tail flip
correction and FlyTracker-style smoothing, finds a chasing snippet, and
generates diagnostic plots:
  1. Time-course (target pos, theta error, ori, ang_vel comparison)
  2. Zoomed time-course (individual frames)
  3. 2D trajectory + relative target position
  4. Video frame overlay (if video is found)
"""
#%%
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rem

from analyses.preprocessing.src.add_ft_actions import add_ft_actions

#%%
# -------------------------------------------------------------------
# Style
# -------------------------------------------------------------------
plot_style = 'dark'
min_fontsize = 12
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style == 'dark' else 'k'

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
acqdir = '/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector'
procdir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'
proc_parentdir = os.path.split(procdir)[0]

figdir = os.path.join(proc_parentdir, 'check_params')
os.makedirs(figdir, exist_ok=True)
print(f"Saving figures to {figdir}")

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
fps = 60
snippet_dur_sec = 3
example_lag = 6
frame_step = 6
arrow_len = 20

# -------------------------------------------------------------------
# Pick a single file to check
# -------------------------------------------------------------------
# Default: first parquet found in procdir
example_file = None  # set manually or via CLI
parquet_files = sorted(glob.glob(os.path.join(procdir, '*_df.parquet')))
if example_file is None and len(parquet_files) > 0:
    example_file = os.path.splitext(os.path.basename(parquet_files[0]))[0].replace('_df', '')
    print(f"Auto-selected file: {example_file}")
assert example_file is not None, "No parquet files found in procdir"

#%%
# -------------------------------------------------------------------
# Load single acquisition
# -------------------------------------------------------------------
fpath = os.path.join(procdir, f'{example_file}_df.parquet')
assert os.path.exists(fpath), f"File not found: {fpath}"
df0 = pd.read_parquet(fpath)
print(f"Loaded {len(df0)} rows from {fpath}")

# Add actions if not already present
if 'chasing' not in df0.columns:
    df0 = add_ft_actions(procdir, acqdir, example_file, verbose=True)

# -------------------------------------------------------------------
# Prepare fly (id==0) and target (id==1) dataframes
# -------------------------------------------------------------------
f1 = df0[df0['id'] == 0].sort_values('frame').copy()
targ_df = df0[df0['id'] == 1].sort_values('frame').copy()
f1['file_name'] = example_file

# -------------------------------------------------------------------
# Recompute ang_vel_fly with head-tail flip correction
# -------------------------------------------------------------------
f1['ang_vel_fly'] = util.recompute_ang_vel(f1['ori'], fps=fps)

# NaN out head-tail flip frames (ori jumps > 90 deg) + 1 frame margin
ori_diff = f1['ori'].diff().abs()
flip_mask = ori_diff > np.pi / 2
flip_mask = flip_mask | flip_mask.shift(1, fill_value=False) | flip_mask.shift(-1, fill_value=False)
n_flips = flip_mask.sum()
pct_flips = n_flips / len(f1) * 100
print(f"Head-tail flips: {n_flips} frames ({pct_flips:.2f}%)")
f1.loc[flip_mask, ['ori', 'ang_vel_fly']] = np.nan

# -------------------------------------------------------------------
# Degree columns
# -------------------------------------------------------------------
f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])
f1['ang_vel_fly_deg'] = np.rad2deg(f1['ang_vel_fly'])
f1['ang_vel_deg'] = np.rad2deg(f1['ang_vel'])  # FT feat ang_vel is unsigned speed; no sign flip
f1['ori_deg'] = np.rad2deg(f1['ori'])
f1['sec'] = f1['frame'] / fps

# -------------------------------------------------------------------
# Save directory
# -------------------------------------------------------------------
save_dir = os.path.join(figdir, example_file)
os.makedirs(save_dir, exist_ok=True)

#%%
# -------------------------------------------------------------------
# Find a chasing snippet
# -------------------------------------------------------------------
assert 'chasing' in f1.columns, "'chasing' column missing -- run add_ft_actions first"
assert f1['chasing'].sum() > 0, f"No chasing frames found in {example_file}"

stretch, f_start, f_end = util.find_action_snippet(f1, action='chasing', fps=fps,
                                                    snippet_dur_sec=snippet_dur_sec)
stretch_targ = targ_df[(targ_df['frame'] >= f_start) & (targ_df['frame'] <= f_end)]
print(f"Snippet: frames {f_start}–{f_end} ({(f_end - f_start)/fps:.1f}s)")

#%%
# -------------------------------------------------------------------
# Figure 1: time-course
# -------------------------------------------------------------------
fig_tc, _ = putil.diagnostics_plot_timecourses(
    stretch, f_start, f_end, example_lag=example_lag,
    fps=fps, bg_color=bg_color,
    title=f'{example_file}  |  frames {f_start}–{f_end}')
fig_tc.savefig(os.path.join(save_dir, 'timecourse.png'),
               dpi=150, bbox_inches='tight')

# -------------------------------------------------------------------
# Figure 1b: zoomed time-course
# -------------------------------------------------------------------
fig_zoom, _ = putil.diagnostics_plot_timecourses_zoom(stretch, bg_color=bg_color)
fig_zoom.savefig(os.path.join(save_dir, 'timecourse_zoom.png'),
                 dpi=150, bbox_inches='tight')

# -------------------------------------------------------------------
# Figure 2: 2D trajectory + relative target position
# -------------------------------------------------------------------
fig2, _ = putil.diagnostics_plot_2d_traj_and_rel(
    stretch, stretch_targ, f_start, f_end,
    arrow_len=arrow_len,
    title=f'{example_file}  |  {(f_end - f_start)/fps:.1f}s snippet')
fig2.savefig(os.path.join(save_dir, 'trajectory_2d.png'),
             dpi=150, bbox_inches='tight')

#%%
# -------------------------------------------------------------------
# Figure 3: Video frame overlay
# -------------------------------------------------------------------
video_path = os.path.join(acqdir, example_file)
found_vid = util.find_video(video_path)

if found_vid is not None:
    print(f"Loading video: {found_vid}")
    frames_to_show = list(range(f_start, f_end, frame_step))
    n_overlay = len(frames_to_show)

    fig_vid, ax_vid = plt.subplots(1, 1, figsize=(8, 8))
    putil.plot_video_overlay(ax_vid, found_vid, frames_to_show,
                             flydf=stretch, targdf=stretch_targ,
                             f_start=f_start, f_end=f_end,
                             arrow_len=arrow_len)
    ax_vid.set_title(f'{example_file}\nframes {f_start}–{f_end}, '
                     f'every {frame_step}th ({n_overlay} frames)')
    plt.tight_layout()
    fig_vid.savefig(os.path.join(save_dir, 'video_overlay.png'),
                    dpi=150, bbox_inches='tight')
else:
    print(f"No video found at {video_path}")

print(f"\nAll figures saved to {save_dir}")

# %%
