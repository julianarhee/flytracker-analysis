#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import sys
import glob
import importlib

import numpy as np
import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt
import seaborn as sns

import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rem

from analyses.preprocessing.src.add_ft_actions import add_ft_actions
import gain_funcs as gf

#%%
plot_style='dark'
min_fontsize=12
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# %%
#species_colors = ['plum', 'mediumseagreen']
species_palette = {'Dmel': 'plum', 
                   'Dyak': 'mediumseagreen',
                   'Dele': 'aquamarine'}

# %%
from datetime import datetime
#today = datetime.now().strftime('%Y%m%d')
#print(today)
#%%
# Acquisition dir:
videodir = '/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector'
#has_meta_csv = True

# Processed data directory (after running transform_data.py)
procdir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'
#aggr_mat_fname ='transformed_projector_data.pkl' 

# local_fpath = None

# Basedir, for figures
proc_parentdir = os.path.split(procdir)[0]

# figdir 
figdir = os.path.join(proc_parentdir, 'plot_gain_projector')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("saving figures to {}".format(figdir))

# %%
# Get metadata
meta_fpath = os.path.join(proc_parentdir, 'courtship_free_behavior_data - 38mm_projector.csv')
assert os.path.exists(meta_fpath), "Meta file does not exist: {}".format(meta_fpath)
# Load meta data
meta0 = pd.read_csv(meta_fpath)
meta = meta0[(meta0['tracked in matlab and checked for swaps']==1)
           & (meta0['exclude']==0) 
           & (meta0['annotated']==1)
           & (meta0['manipulation_male']=='no paint')
           & (meta0['painted']=='none')
           & (meta0['genotype_male']=='P1a-CsChR')
           & (meta0['calibration']==0)].copy()
# date-fly
meta['date_fly'] = ['_'.join( [f.split('-')[0], \
                        f.split('_')[1]] ) for f in meta['file_name']]
print(meta.shape)
# %
n_species = meta.groupby('species_male', as_index=False)['date_fly'].nunique()
print(n_species)

#%%
create_new = False
create_aggregate = False

# Check for existing aggregate:
found_aggr_fpaths = sorted(glob.glob(os.path.join(proc_parentdir, f'2026*_transformed_yak_mel_controls_chasing.parquet')))
for fpath in sorted(found_aggr_fpaths):
    print(fpath)

#%%
if len(found_aggr_fpaths) > 0:
    try:
        print("Loading latest existing aggregate from {}".format(found_aggr_fpaths[-1]))
        df0 = pd.read_parquet(found_aggr_fpaths[-1])
        assert 'stim_direction' in df0.columns, "stim_direction column not found in dataframe, remaking"
    except Exception as e:
        print(f"Error loading aggregate from {found_aggr_fpaths[0]}: {e}")
        print("Remaking aggregate")
        create_aggregate = True
else:
    print("No existing aggregate found, transforming data")
    create_aggregate = True

if create_aggregate:
    today = datetime.now().strftime('%Y%m%d')
    proc_parentdir = os.path.split(procdir)[0]
    aggr_fpath = os.path.join(proc_parentdir, f'{today}_transformed_yak_mel_controls_chasing.parquet')

#%%
# Transform data 
all_acqs = meta['file_name'].unique()

if create_aggregate:
    #%
    # Add actions
    df_list = []; errs=[];
    for acq in all_acqs: #curracqs:
        try:
            df_ = add_ft_actions(procdir, videodir, acq, verbose=False)

            # Assign stimulus direction from meta
            currm = meta[meta['file_name']==acq]
            assert len(currm)>0, 'No meta data for {}'.format(acq)
            assert len(currm)==1, 'Multiple meta data for {}'.format(acq)
            stim_dir = currm['stim_direction'].unique()[0] #fn.split('_')[-1]
            df_['stim_direction'] = stim_dir

            # Add file info 
            df_['file_name'] = os.path.split(acq)[-1]
            if 'species' not in df_.columns: 
                df_['species'] = util.get_species_from_acquisition_name(acq)
            #if reassign_acquisition_name:
            df_['date_fly'] = ['_'.join([f.split('-')[0], f.split('_')[1]]) for f in df_['file_name']]
            df_['acquisition'] = ['_'.join([a, b]) for a, b in df_[['date_fly', 'species']].values]

            df_list.append(df_)
        except Exception as e:
            print(f"Error adding actions for {acq}: {e}")
            errs.append((acq, e))
            continue
        # df.to_parquet(os.path.join(procdir, f'{acq}_df.parquet'))
    df0 = pd.concat(df_list)

    #%
    # Save transformed data
    #today_str = datetime.now().strftime('%Y%m%d')
    #aggr_fpath = os.path.join(procdir, f'{today}_transformed_yak_mel_controls_chasing.parquet')
    df0.to_parquet(aggr_fpath, engine='pyarrow', compression='snappy')
    print("Transformed data saved to {}".format(aggr_fpath)) 

#%%
#if 'ang_vel_abs' not in df0.columns:
#    df0['ang_vel_abs'] = np.abs(df0['ang_vel'])
#    df0, errors = gf.transform_projector_data(acqdir, all_acqs,
#                            procdir, movie_fmt='.avi',
#                            subdir=None, flyid1=0, flyid2=1,
#                            create_new=create_new, 
#                            reassign_acquisition_name=True)

#%%
# errors - no actions (no courtship)
'20250514-1051_fly1_Dyak-p1_5do_gh_2dR_ccw'
'20250710-1505_fly1_Dyak-p1_3do_gh_2dR'
'20250910-1338_fly3_Dyak-p1_5do_gh_2dR' 
'20250910-1409_fly5_Dyak-p1_5do_gh_2dR'


# %%
# Assigm stimulus direction from meta
f1 = df0[(df0['id']==0)].copy()
#%
# Assign "pr_direction" progressive or regressive:
# stim_direction is CCW, and target_position < 0: progressive
# stim_direciton is CCW, and target_position > 0: regressive
# is CW and > 0: progressive
# is CW and < 0: regressive
f1['pr_direction'] = None
f1.loc[(f1['stim_direction']=='CCW') & (f1['targ_pos_theta']<0), 'pr_direction'] = 'progressive'
f1.loc[(f1['stim_direction']=='CCW') & (f1['targ_pos_theta']>0), 'pr_direction'] = 'regressive'
f1.loc[(f1['stim_direction']=='CW') & (f1['targ_pos_theta']>0), 'pr_direction'] = 'progressive'
f1.loc[(f1['stim_direction']=='CW') & (f1['targ_pos_theta']<0), 'pr_direction'] = 'regressive'
#%%
# Recompute ang_vel_fly from ori, fixing head-tail flips with discont=pi/2
fps = 60
# f1['ang_vel_fly'] = f1.groupby('file_name')['ori'].transform(util.recompute_ang_vel, fps=fps)

# NaN out head-tail flip frames (ori jumps >= 180°) + 1 frame margin on each side
ori_diff = f1.groupby('file_name')['ori'].diff().abs()
flip_mask = round(ori_diff) >= np.pi*0.9 #/ 2
flip_mask = flip_mask | flip_mask.shift(1, fill_value=False) | flip_mask.shift(-1, fill_value=False)
flips_per_file = f1.loc[flip_mask].groupby('file_name').size()
flips_per_file = flips_per_file.reindex(f1['file_name'].unique(), fill_value=0)
frames_per_file = f1.groupby('file_name').size()
pct_per_file = flips_per_file / frames_per_file * 100
print(f"Head-tail flips per file: {flips_per_file.mean():.1f} +/- {flips_per_file.std():.1f} "
      f"({pct_per_file.mean():.2f} +/- {pct_per_file.std():.2f}% of frames per file)")
# f1.loc[flip_mask, ['ori', 'ang_vel_fly']] = np.nan

# Shift variables by lag of 200ms delay
f1 = util.shift_variables_by_lag(f1, lag=2, file_grouper='file_name')

# Convert to deg and get abs
f1['ang_vel_fly_shifted_abs'] = np.abs(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_shifted_deg'] = np.rad2deg(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_deg'] = np.rad2deg(f1['ang_vel_fly'])
f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])

#%%
# Bin by object position
f1 = gf.bin_by_object_position(f1, start_bin=-180, end_bin=180, bin_size=20)
f1['binned_theta_error_num'] = pd.to_numeric(f1['binned_theta_error'], errors='coerce')

f1.reset_index(drop=True, inplace=True)

#%%
filter_manual = False

deg_lim = 180
vel_lim = 10
# filter chase
if filter_manual:
    chasedf = f1[f1['chasing']==1].copy()
else:
    chasedf = f1[(f1['theta_error_deg']<deg_lim)
            & (f1['theta_error_deg']>-deg_lim)
            & (f1['vel']>vel_lim)
            #& (f1['max_wing_ang']>np.deg2rad(30))
            ].copy()

# chase_str
chase_str = 'manual' if filter_manual else f'theta-{deg_lim}-vel-{vel_lim}'
#chasedf = gf.bin_by_object_position(chasedf, start_bin=-180, end_bin=180, bin_size=20)
chasedf.reset_index(drop=True, inplace=True)

#chasedf['binned_theta_error_num'] = chasedf['binned_theta_error'].astype(int)
chasedf['binned_theta_error_num'] = pd.to_numeric(chasedf['binned_theta_error'], errors='coerce')

#%%a
# PLOT GAIN
# ------------------------------------------------------------
hue_var = 'pr_direction'
pr_palette = {'progressive': 'darkgreen', 'regressive': 'purple'}
cw_palette = {'CW': 'cyan', 'CCW': 'magenta'}
stim_palette = pr_palette if hue_var == 'pr_direction' else cw_palette
hue_str = 'PRO-REG' if hue_var == 'pr_direction' else 'CW-CCW'
start_bin = -180
end_bin = 180
lw = 2

figsize = (8, 4)
n_species = chasedf['species'].nunique()

# Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error',
           'binned_theta_error_num']
#if assay == '38mm_projector':
grouper.append(hue_var) 

yvar = 'ang_vel_fly_shifted_deg'
#import scipy.stats as spstats
# Get average ang vel across individuals
mean_chase = chasedf.groupby(grouper)[yvar].mean().reset_index()
# Get circular mean of ang vel across groups

#avg_ang_vel_no_levels = chase_.copy() #groupby(grouper)[yvar].mean().reset_index()
err = 'se'
print(mean_chase.groupby('species')['acquisition'].nunique())

fig, axn = plt.subplots(1, n_species, figsize=figsize,
                        sharex=True, sharey=True)
for si, (sp, plotd) in enumerate(mean_chase.groupby('species')):
    #plotd = avg_ang_vel[avg_ang_vel['species']=='Dyak'].copy()
    if n_species==1:
        ax=axn
    else:
        ax=axn[si]

    sns.lineplot(data=plotd[plotd['binned_theta_error_num']<0], 
                x='binned_theta_error', y=yvar, ax=ax,
                hue=hue_var, palette=stim_palette, 
                errorbar=err, marker='o', 
                markersize=0, markeredgewidth=0,
                err_style='bars', legend=0, lw=lw, err_kws={'linewidth': lw})
    sns.lineplot(data=plotd[plotd['binned_theta_error_num']>0], 
                x='binned_theta_error', y=yvar, ax=ax,
                hue=hue_var, palette=stim_palette, 
                errorbar=err, marker='o', 
                markersize=0, markeredgewidth=0,
                err_style='bars', lw=lw, err_kws={'linewidth': lw},
                legend=si==1)
    # if legend exist, move it
    if si==1:
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                        frameon=False, title='movement dir', 
                        fontsize=min_fontsize-2)
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_xticks(np.linspace(start_bin, end_bin, 5))
    ax.set_title(sp)

ax.set_xlim([-180, 180])
ax.set_xticks(np.linspace(-180, 180, 9))    

#%%
# Plot individual flies?
individual_lw = 0.5
individual_alpha = 1
mean_lw = 3

fig, axn = plt.subplots(1, n_species, figsize=figsize,
                        sharex=True, sharey=True)
for si, (sp, plotd) in enumerate(chasedf.groupby('species')):
    #plotd = avg_ang_vel[avg_ang_vel['species']=='Dyak'].copy()
    if n_species==1:
        ax=axn
    else:
        ax=axn[si]

    for side in ['left', 'right']:
        if side == 'left':
            plotd_ = plotd[plotd['binned_theta_error_num']<0]
            mean_ = mean_chase[(mean_chase['species']==sp)
                                & (mean_chase['binned_theta_error_num']<0)].copy()
        else:
            plotd_ = plotd[plotd['binned_theta_error_num']>0]   
            mean_ = mean_chase[(mean_chase['species']==sp)
                                & (mean_chase['binned_theta_error_num']>0)].copy()
        sns.lineplot(data=plotd_, 
                x='binned_theta_error', y=yvar, ax=ax,
                hue=hue_var, palette=stim_palette, 
                style='acquisition', dashes=False, 
                lw=individual_lw, alpha=individual_alpha,
                errorbar=None,
                markersize=0, markeredgewidth=0,
                legend=0)

        # Plot means
        sns.lineplot(data=mean_, 
                    x='binned_theta_error', y=yvar, ax=ax,
                    hue=hue_var, palette=stim_palette, 
                    lw=mean_lw,
                    errorbar=err, #marker='o', 
                    markersize=0, markeredgewidth=0,
                    err_style='bars', legend=0, 
                    err_kws={'linewidth': mean_lw})

        #ax.set_ylim([-500, 500])
        ax.set_xticks(np.linspace(start_bin, end_bin, 5))
        ax.set_title(sp)
        ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
        ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)

    #ax.set_ylim([-500, 500])

#%%
# Compare gain plots across different lag values
lags_to_compare = [0, 3, 6, 9, 12, 24]  # frames

# Sort by frame within each file so shift(-lag) moves to correct future timepoint
f1 = f1.sort_values(['file_name', 'frame']).reset_index(drop=True)

# Precompute all shifted columns at once
ang_vel_grouped = f1.groupby('file_name')['ang_vel_fly']
for lag in lags_to_compare:
    col = f'ang_vel_shifted_deg_lag{lag}'
    f1[col] = np.rad2deg(ang_vel_grouped.shift(-lag))

# Build filtered df once (filter doesn't depend on lag)
chase_mask = ((f1['theta_error_deg'] < deg_lim)
              & (f1['theta_error_deg'] > -deg_lim)
              & (f1['vel'] > vel_lim))
chasedf_all = f1.loc[chase_mask].copy()
chasedf_all['binned_theta_error_num'] = pd.to_numeric(
    chasedf_all['binned_theta_error'], errors='coerce')

# Pre-group by species and acquisition for fast iteration
species_list = sorted(chasedf_all['species'].unique())
n_species_lag = len(species_list)

n_lags = len(lags_to_compare)
fig, axn = plt.subplots(n_species_lag, n_lags,
                        figsize=(n_lags * 4, n_species_lag * 4),
                        sharex=True, sharey=True, squeeze=False)

for li, lag in enumerate(lags_to_compare):
    yvar_lag = f'ang_vel_shifted_deg_lag{lag}'

    # Mean per acquisition per bin (for individual lines)
    grouper_lag = ['species', 'acquisition', 'binned_theta_error',
                   'binned_theta_error_num', hue_var]
    mean_acq = chasedf_all.groupby(grouper_lag)[yvar_lag].mean().reset_index()

    # Grand mean per bin across acquisitions (for thick mean line + SE)
    grouper_mean = ['species', 'binned_theta_error', 'binned_theta_error_num', hue_var]
    grand_mean = mean_acq.groupby(grouper_mean)[yvar_lag].agg(['mean', 'sem']).reset_index()

    for si, sp in enumerate(species_list):
        ax = axn[si, li]
        sp_mean = mean_acq[mean_acq['species'] == sp]

        for side_sign in [-1, 1]:
            plotd_ = sp_mean[sp_mean['binned_theta_error_num'] * side_sign > 0]
            sns.lineplot(data=plotd_,
                         x='binned_theta_error', y=yvar_lag, ax=ax,
                         hue=hue_var, palette=stim_palette,
                         lw=mean_lw, errorbar=err,
                         markersize=0, markeredgewidth=0,
                         err_style='bars', legend=0,
                         err_kws={'linewidth': mean_lw})

        ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
        ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
        ax.set_xticks(np.linspace(start_bin, end_bin, 5))
        ax.set_title(f'{sp} | lag={lag}')
        if li == 0:
            ax.set_ylabel(yvar)

fig.suptitle(f'Gain ({hue_str}): progressive vs. regressive by lag', y=1.01)
plt.tight_layout()

#%%
# ---- Diagnostic: time-course for a single file to verify lag shift ----
fps = 60  # adjust if different
example_lag = 6  # frames to compare against lag=0
snippet_dur_sec = 3  # seconds of trace to show

# Create examples save dir
example_figdir = os.path.join(figdir, 'examples')

chase_counts = (f1[f1['chasing'] == 1]
                .groupby('file_name')['frame']
                .count()
                .sort_values(ascending=False))
example_file = chase_counts.index[0]
print(f"Using file: {example_file}  ({chase_counts.iloc[0]} chasing frames)")

save_example_dir = os.path.join(example_figdir, example_file)
os.makedirs(save_example_dir, exist_ok=True)

edf = f1[f1['file_name'] == example_file].sort_values('frame').copy()
targ_df = df0[(df0['file_name'] == example_file) & (df0['id'] == 1)].sort_values('frame').copy()

stretch, f_start, f_end = util.find_action_snippet(edf, action='chasing', fps=fps, snippet_dur_sec=snippet_dur_sec)
stretch_targ = targ_df[(targ_df['frame'] >= f_start) & (targ_df['frame'] <= f_end)]
print(f"Showing frames {f_start}-{f_end} ({(f_end-f_start)/fps:.1f}s)")

# Precompute degree columns
stretch['ang_vel_deg'] = np.rad2deg(stretch['ang_vel'])  # FT feat ang_vel is unsigned speed; no sign flip
stretch['ori_deg'] = np.rad2deg(stretch['ori'])
stretch['ang_vel_fly_deg'] = np.rad2deg(stretch['ang_vel_fly'])

# ---- Figure 1: time-course ----
fig_tc, _ = putil.diagnostics_plot_timecourses(stretch, f_start, f_end, example_lag=example_lag,
                                               fps=fps, bg_color=bg_color,
                                               title=f'{example_file}  |  frames {f_start}–{f_end}')
fig_tc.savefig(os.path.join(save_example_dir, 'timecourse.png'), dpi=150, bbox_inches='tight')

# ---- Figure 1b: zoomed time-course ----
fig_zoom, _ = putil.diagnostics_plot_timecourses_zoom(stretch, bg_color=bg_color)
fig_zoom.savefig(os.path.join(save_example_dir, 'timecourse_zoom.png'), dpi=150, bbox_inches='tight')

# ---- Figure 2: 2D trajectory + relative target position ----
arrow_len = 20
fig2, _ = putil.diagnostics_plot_2d_traj_and_rel(stretch, stretch_targ, f_start, f_end,
                                                  arrow_len=arrow_len,
                                                  title=f'{example_file}  |  {(f_end-f_start)/fps:.1f}s snippet')
fig2.savefig(os.path.join(save_example_dir, 'trajectory_2d.png'), dpi=150, bbox_inches='tight')

#%%
# ---- Figure 3: Video frame overlay ----
video_path = os.path.join(videodir, example_file)
found_vid = util.find_video(video_path)

if found_vid is not None:
    print(f"Loading video: {found_vid}")
    frame_step = 6
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
    fig_vid.savefig(os.path.join(save_example_dir, 'video_overlay.png'), dpi=150, bbox_inches='tight')
else:
    print(f"No video found at {video_path}")

# %%
