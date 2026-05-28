#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
from re import I
import sys
import glob
import importlib
import subprocess
import cv2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import libs.utils as util
import libs.plotting as putil
import libs.stats as sutil

from analyses.gain.src import gain_funcs as gf
import transform_data.relative_metrics as rel

#%%
def infer_led_blocks(led_onset, n_LED_blocks, led_block_nframes, curr_leds):
    '''
    Infer LED block starts.
    Level 0 starts at frame 0, and level 1 starts at led_onset.

    Arguments:
        led_onset -- frame number of LED onset
        n_LED_blocks -- number of LED blocks
        led_block_nframes -- number of frames per LED block

    Returns:
        led_blocks -- DataFrame with LED block starts and ends
    '''
    led0_start = int(led_onset) - led_block_nframes
    if led0_start < 0:
        led0_start = 0
    led_starts = [led0_start]
    for i in range(1, n_LED_blocks):
        led_starts.append(int(led_onset) + (i - 1) * led_block_nframes)
    led_starts = np.array(led_starts, dtype=int)

    led_ends = led_starts + led_block_nframes - 1
    # Replace first led_end with actual led_stat
    led_ends[0] = led_onset - 1
    led_ends[-1] = total_nframes #- 1

    led_blocks = pd.DataFrame({
        'led_level': np.arange(n_LED_blocks, dtype=int),
        'led_start': led_starts,
        'led_end': led_ends,
    })

    # add intensity of each led block
    led_blocks.loc[led_blocks['led_level']==0, 'led_intensity'] = 0
    for l in range(1, n_LED_blocks):
        led_blocks.loc[led_blocks['led_level']==l, 'led_intensity'] = curr_leds[l-1]

    return led_blocks


def overlap_len(start_a, end_a, start_b, end_b):
    """
    Inclusive overlap length between intervals [start_a, end_a] and [start_b, end_b].
    """
    lo = max(start_a, start_b)
    hi = min(end_a, end_b)
    return max(0, hi - lo + 1)


def build_speed_blocks(
    led_blocks,
    speed_onset_df,
    n_speed_blocks,
    speed_block_nframes,
    total_nframes,
    curr_speeds,
    mode='annotated',
):
    """
    Build speed blocks within each LED block.

    Parameters
    ----------
    curr_speeds : list
        Speed values (e.g. Hz) corresponding to speed levels 0..n_speed_blocks-1.
    mode : str
        'annotated' -> use all speed_block_onset start frames directly;
                       ends are next level start - 1.
        'inferred'  -> infer all speed blocks from LED block start using
                       fixed speed_block_nframes spacing.
    """
    assert mode in ['annotated', 'inferred'], "mode must be 'annotated' or 'inferred'"

    speed_rows = []
    #speed_onsets = speed_onset_df[speed_onset_df['action'] == 'speed_block_onset'].copy()
    speed_onsets = speed_onset_df.sort_values('start').reset_index(drop=True)

    if max(led_blocks['led_intensity']) < 50: # Low LED intensity for Dmel
        led_type = 'low_led'
    else:
        led_type = 'full_led'

    if max(curr_speeds) < 80: # Low speed for Dyak
        speed_type = 'slow_speed'
    else:
        speed_type = 'standard_speed'

    last_ix = 0
    for li, led_row in led_blocks.iterrows():
        led_level = int(led_row['led_level'])
        led_start = int(led_row['led_start'])
        led_end = int(led_row['led_end'])
        led_intensity = int(led_row['led_intensity'])

        # Just use manual, in case of jitter:
        assert len(speed_onsets) == 30, "Expected 30 speed_block_onset events, found {}".format(len(speed_onsets))
        if li == 0:
            indices = np.arange(last_ix, last_ix+n_speed_blocks-1)
            assert indices[-1] != speed_onsets.index.tolist()[-1], "Using the wrong final index"
            curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
            starts = [0]
            starts.extend(curr_onsets)
            starts = np.array(starts, dtype=int)
            ends = np.concatenate([starts[1:] - 1, [led_end]])
        else:
            indices = np.arange(last_ix, last_ix+n_speed_blocks)
            assert indices[-1] != speed_onsets.index.tolist()[-1], "Using the wrong final index"
            curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
            starts = curr_onsets
            ends = np.concatenate([starts[1:] - 1, [led_end]])
        assert len(curr_onsets) == len(indices), "Expected {} speed_block_onset events, found {}".format(len(indices), len(curr_onsets))
        # Update last index  
        last_ix = indices[-1] + 1

#         if mode == 'annotated':
#             curr_onsets = speed_onsets[
#                 (speed_onsets['start'].astype(int) >= led_start-1)
#                 & (speed_onsets['start'].astype(int) <= led_end)
#             ]['start'].astype(int).sort_values().to_numpy()
#             # If first block, 1 less because annotations start at first movement onset 
#             if li == 0:
#                 curr_n_speed_blocks = n_speed_blocks - 1
#             else:
#                 curr_n_speed_blocks = n_speed_blocks
#             assert len(curr_onsets) >= curr_n_speed_blocks, (
#                 f"LED block {led_level}, ({led_start}, {led_end}): expected at least {curr_n_speed_blocks} speed_block_onset "
#                 f"events, found {len(curr_onsets)}: {curr_onsets}"
#             )
#             # Use exactly n_speed_blocks onsets; extras at block edges are ignored.
#             if li==0:
#                 starts = [0]
#                 starts.extend(curr_onsets)
#             else:
#                 starts = curr_onsets #[:n_speed_blocks]
#             starts = np.array(starts, dtype=int)
#             # Ends: each block ends one frame before the next starts; last ends at led_end.
#             ends = np.concatenate([starts[1:] - 1, [led_end]])
#         else:
#             starts = np.array(
#                 [led_start + s * speed_block_nframes for s in range(n_speed_blocks)],
#                 dtype=int,
#             )
#             ends = starts + speed_block_nframes - 1
#             ends[-1] = min(led_end, int(total_nframes) - 1)
# 
        for s_level in range(n_speed_blocks):
            speed_rows.append({
                'led_level': led_level,
                'led_type': led_type,
                'speed_level': s_level,
                'speed_hz': curr_speeds[s_level],
                'speed_type': speed_type,
                'start': int(starts[s_level]),
                'end': int(ends[s_level]),
                'led_intensity': led_intensity,
                'speed_block_source': mode,
            })

    return pd.DataFrame(speed_rows)


def build_speed_blocks_cw_ccw(
    speed_onset_df,
    n_speed_blocks,
    speed_block_nframes,
    total_nframes,
    curr_speeds,
    led_intensity=0,
    led_onset_frame=1200, # 20s in 
    mode='annotated',
    led_level=1, #'',
    led_type = 'standard',
    speed_type='standard',
):
    """
    Build speed blocks within each LED block.

    Parameters
    ----------
    curr_speeds : list
        Speed values (e.g. Hz) corresponding to speed levels 0..n_speed_blocks-1.
    mode : str
        'annotated' -> use all speed_block_onset start frames directly;
                       ends are next level start - 1.
        'inferred'  -> infer all speed blocks from LED block start using
                       fixed speed_block_nframes spacing.
    """
    assert mode in ['annotated', 'inferred'], "mode must be 'annotated' or 'inferred'"

    speed_rows = []
    #speed_onsets = speed_onset_df[speed_onset_df['action'] == 'speed_block_onset'].copy()
    speed_onsets = speed_onset_df.sort_values('start').reset_index(drop=True)
# 
# 
#     last_ix = 0
#     for li, led_row in led_blocks.iterrows():
#         led_level = int(led_row['led_level'])
#         led_start = int(led_row['led_start'])
#         led_end = int(led_row['led_end'])
#         led_intensity = int(led_row['led_intensity'])
# 
#         # Just use manual, in case of jitter:
#         assert len(speed_onsets) == 30, "Expected 30 speed_block_onset events, found {}".format(len(speed_onsets))
#         if li == 0:
#             indices = np.arange(last_ix, last_ix+n_speed_blocks-1)
#             assert indices[-1] != speed_onsets.index.tolist()[-1], "Using the wrong final index"
#             curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
#             starts = [0]
#             starts.extend(curr_onsets)
#             starts = np.array(starts, dtype=int)
#             ends = np.concatenate([starts[1:] - 1, [led_end]])
#         else:
#             indices = np.arange(last_ix, last_ix+n_speed_blocks)
#             assert indices[-1] != speed_onsets.index.tolist()[-1], "Using the wrong final index"
#             curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
#             starts = curr_onsets
#             ends = np.concatenate([starts[1:] - 1, [led_end]])
 
    #for s_level in range(n_speed_blocks):
    # first add stationary baseline:
    speed_rows.append({
        'led_level': led_level,
        'led_type': led_type,
        'speed_level': 0,
        'speed_hz': curr_speeds[0],
        'speed_type': speed_type,
        'start': 0,
        'end': led_onset_frame-1,
        'led_intensity': led_intensity,
        'speed_block_source': mode,
        'led_onset_frame': led_onset_frame
    })
    for si, speed_onset in speed_onsets.iterrows():
        start = speed_onset['start']
        end = start + speed_block_nframes - 1
        s_level = si+1
        if si == len(speed_onsets)-1:
            break
        speed_rows.append({
            'led_level': led_level,
            'led_type': led_type,
            'speed_level': s_level,
            'speed_hz': curr_speeds[s_level],
            'speed_type': speed_type,
            'start': int(start),
            'end': int(end),
            'led_intensity': led_intensity,
            'speed_block_source': mode,
            'led_onset_frame': led_onset_frame
        })

    return pd.DataFrame(speed_rows)



def count_courtship_frames(speed_blocks, actions_df, total_nframes):
    """
    Count courtship frames per LED/speed block using a frame-wise mask.
    Each courtship bout can span multiple blocks; frames are counted once per block.

    Parameters
    ----------
    speed_blocks : pd.DataFrame
        Output of build_speed_blocks.
    actions_df : pd.DataFrame
        Actions dataframe containing 'courtship' rows with 'start' and 'end'.
    total_nframes : int
        Total number of frames in the video.

    Returns
    -------
    pd.DataFrame
        Columns: led_level, speed_level, speed_hz, courtship_frames, courtship_frac.
    """
    courtship = actions_df[actions_df['action'] == 'courtship'].copy()
    courtship['start'] = courtship['start'].astype(int)
    courtship['end'] = courtship['end'].astype(int)

    courtship_mask = np.zeros(int(total_nframes), dtype=bool)
    for _, bout in courtship.iterrows():
        bstart = max(0, int(bout['start']))
        bend = min(int(total_nframes) - 1, int(bout['end']))
        if bend >= bstart:
            courtship_mask[bstart:bend + 1] = True

    counts = []
    for _, blk in speed_blocks.iterrows():
        bstart = int(blk['start'])
        bend = int(blk['end'])
        cframes = int(courtship_mask[bstart:bend + 1].sum())
        block_len = bend - bstart + 1
        led_onset_frame = int(blk['led_onset_frame'])
        # If block starts before LED onset, count courtship frames before LED onset separately
        # Add both blocks to counts
        cframes_before_led = 0
        block_len_before_led = 0
        if bstart < led_onset_frame:
            # There are 2 blocks
            blocks = [(bstart, led_onset_frame-1), (led_onset_frame, bend)]
            #print(blk['speed_hz'], blocks)
        else:
            blocks = [(bstart, bend)]
        for block in blocks:
            cframes = int(courtship_mask[block[0]:block[1] + 1].sum())
            block_len = block[1] - block[0] + 1
            is_led_off = block[0] <= led_onset_frame             
            counts.append({
                'led_level': int(blk['led_level']),
                'led_intensity': int(blk['led_intensity']),
                'led_type': blk['led_type'],
                'speed_type': blk['speed_type'],
                'speed_level': int(blk['speed_level']),
                'speed_hz': blk['speed_hz'],
                'courtship_frames': cframes,
                'courtship_frac': cframes / block_len if block_len > 0 else np.nan,
                'led_is_on': not(is_led_off)
            })

    return pd.DataFrame(counts)


#%%
plot_style='dark'
min_fontsize=12
putil.set_sns_style(plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'w'

#%%
# Set datapaths
rootdir = '/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector'

videodir = '/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector'
processedmat_dir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_calibrated/FlyTracker/processed_mats'
basedir = os.path.split(processedmat_dir)[0]

# Set figure dir
figdir = os.path.join(basedir, 'calibrated')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

# Load metadata
meta_fpath = glob.glob(os.path.join(rootdir, '*.csv'))[0]
meta0 = pd.read_csv(meta_fpath)
meta0.head()

#%
protocol = '40s_10_120_prj5ms'
# Get calibration data only
meta = meta0[
      (meta0['tracked in matlab and checked for swaps']==1)
    #& (meta0['calibration']==1)
    # Check if '40s_10_120_prj5ms' in traj_in column values
    & meta0['traj_in'].str.contains(protocol)
    & (meta0['speed_blocks_marked']==1)
    #& ~(meta0['led_onset_frame'].isna())
    ].copy()
meta.shape

figid = f'projector_{protocol}'
print(figid)

#%%
conds = ['species_male', 'age_male', 'days_on_retinol', 'speeds', 'intensity_light']
meta_counts = meta.groupby(conds)['file_name'].nunique()
meta_counts = meta_counts.reset_index()
#meta_counts.columns = conds + ['file_count']
print(meta_counts)

#%%
n_LED_blocks = 2
n_speed_blocks = 8

block_dur_sec = 40
dur_min =  n_speed_blocks * block_dur_sec #*n_LED_blocks * n_speed_blocks * block_dur_sec
print(dur_min)
fps = 60

total_nframes = fps*dur_min
print(total_nframes)

# block sizes (frames)
speed_block_nframes = int(round(block_dur_sec * fps))
led_block_nframes = int(round(n_speed_blocks * speed_block_nframes))
    #%

#%%
fn = meta['file_name'].unique()[0]
led_onset = 1200

# Get actions file
found_actions_paths = glob.glob(os.path.join(rootdir, fn, '*', '*-actions.mat'))
assert len(found_actions_paths)==1, 'Expected 1 actions file, found {}'.format(len(found_actions_paths))
# Load actions file
actions_df = util.load_ft_actions(found_actions_paths, split_end=False)
speed_onset_df = actions_df[actions_df['action']=='IR LED on'].copy()
speed_onset_df.loc[speed_onset_df['action']=='IR LED on', 'action'] = 'speed_block_onset'
speed_onset_df = speed_onset_df.sort_values(by='start')

led_intensity = meta[meta['file_name']==fn]['intensity_light'].values[0]
print(led_intensity)
curr_speeds = [0, 10, 20, 40, 60, 80, 100, 120]
speed_block_mode = 'annotated'

# %%
max_std_frames = 8 #10
c_list = []
errors = []
missing_files = []
for fn in meta['file_name'].unique():
    #%
    if not os.path.exists(os.path.join(rootdir, fn)):
        print('ERR: file not found: {}'.format(fn))
        missing_files.append(fn)
        #continue
    print('Found file: {}'.format(fn))

    led_intensity = meta[meta['file_name']==fn]['intensity_light'].values[0]
    stim_dir = meta[meta['file_name']==fn]['stim_direction'].values[0]
    print(led_intensity, stim_dir)

    # Get actions file
    found_actions_paths = glob.glob(os.path.join(rootdir, fn, '*', '*-actions.mat'))
    assert len(found_actions_paths)==1, 'Expected 1 actions file, found {}'.format(len(found_actions_paths))
    # Load actions file
    actions_df = util.load_ft_actions(found_actions_paths, split_end=False)
    speed_onset_df = actions_df[actions_df['action']=='IR LED on'].copy()
    speed_onset_df.loc[speed_onset_df['action']=='IR LED on', 'action'] = 'speed_block_onset'
    speed_onset_df = speed_onset_df.sort_values(by='start')

    try:
        assert np.diff(speed_onset_df['start']).std() < max_std_frames, "Speed onset jitter is too high"
    except Exception as e:
        errors.append((fn, e))
    #%
    # --------------------------------------------------------------------------
    # 1) Infer LED block starts.
    #    Level 0 starts at frame 0, and level 1 starts at led_onset.
    # --------------------------------------------------------------------------
    # Wrap this in a simple function
#    curr_leds = [int(l) for l in meta[meta['file_name']==fn]['intensity_light'].values[0].split(',')]
#    led_blocks = infer_led_blocks(led_onset, 
#                    n_LED_blocks, led_block_nframes,
#                    curr_leds)
    # print(led_blocks)
    #%
    # --------------------------------------------------------------------------
    # 2) Build speed blocks.
    #    mode = 'annotated': use speed_block_onset timing from actions_
    #    mode = 'inferred' : use LED block timing only
    # --------------------------------------------------------------------------
    #curr_speeds = list(range(n_speed_blocks))  # replace with actual speed values if known
    #curr_speeds = [int(s) for s in meta[meta['file_name']==fn]['speeds'].values[0].split(',')]
    curr_speeds = [0, 10, 20, 40, 60, 80, 100, 120]
    speed_block_mode = 'annotated'
    try:
        speed_blocks = build_speed_blocks_cw_ccw(
            #led_blocks=led_blocks,
            speed_onset_df=speed_onset_df,
            n_speed_blocks=n_speed_blocks,
            speed_block_nframes=speed_block_nframes,
            total_nframes=total_nframes,
            curr_speeds=curr_speeds,
            mode=speed_block_mode,
            led_intensity=led_intensity,
            led_onset_frame=led_onset,
            #led_level=led_level,
            #speed_type=speed_type,
        )
    #print(speed_blocks.head())
    except Exception as e:
        print(f"Error building speed blocks for {fn}: {e}")
        errors.append((fn, e))
        continue
    #%
    # --------------------------------------------------------------------------
    # 3) Count courtship frames by overlap with each LED/speed block.
    # --------------------------------------------------------------------------
    courtship_counts = count_courtship_frames(speed_blocks, actions_df, total_nframes)
    #print(courtship_counts)
    courtship_counts['file_name'] = fn
    courtship_counts['species'] = 'Dyak' if 'yak' in fn else 'Dmel'
    age = meta[meta['file_name']==fn]['age_male'].values[0]
    ATR = meta[meta['file_name']==fn]['days_on_retinol'].values[0]
    courtship_counts['age'] = age
    courtship_counts['ATR'] = ATR
    courtship_counts['age-ATR'] = '-'.join([str(age), str(ATR)])
    courtship_counts['led_intensity'] = led_intensity
    courtship_counts['stim_dir'] = stim_dir
   
    c_list.append(courtship_counts)

courtship_counts_all = pd.concat(c_list)

#%%
print("Errors:")
for e in errors:
    print(e)

print("Missing files:")
for m in missing_files:
    print(m)

courtship_counts_all.reset_index(drop=True, inplace=True)
#%%
# Add some meta columns
courtship_counts_all['date'] = [int(a.split('-')[0]) for a in courtship_counts_all['file_name']]
# Find 'fly##' in file_name:
courtship_counts_all['fly_num'] = [int(a.split('fly')[1].split('_')[0]) for a in courtship_counts_all['file_name']]
courtship_counts_all['fly_id'] = [f'fly{fnum}' for fnum in courtship_counts_all['fly_num']]

courtship_counts_all['acquisition'] = ['_'.join([str(a), b, c]) for a, b, c in courtship_counts_all[['date', 'fly_id', 'species']].values]

#courtship_counts_all[['file_name', 'acquisition']]
courtship_counts_all.groupby('species')['acquisition'].nunique()

#%%
# Save
aggr_fpath = os.path.join(basedir, 'cw_ccw_calibrated_courtship_counts.parquet')
courtship_counts_all.to_parquet(aggr_fpath)

print(f"Saved aggregated courtship counts to: {aggr_fpath}")

# %%
# Get number of files for each age-ATR and include in legend
conds = ['species', 'age-ATR', 'led_type', 'speed_type']
age_counts = courtship_counts_all.groupby(conds)['file_name'].nunique()
age_counts = age_counts.reset_index()
age_counts.columns = conds + ['file_count']
print(age_counts)


# %%
# Overall courtship vs. LED speed (ignore conditions)
species_palette = {'Dmel': 'plum', 
                   'Dyak': 'mediumseagreen'} 
#for sp, df_ in courtship_counts_all.groupby('species'):
# Put counts of each species in the legenda
n_species = courtship_counts_all['species'].nunique()
species_counts = courtship_counts_all.groupby(['species', 'stim_dir'])['file_name'].nunique()
species_counts = species_counts.reset_index()
species_counts.columns = ['species', 'stim_dir', 'file_count']
print(species_counts)
plotd = courtship_counts_all.copy()
fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
for i, (    cdir, cdf_) in enumerate(plotd.groupby('stim_dir')):
    ax=axn[i]
    sns.lineplot(data=cdf_, ax=ax,
                x='speed_hz', y='courtship_frac', 
                hue='species', legend=0, 
                palette=species_palette)
    ax.set_title(f'Courtship by speed, stim_dir: {cdir}')
    ax.set_xlabel('Speed (Hz)')
    ax.set_ylabel('Courtship frames')
    # Custom legend 
    legend_handles = [mpl.lines.Line2D([0], [0], color=species_palette[sp], lw=4) for sp in sorted(species_counts['species'])]
    legend_labels = [f"{sp} ({species_counts[species_counts['species']==sp]['file_count'].values[0]})" for sp in sorted(species_counts['species'])]
    plt.legend(legend_handles, legend_labels, loc='lower left', bbox_to_anchor=(1, 1), 
        frameon=False, title='', fontsize=10)  
#plt.show()

putil.label_figure(fig, figid)
figname = 'frac_courtship_by_speed'
#plt.savefig(os.path.join(figdir, f'{figname}.png'))
#plt.savefig(os.path.join(figdir, f'{figname}.svg'))

#%%
# Check 0 speed:
plotd = courtship_counts_all[courtship_counts_all['speed_hz']==0].copy()
fig, ax = plt.subplots(figsize=(10, 5))
sns.stripplot(data=plotd, ax=ax,
    x='acquisition', y='courtship_frac', 
    hue='led_is_on',# palette=species_palette, dodge=False,
    legend=1)
# Rotate x-ticks to be vertical
ax.tick_params(axis='x', labelrotation=90)
ax.set_title('0 speed')
ax.set_ylabel('Fraction courtship')
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), 
    frameon=False, title='')  

putil.label_figure(fig, figid)
figname = 'frac_courtship_speed0'
plt.savefig(os.path.join(figdir, f'{figname}.png'))
#plt.savefig(os.path.join(figdir, f'{figname}.svg'))
#%
plotd[plotd['courtship_frac']>0.5]

#%%
# Look at distribution of fraction courtship for each acquisition
plotd = courtship_counts_all.copy()
fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                        sharex=False, sharey=True)
for i, (sp, df_) in enumerate(plotd.groupby('species')):
    ax=axn[i]
    sns.stripplot(data=df_, ax=ax,
        x='acquisition', y='courtship_frac', 
        hue='speed_hz', palette='magma', dodge=False,
        legend=i==1)
    ax.set_title(f'{sp}')
    # Rotate x-ticks to be vertical
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Fraction courtship')
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), 
    frameon=False, title='speed (mm/s)')  
ax.set_ylabel('Fraction courtship')
#plt.show()
figname = 'per_fly_courtship_split_speed'
putil.label_figure(fig, figid)
plt.savefig(os.path.join(figdir, f'{figname}.png'))

# Box plot distribution
plotd = courtship_counts_all.copy()
fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                        sharex=False, sharey=True)
for i, (sp, df_) in enumerate(plotd.groupby('species')):
    ax=axn[i]
    sns.boxplot(data=df_, ax=ax,
        x='acquisition', y='courtship_frac')
        #hue='speed_hz', palette='magma', dodge=False,
        #legend=i==1)
    ax.set_title(f'{sp}')
    # Rotate x-ticks to be vertical
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Fraction courtship')

putil.label_figure(fig, figid)
figname = 'per_fly_courtship_boxplot'
plt.savefig(os.path.join(figdir, f'{figname}.png'))

#%%
acquisition_parentdir = rootdir
acqs = meta['file_name'].unique()
create_new = False

if not create_new:
    df_list = []
    for i, acq in enumerate(acqs):
        if i%10==0:
            print(f'Loading {i} of {len(acqs)}: {acq}')
        df_ = rel.load_processed_df(processedmat_dir, acq, create_new=create_new)
        assert 'pr_direction' in df_.columns, 'pr_direction not in df_.columns'
        assert 'stim_direction' in df_.columns, 'stim_direction not in df_.columns'
        df_list.append(df_)
    df0_all = pd.concat(df_list)
    print("Loaded all processed data")

#%%
if create_new:
    # Transform data 
    df0_all, errors = gf.transform_projector_data(acquisition_parentdir, acqs,
                                        processedmat_dir, movie_fmt='.avi',
                                        subdir=None, flyid1=0, flyid2=1,
                                        create_new=create_new, 
                                        reassign_acquisition_name=True)
    #% Reset index for easier indexing
    df0_all.reset_index(drop=True, inplace=True)

    # Assign stimulus direction from meta
    for fn, df_ in df0_all.groupby('file_name'):
        df_ = gf.assign_stim_directions(df_, meta)

        # Save df_ to processedmat_dir
        df_.to_parquet(os.path.join(processedmat_dir, f'{fn}_df.parquet'))

        # Overwrite original dataframe
        df0_all.loc[df_.index] = df_


#%%
# ===============================================================
# VF ANALYSES
# ================================================================
f1 = df0_all[df0_all['id']==0].copy()
#%
f1['ang_vel_abs'] = np.abs(f1['ang_vel'])
f1 = util.shift_variables_by_lag(f1, file_grouper='file_name', lag=12)
f1['ang_vel_fly_shifted_abs'] = np.abs(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_shifted_deg'] = np.rad2deg(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_deg'] = np.rad2deg(f1['ang_vel_fly'])

#%%
# Add bins
f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])
f1 = gf.bin_by_object_position(f1, start_bin=-180, end_bin=180, bin_size=20)
f1['binned_theta_error_num'] = pd.to_numeric(f1['binned_theta_error'], errors='coerce')

f1.reset_index(drop=True, inplace=True)

#%%
# Get average by fly
# # Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error',
          'binned_theta_error_num', 'pr_direction']
yvar = 'ang_vel_fly_shifted_deg'
mean_f1 = f1.groupby(grouper)[yvar].mean().reset_index()

#%%
# GAIN PLOTS
# ------------------------------------------------------------
pr_palette = {'progressive': 'darkgreen', 'regressive': 'purple'}
lw=2
err = 'se'

# Plot
fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
for i, (sp, df_) in enumerate(f1.groupby('species')):
    ax=axn[i]
    #sns.lineplot(data=df_, ax=ax,
    #    x='binned_theta_error', y='ang_vel_fly_shifted_deg',
    #    hue='pr_direction', palette=pr_palette)
    sns.lineplot(data=df_[df_['binned_theta_error_num']<0], 
                x='binned_theta_error', y=yvar, ax=ax,
                hue='pr_direction', palette=pr_palette, 
                errorbar=err, marker='o', 
                markersize=0, markeredgewidth=0,
                err_style='bars', legend=0, lw=lw, err_kws={'linewidth': lw})
    sns.lineplot(data=df_[df_['binned_theta_error_num']>0], 
                x='binned_theta_error', y=yvar, ax=ax,
                hue='pr_direction', palette=pr_palette, 
                errorbar=err, marker='o', 
                markersize=0, markeredgewidth=0,
                err_style='bars', lw=lw, err_kws={'linewidth': lw},
                legend=i==1)
    ax.set_title(f'{sp}')
    ax.set_xlabel('Theta error (deg)')
    ax.set_ylabel('Ang vel (deg/s)')
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                    frameon=False, title='', 
                    fontsize=min_fontsize)
ax.set_xlim([-180, 180])
ax.set_xticks(np.linspace(-180, 180, 9))

putil.label_figure(fig, figid)
figname = 'gain_species'
plt.savefig(os.path.join(figdir, f'{figname}.png'))

#%%
# Distn of targets
# ------------------------------------------------------------

# from analyses.preprocessing.src.add_ft_actions import add_ft_actions

#%%
df_list = []
for currf, df_ in df0_all.groupby('file_name'):
    #df_ = df0_all[df0_all['file_name']==currf].copy()
    actions_fpath = os.path.join(videodir, currf, currf, f'{currf}-actions.mat')
    assert os.path.exists(actions_fpath), f'File {actions_fpath} does not exist'

    # Load actions
    actions = util.load_ft_actions([actions_fpath], split_end=False)
    df_ = util.assign_action_frames_to_df(df_, actions)

    df_list.append(df_)

df0_all = pd.concat(df_list)

# %%
fig, ax = plt.subplots(figsize=(5, 5))
f1 = df0_all[df0_all['id']==0].copy()

# Get df for curr file
crt_df = f1[(f1['courtship']==True)].copy()

# Plot
sns.histplot(data=crt_df, ax=ax,
            x= 'targ_pos_theta',
            hue='species', palette=species_palette,
            bins=100, kde=True, 
            stat='probability', common_norm=False)
ax.set_xlabel('Target position (deg)')
ax.set_ylabel('Count')
plt.show()

#%%
# Plot sparse distribution of target position as scatterplot on polar plot
# ------------------------------------------------------------
for i, plot_type in enumerate(['polar', 'scatter']):
    if plot_type == 'polar':
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        markersize = 20
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        markersize = 30
    sns.scatterplot(data=crt_df, ax=ax,
                x='targ_pos_theta', y='targ_pos_radius',
                hue='species', palette=species_palette, 
                alpha=0.1, s=markersize, edgecolor='none')
    # if polar, set theta_zero_location to N
    if plot_type == 'polar':
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), 
                    frameon=False, title='', fontsize=min_fontsize)  
    putil.label_figure(fig, figid)
    figname = f'targ_pos_{plot_type}'
    plt.savefig(os.path.join(figdir, f'{figname}.png'))

# # For each acquisition, plot box plot of target position
# fig, axn = plt.subplots(1, 2, figsize=(10, 5), 
#                         sharex=True, sharey=True)
# 
# for i, (sp, spdf) in enumerate(df0_all.groupby('species')):
#     ax=axn[i]
#     sns.boxplot(data=spdf, ax=ax,
#                 x='acquisition', y='targ_pos_theta',
#                 color=bg_color)
#                 #hue='species', palette=species_palette)
#     ax.set_title(f'{sp}')
#     # Rotate x-ticks to be vertical
#     ax.tick_params(axis='x', labelrotation=90)
#     ax.set_xlabel('Acquisition')
#     ax.set_ylabel('Target position (deg)')


#%%
# Fraction of courtship frames in frontal and lateral VF
# ------------------------------------------------------------
import scipy.stats as spstats

frontal_deg = 25 #= np.deg2rad(30)
lateral_deg = 45 #= np.deg2rad(50)

# Assign targ_pos_theta to vf_position, frontal or lateral
f1['vf_position'] = np.nan
f1.loc[f1['targ_pos_theta'] < np.deg2rad(frontal_deg), 'vf_position'] = 'frontal'
f1.loc[f1['targ_pos_theta'] >= np.deg2rad(lateral_deg), 'vf_position'] = 'lateral'

#%
# Plot N of frames in each vf_position for each species
crt = f1[(f1['courtship']==True)].copy()

# Get fraction of frames in each vf_position for each acquisition
vf_position_frac = crt.groupby(['species', 'acquisition', 'vf_position'])['frame'].count() / crt.groupby('acquisition')['frame'].count()
vf_position_frac = vf_position_frac.reset_index()
vf_position_frac.columns = ['species', 'acquisition', 'vf_position', 'fraction']

# Plot distribution of fraction of frames in each vf_position for each acquisition
fig, axn = plt.subplots(1, 2, figsize=(5, 3))
for i, (vf, vf_df) in enumerate(vf_position_frac.groupby('vf_position')):
    ax=axn[i]
    sns.barplot(data=vf_df, ax=ax,
                x='species', y='fraction',
                hue='species', palette=species_palette,
                fill=False) # width=bar_width)
    ax.legend_.remove()
    sns.stripplot(data=vf_df, ax=ax,
                x='species', y='fraction',
                hue='species', palette=species_palette,
                legend=0)
    ax.set_xlabel('')

    # title
    if vf == 'frontal':
        ax.set_title(f'Frontal: targ. <= {frontal_deg} deg')
    elif vf == 'lateral':
        ax.set_title(f'Lateral: targ. >= {lateral_deg} deg')

    # Do stats to test for species difference
    res = spstats.mannwhitneyu(vf_df[vf_df['species']=='Dmel']['fraction'],
                               vf_df[vf_df['species']=='Dyak']['fraction'])
    print(f'{vf}: {res.pvalue}')
    putil.annotate_axis(ax, sutil.pval_to_stars(res.pvalue),
                        color=bg_color, fontsize=min_fontsize+2)

    # Set y-tick labels
    max_y = round(vf_df['fraction'].max(), 1)
    n_ticks = 4 if max_y%0.2==0 else 5    
    ax.set_yticks(np.linspace(0, max_y+0.1, n_ticks))
    ax.set_ylabel('Fraction of courtship frames')
    ax.set_xlabel('')
    sns.despine(offset=4, trim=True, bottom=True)

    ax.set_box_aspect(2)

plt.subplots_adjust(hspace=0.8)

putil.label_figure(fig, figid)
figname = 'vf_position_frac_courtship'
plt.savefig(os.path.join(figdir, f'{figname}.png'))
plt.savefig(os.path.join(figdir, f'{figname}.svg'))
# %%
