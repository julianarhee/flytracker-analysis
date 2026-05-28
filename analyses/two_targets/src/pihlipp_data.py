#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:00:00 2026

Author: Juliana Rhee
Email:  juliana.rhee@gmail.com

This script imports the data from Pihlipp's experiment and processes it.
"""
#%%
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import libs.plotting as putil
import libs.utils as util
import libs.utils_2p as util2p
import analyses.steering.src.gain_funcs as gf 
import transform_data.relative_metrics as rem
#import mat73
import scipy.stats as spstats


#%% 
# Plotting
plot_style = 'white'
min_fontsize = 6
putil.set_sns_style(plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'
species_colors = {'mel': 'cyan', 'yak': 'magenta'}

# Embed fonts as actual text in PDF/SVG exports (not outlines)
plt.rcParams['pdf.fonttype'] = 42   # TrueType fonts in PDF
plt.rcParams['ps.fonttype'] = 42    # TrueType fonts in PS
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG

#%%
rootdir = '/Volumes/Juliana/2d-projector'
session = 'philipp-data'
basedir = os.path.join(rootdir, session)
acquisitio_parentdir = os.path.join(basedir, 'tracked')
acqs = os.listdir(acquisitio_parentdir)
print(f"Found {len(acqs)} acquisitions")

figdir = os.path.join(basedir, 'figures')

if plot_style=='white':
    figdir = os.path.join(figdir, 'white')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(f"Saving figs to: {figdir}")



# %%
mel_acqs = [acq for acq in acqs if 'csMai' in acq]
yak_acqs = [acq for acq in acqs if 'yak' in acq]
# 
# acq = yak_acqs[0]
# 
# common_cols = ['id', 'frame', 'sec']
# 
# tf_list = []
# for acq in yak_acqs:
#     acq_dir = os.path.join(basedir, acq)
#     print(f"Acquisition directory: {acq_dir}")
# 
#     # Load virmen trk and feat
#     calib, trk, feat = util.load_flytracker_data(acq_dir, 
#                             calib_is_upstream=False, filter_ori=True)
#     #print(acq, trk.shape, feat.shape)
# 
#     #%
#     # Get fly IDs
#     sex_ids = os.path.join(acq_dir, 'sex_index.txt')
#     # This is a text file with headers m, f, f, and their corresponding id values
#     header = [0]
#     sex_ids = pd.read_csv(sex_ids, sep=',', header=header)
#     print(sex_ids)
#     assert sex_ids['m'].values[0]==1, f"{acq}: Check sex index file"
# 
#     # merge feat and trk by common columns
#     tf = pd.merge(trk, feat, on=common_cols, how='inner')
#     assert tf.shape[1] == (trk.shape[1] + feat.shape[1] - len(common_cols)), "Bad merge: {acq}, {tf.shape[1]} != {trk.shape[1]} + {feat.shape[1]} - {len(common_cols)}"
#     tf['acquisition'] = acq
#     tf['species'] = 'yak'
#     tf_list.append(tf)
# 
# print(f"Found {len(tf_list)} acquisitions")
# #%%
# 
# # Now add mel
# for acq in mel_acqs:
#     acq_dir = os.path.join(basedir, acq)
#     print(f"Acquisition directory: {acq_dir}")
# 
#     # Load virmen trk and feat
#     calib, trk, feat = util.load_flytracker_data(acq_dir, 
#                             calib_is_upstream=False, filter_ori=True)
#     print(acq, trk.shape, feat.shape)
#     #%
#     # Get fly IDs
#     sex_ids = os.path.join(acq_dir, 'sex_index.txt')
#     # This is a text file with headers m, f, f, and their corresponding id values
#     header = [0]
#     sex_ids = pd.read_csv(sex_ids, sep=',', header=header)
#     print(sex_ids)
#     assert sex_ids['m'].values[0]==1, f"{acq}: Check sex index file"
# 
#     # merge feat and trk by common columns
#     tf = pd.merge(trk, feat, on=common_cols, how='inner')
#     assert tf.shape[1] == (trk.shape[1] + feat.shape[1] - len(common_cols)), "Bad merge: {acq}, {tf.shape[1]} != {trk.shape[1]} + {feat.shape[1]} - {len(common_cols)}"
# 
#     tf['acquisition'] = acq
#     tf['species'] = 'mel'
#     tf_list.append(tf)
# 
# print("Total number of acquisitions: ", len(tf_list))
# 
 #%%
# Combine
# trackdf = pd.concat(tf_list)

# Save to parquet
#aggr_fpath = os.path.join(basedir, 'aggr_trk_feat.parquet')
#trackdf.to_parquet(aggr_fpath)

# %%
# trk = trackdf[trackdf['acquisition']==yak_acqs[2]].copy()
# 
# id_colors = {0: 'k', 1: 'r', 2: 'b'}
# fig, ax  = plt.subplots( figsize=(10, 5))
# for id, trk_ in trk.groupby('id'):
#     ax.plot(trk_['pos_x'], trk_['pos_y'], 
#         color=id_colors[id], alpha=0.5, lw=0.5)
# #sns.scatterplot(data=trk, ax=ax,
# #                x='pos_x', y='pos_y', hue='id', palette=id_colors,
# #                alpha=0.5, edgecolor='none', s=1)
# #ax.legend_.remove()
# ax.set_aspect(1)
# ax.set_title(acq)
# %%
# Transform data
#def add_pairwise_metrics(trk_, feat_, calib):
#    ppm = calib.get('PPM', 1)
#    for fid, oid in [(flyid1, flyid2), (flyid2, flyid1)]:
#        f_trk = trk_[trk_['id']==fid]
#        o_trk = trk_[trk_['id']==oid]
#        dist = util.compute_dist_to_other(
#            f_trk['pos_x'].values, f_trk['pos_y'].values,
#            o_trk['pos_x'].values, o_trk['pos_y'].values,
#            pix_per_mm=ppm)
#        feat_.loc[feat_['id']==fid, 'dist_to_other'] = dist
#        facing_angle = util.compute_facing_angle(
#            f_trk['ori'].values, f_trk['pos_x'].values, f_trk['pos_y'].values,
#            o_trk['pos_x'].values, o_trk['pos_y'].values)
#        feat_.loc[feat_['id']==fid, 'facing_angle'] = facing_angle
#
#    return trk_, feat_

def plot_switch_bouts(f1, frames, switch_positions, fps, species='', acq='',
                      n_sec_pre=1.0, n_sec_post=1.0, n_plots=None, 
                      xlim=650, start_color=[0.7]*3,
                      figdir=None, min_chase_dur=1):
    '''Plot absolute and relative position around each target switch.

    Shows a fixed time window (n_sec_pre, n_sec_post) around each switch
    frame, including all frames regardless of target_id (0 = not facing
    either target, 1 = facing target 1, 2 = facing target 2).

    Args
    ----
    f1 :               (pd.DataFrame) fly 0 data with target_id, frame, pos, targ_rel_pos columns
    frames :           (np.ndarray) frame values array (f1['frame'].values)
    switch_positions : (list) indices into frames where switches occur
    fps :              (float) frames per second
    species, acq :     (str) labels for figure title
    n_sec_pre :        (float) seconds to show before switch (default 1.0)
    n_sec_post :       (float) seconds to show after switch (default 1.0)
    n_plots :          (int|None) number of switches to plot (None = all)
    xlim :             (float) axis limits for relative position plot
    figdir :           (str|None) directory to save figures (None = don't save)
    '''
    target_colors = {0: [0.5, 0.5, 0.5], 1: [0.2, 0.6, 1.0], 2: [1.0, 0.3, 0.3]}
    markers = {0: '.', 1: 'o', 2: 's'}
    n_frames_pre = int(n_sec_pre * fps)
    n_frames_post = int(n_sec_post * fps)

    positions = switch_positions[:n_plots] if n_plots is not None else switch_positions
    for si, ix in enumerate(positions):
        switch_frame = frames[ix]
        curr_frames = np.arange(switch_frame - n_frames_pre, switch_frame + n_frames_post + 1)
        curr_df = f1[f1['frame'].isin(curr_frames)]
        if len(curr_df) == 0:
            continue

        fig, axn = plt.subplots(1, 2, figsize=(10, 5))
        frame_range = curr_df['frame'].max() - curr_df['frame'].min()
        alpha = (curr_df['frame'].values - curr_df['frame'].min()) / max(frame_range, 1)
        alpha = 0.1 + 0.85 * alpha

        for tid, grp in curr_df.groupby('target_id'):
            mask = curr_df['target_id'] == tid
            axn[0].scatter(curr_df.loc[mask, 'pos_x'], curr_df.loc[mask, 'pos_y'],
                    c=[target_colors[tid]], alpha=alpha[mask],
                    marker=markers[tid], s=40, label=f'target {tid}')
            axn[1].scatter(curr_df.loc[mask, 'targ_rel_pos_y'],
                           curr_df.loc[mask, 'targ_rel_pos_x'],
                    c=[target_colors[tid]], alpha=alpha[mask],
                    marker=markers[tid], s=40, label=f'target {tid}')

        switch_row = curr_df[curr_df['frame'] == switch_frame]
        axn[1].scatter(switch_row['targ_rel_pos_y'], switch_row['targ_rel_pos_x'],
                c=start_color, marker='*', s=200, zorder=10, label='switch')

        axn[1].axvline(x=0, color='0.7', linestyle='--', lw=0.5)
        axn[1].axhline(y=0, color='0.7', linestyle='--', lw=0.5)
        axn[1].set_xlim([-xlim, xlim])
        axn[1].set_ylim([-xlim, xlim])
        axn[0].set_xlabel('pos_x'); axn[0].set_ylabel('pos_y')
        axn[1].set_xlabel('targ_rel_pos_y'); axn[1].set_ylabel('targ_rel_pos_x')
        for ax in axn:
            ax.set_aspect(1)
        fig.suptitle(f'{species} {acq}\nswitch {si+1}/{len(switch_positions)}, '
                     f'frame {switch_frame}, min chase >= {min_chase_dur}s', fontsize=10)
        fig.tight_layout()
 
        if figdir is not None:
            curr_fig_dir = os.path.join(figdir, acq)
            if not os.path.exists(curr_fig_dir):
                os.makedirs(curr_fig_dir)
            figname = f'{species}_switch{si+1}_frame{switch_frame}.png'
            fig.savefig(os.path.join(curr_fig_dir, figname), bbox_inches='tight')



#%%
# Check 1 pair at a time
target_colors = {0: [0.5, 0.5, 0.5], 1: [0.2, 0.6, 1.0], 2: [1.0, 0.3, 0.3]}

idx = 2
#acqs = [mel_acqs[5], yak_acqs[ix]]
#acqs = [mel_acqs[5], yak_acqs[1]]

mel_ex = mel_acqs[1]
yak_ex = yak_acqs[6] # 4, 6
acqs = [mel_ex, yak_ex]
#acqs = [mel_acqs[idx], yak_acqs[idx]]
acq_str = '_'.join(acqs)

fig, axn  = plt.subplots(1, 2, figsize=(10, 5), 
                         sharex=True, sharey=True)

for i, acq in enumerate(acqs):
    print(acq)
    ax=axn[i]
    acq_dir = os.path.join(basedir, 'tracked', acq)
    calib, trk, feat = util.load_flytracker_data(acq_dir, 
                            calib_is_upstream=False, filter_ori=True)

    #id_colors = {0: bg_color, 1: 'r', 2: 'b'}
    species = 'yak' if 'yak' in acq else 'mel'
    ax.set_title(species)
    for id, trk_ in trk.groupby('id'):
        ax.plot(trk_['pos_x'], trk_['pos_y'], 
            color=target_colors[id], alpha=0.5, lw=0.5)
    #sns.scatterplot(data=trk, ax=ax,
    #                x='pos_x', y='pos_y', hue='id', palette=id_colors,
    #                alpha=0.5, edgecolor='none', s=1)
    #ax.legend_.remove()
    ax.set_aspect(1)

figname = f'trajectories_mel-v-yak_{acq_str}'
print(figname)
plt.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')

#%%
processedmat_dir = os.path.join(basedir, 'processed_mats')
if not os.path.exists(processedmat_dir):
    os.makedirs(processedmat_dir)
#print(f"Acquisition directory: {acq_dir}")

all_acqs = mel_acqs + yak_acqs
print(f"Found {len(all_acqs)} acquisitions")
assert len(all_acqs) == len(mel_acqs) + len(yak_acqs), "Not all acquisitions found"

# check if aggregate data exists
processed_fpath = os.path.join(basedir, 'processed_mel_yak.parquet')

load_feat_trk = False

#%%
new_feat_trk = False
load_feat_trk = True
if load_feat_trk:
    if os.path.exists(processed_fpath):
        print(f"Loading existing data from {processed_fpath}")
        alldata = pd.read_parquet(processed_fpath)
    else:
        print(f"Creaing NEW data and saving to {processed_fpath}")
        new_feat_trk = True

#%%
if load_feat_trk and new_feat_trk:
    comb_list = []
    for acq in all_acqs:
        acq_dir = os.path.join(basedir, acq)
        # Load virmen trk and feat
        calib, trk, feat = util.load_flytracker_data(acq_dir, 
                                calib_is_upstream=False, filter_ori=True)
        frame_width = calib['w']
        frame_height = calib['h']
        cop_ix = None
        species = 'yak' if 'yak' in acq else 'mel'
        #print(species)

        df_list = []
        # Get pairwise info
        for flyid1, flyid2 in [(0, 1), (0, 2)]:
        #flyid1 = 0
        #flyid2 = 1
            curr_ids = [flyid1, flyid2]
            trk_ = trk[trk['id'].isin(curr_ids)].copy()
            feat_ = feat[feat['id'].isin(curr_ids)].copy()

            # Add dist_to_other and facing_angle if not in columns
            if 'dist_to_other' not in feat_.columns:
                trk_, feat_ = add_pairwise_metrics(trk_, feat_, calib)

            trk_['ori'] = -1*trk_['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
            df_ = rem.do_transformations_on_df(trk_, frame_width, frame_height, 
                                            feat_=feat_, cop_ix=cop_ix,
                                            flyid1=flyid1, flyid2=flyid2, 
                                            get_relative_sizes=False)

            df_['pair'] = f"{flyid1}_{flyid2}"
            df_['acquisition'] = acq
            df_['species'] = species

            df_list.append(df_)

        comb = pd.concat(df_list)
        # Save to procssed_mat dir
        out_fpath = os.path.join(processedmat_dir, f'{acq}_comb.parquet')
        comb.to_parquet(out_fpath)
        print(f"Saved to {out_fpath}")
        comb_list.append(comb)

    alldata = pd.concat(comb_list)
    print(alldata['species'].unique())

    # Save all data
    alldata.to_parquet(out_fpath)
    print(f"Saved to {out_fpath}")

#%% 
# EXAMPLE: plot position of dot in male's foV
# =========================================
curr_acq = yak_acqs[1]
#curr_acq = mel_acqs[1]
comb = alldata[alldata['acquisition']==curr_acq].copy()
#hue_norm = (np.deg2rad(-60), np.deg2rad(60)) #(-np.pi, np.pi)
hue_norm = None
min_facing_angle = np.deg2rad(60)
min_vel = 10

chasedf = comb[(comb['species']=='yak')
             & (comb['facing_angle']<=min_facing_angle)
             & (comb['vel']>=min_vel)
             ].copy()
chase_frames = chasedf['frame'].unique()

filtdf = comb[comb['frame'].isin(chase_frames)]

for sp, c_ in filtdf.groupby('species'):
    fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                            sharex=True, sharey=True)
    for i, (fpair, pairdf_) in enumerate(c_.groupby('pair')):
        ax=axn[i]
        ax.plot(c_[c_['id']==0]['pos_x'], 
                c_[c_['id']==0]['pos_y'], 
                color='k', alpha=0.2, lw=0.5)
        sns.scatterplot(data=pairdf_[pairdf_['id']==0], ax=ax, 
                        x='pos_x', y='pos_y', 
                        #hue='theta_error', palette='RdBu',
                        hue='dist_to_other', palette='viridis',
                        hue_norm=hue_norm,
                        size='dist_to_other', size_norm=(0, 500),
                        alpha=1, edgecolor='none')
        # other flyid
        other_id = int(fpair.split('_')[1])
        ax.plot(pairdf_[pairdf_['id']==other_id]['pos_x'], 
                pairdf_[pairdf_['id']==other_id]['pos_y'], 'k.', 
                markersize=10)
        sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)     
        ax.set_aspect(1)
        ax.set_title(f'{sp} pair: {fpair}')

#%%
# Get switch df
new_switches = False

f1_switch_fpath = os.path.join(basedir, 'f1_switches.parquet')
switch_fpath = os.path.join(basedir, 'switch_df.parquet')

if os.path.exists(f1_switch_fpath):
    print(f"Loading existing data from {f1_switch_fpath}")
    all_f1 = pd.read_parquet(f1_switch_fpath)

    # Load switch df
    switch_df = pd.read_parquet(switch_fpath)
    #switches = switch_df.groupby('acquisition')['switch_frame'].apply(list).to_dict()
else:
    print(f"Creating new data and saving to {f1_switch_fpath}")
    new_switches = True

# %%
new_switches = True
plot_switches = False

if new_switches:
    # Cycle trhough ALL data and get switches
    # =========================================
    #sp = 'mel'
    #sp_ix = 0 if sp == 'mel' else 1
    #curr_fly = comb[comb['species']==sp].copy()
    #acq = acqs[sp_ix]
    fps = calib['FPS']
    min_chase_dur = 1
    max_gap_frames = int(0.1 * fps)
    min_chase_frames = int(min_chase_dur * fps)
    switches = dict()
    direct_facing_angle = np.deg2rad(80)

    f1_list = []

    for (sp, acq), curr_fly in alldata.groupby(['species', 'acquisition']):
        # Find where male fly (id=0) is facing each of the other flies separately
        fpair = '0_1'
        pair1 = curr_fly[(curr_fly['id']==0) 
                    & (curr_fly['acquisition']==acq)
                    & (curr_fly['pair']==fpair)
                    & (curr_fly['facing_angle']<=direct_facing_angle)
                    ].copy()
        fpair2 = '0_2'
        pair2 = curr_fly[(curr_fly['id']==0) 
                    & (curr_fly['acquisition']==acq)
                    & (curr_fly['pair']==fpair2)
                    & (curr_fly['facing_angle']<=direct_facing_angle)
                    ].copy()

        # Assign target_id to 1 if in pair1 frames
        pair1_frames = pair1['frame'].unique()
        pair2_frames = pair2['frame'].unique()
        curr_fly['target_id'] = 0
        curr_fly.loc[curr_fly['frame'].isin(pair1_frames), 'target_id'] = 1
        curr_fly.loc[curr_fly['frame'].isin(pair2_frames), 'target_id'] = 2

        # Drop duplicate frames (fly 0 appears once per pair in comb)
        f1 = curr_fly[curr_fly['id']==0].drop_duplicates(subset='frame').copy()

        # Find switches between target 1 and 2, ignoring brief gaps of target_id=0.
        # Collapse the sequence to only frames where target_id != 0, then find
        # transitions from 1->2 or 2->1 that are separated by <= max_gap_frames of 0s.
        tids = f1['target_id'].values
        frames = f1['frame'].values

        nonzero_mask = tids != 0
        nz_indices = np.where(nonzero_mask)[0]
        nz_tids = tids[nz_indices]

        # Find runs of consecutive target_id in the nonzero sequence.
        # Each run records (target_id, start_index, end_index) in nz_indices space.
        runs = []
        run_start = 0
        for i in range(1, len(nz_tids)):
            if nz_tids[i] != nz_tids[run_start]:
                runs.append((nz_tids[run_start], run_start, i - 1))
                run_start = i
        if len(nz_tids) > 0:
            runs.append((nz_tids[run_start], run_start, len(nz_tids) - 1))

        # A valid switch requires:
        #   1. consecutive runs with different target_id (1->2 or 2->1)
        #   2. gap between runs <= max_gap_frames
        #   3. the pre-switch run lasted >= min_chase_frames
        switch_positions = []
        for r in range(1, len(runs)):
            prev_tid, prev_start, prev_end = runs[r - 1]
            curr_tid, curr_start, curr_end = runs[r]
            if prev_tid == curr_tid:
                continue
            gap = frames[nz_indices[curr_start]] - frames[nz_indices[prev_end]]
            prev_dur = frames[nz_indices[prev_end]] - frames[nz_indices[prev_start]]
            if gap <= max_gap_frames and prev_dur >= min_chase_frames:
                switch_positions.append(nz_indices[curr_start])

        print(f"{sp}: Found {len(switch_positions)} target switches "
              f"(1<->2, gap <= {max_gap_frames} frames, min chase >= {min_chase_frames} frames)")

        switches[acq] = switch_positions

        f1['acquisition'] = acq
        f1['species'] = sp
        f1_list.append(f1)

        if plot_switches:
            plot_switch_bouts(f1, frames, switch_positions, fps,
                          species=sp, acq=acq, n_plots=5,
                          figdir=figdir, min_chase_dur=min_chase_dur)

    all_f1 = pd.concat(f1_list)

    all_f1 = all_f1.reset_index(drop=True)
    print(all_f1.shape)
    # Save
    all_f1.to_parquet(f1_switch_fpath)
    print(f"Saved to {f1_switch_fpath}")

    #%
    # Save switches, convert to dataframe first
    d_list = []
    for acq, switch_positions in switches.items():
        sp = 'yak' if 'yak' in acq else 'mel'
        d_ = pd.DataFrame(data={'acquisition': acq, 
                                'switch_frame': switch_positions,
                                'species': sp})
        d_list.append(d_)
    switch_df = pd.concat(d_list)
    #%
    switch_df.to_parquet(switch_fpath)

    print(f"Saved to {switch_fpath}")

#%% Examine switches for a single acquisition
test_sp = 'yak'
test_acq = mel_acqs[1] if test_sp == 'mel' else yak_acqs[4]
#test_acq = yak_acqs[4]
curr_fly = alldata[(alldata['species']==test_sp) 
                  & (alldata['acquisition']==test_acq)].copy()

#fps = calib['FPS']
#min_chase_dur = 1 #2.0 
#max_gap_frames = int(0.1 * fps)
#min_chase_frames = int(min_chase_dur * fps)
#direct_facing_angle = np.deg2rad(60)

pair1 = curr_fly[(curr_fly['id']==0) 
            & (curr_fly['pair']=='0_1')
            & (curr_fly['facing_angle']<=direct_facing_angle)].copy()
pair2 = curr_fly[(curr_fly['id']==0) 
            & (curr_fly['pair']=='0_2')
            & (curr_fly['facing_angle']<=direct_facing_angle)].copy()

curr_fly['target_id'] = 0
curr_fly.loc[curr_fly['frame'].isin(pair1['frame'].unique()), 'target_id'] = 1
curr_fly.loc[curr_fly['frame'].isin(pair2['frame'].unique()), 'target_id'] = 2

f1 = curr_fly[curr_fly['id']==0].drop_duplicates(subset='frame').copy()
tids = f1['target_id'].values
frames = f1['frame'].values

nonzero_mask = tids != 0
nz_indices = np.where(nonzero_mask)[0]
nz_tids = tids[nz_indices]

runs = []
run_start = 0
for i in range(1, len(nz_tids)):
    if nz_tids[i] != nz_tids[run_start]:
        runs.append((nz_tids[run_start], run_start, i - 1))
        run_start = i
if len(nz_tids) > 0:
    runs.append((nz_tids[run_start], run_start, len(nz_tids) - 1))

switch_positions = []
for r in range(1, len(runs)):
    prev_tid, prev_start, prev_end = runs[r - 1]
    curr_tid, curr_start, curr_end = runs[r]
    if prev_tid == curr_tid:
        continue
    gap = frames[nz_indices[curr_start]] - frames[nz_indices[prev_end]]
    prev_dur = frames[nz_indices[prev_end]] - frames[nz_indices[prev_start]]
    if gap <= max_gap_frames and prev_dur >= min_chase_frames:
        switch_positions.append(nz_indices[curr_start])

print(f"Found {len(switch_positions)} switches for {test_acq}")

plot_switch_bouts(f1, frames, switch_positions, fps,
                  species=test_sp, acq=test_acq, n_plots=30,
                  figdir=figdir, min_chase_dur=min_chase_dur)

#%%
# EXAMPLE: Tracking frames, split inner vs. outer target
# =========================================
#curr_acq = yak_acqs[1]
#curr_acq = mel_acqs[1]
#mel_ex = mel_acqs[1]
#yak_ex = yak_acqs[4] # 4, 6

curr_acq = yak_ex #mel_ex
min_vel = 10
# min_facing_angle = np.deg2rad(20)
xlim = [50, 750]
ylim = [50, 750]

id_colors = {1: 'r', 2: 'cornflowerblue'}
curr_fly = alldata[alldata['acquisition']==curr_acq].copy()
sp = 'yak' if 'yak' in curr_acq else 'mel'
# # For a single pair1, pair2 set, plot position of male
# # and target ID frames
# Find where male fly (id=0) is facing each of the other flies separately
fpair = '0_1'
pair1 = curr_fly[ #(curr_fly['id']==0) 
             (curr_fly['acquisition']==curr_acq)
            & (curr_fly['pair']==fpair)
            & (curr_fly['facing_angle']<=direct_facing_angle)
            ].copy()
fpair2 = '0_2'
pair2 = curr_fly[ #[(curr_fly['id']==0) 
             (curr_fly['acquisition']==curr_acq)
            & (curr_fly['pair']==fpair2)
            & (curr_fly['facing_angle']<=direct_facing_angle)
            ].copy()

fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                        sharex=True, sharey=True)

for other_id, pair_ in zip([1, 2], [pair1, pair2]):
    ax=axn[other_id-1]
    ax.scatter(pair_[pair_['id']==other_id]['pos_x'],
               pair_[pair_['id']==other_id]['pos_y'],
               color=target_colors[other_id], s=5)
    ax.scatter(pair_[pair_['id']==0]['pos_x'],
               pair_[pair_['id']==0]['pos_y'],
               color=bg_color, s=5, alpha=0.3) 
    ax.set_title(f'{sp}, target={other_id}')
    ax.set_aspect(1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fig.suptitle(f'{sp}, min_angle={np.rad2deg(direct_facing_angle).round(1)}, min_vel={min_vel}')

figname = f'tracking_frames_{sp}_min-angle-{direct_facing_angle}_chase-dur-{min_chase_dur}s_{curr_acq}'
print(figname)
plt.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')

#%%
# Plot ALL SWITCH FRAMES per fly: 
# =========================================
n_sec_pre = 0.2
n_sec_post = 0.
xlim = 650
markersize=20
fps = 30
n_frames_pre = int(n_sec_pre * fps)
n_frames_post = int(n_sec_post * fps)
print(n_frames_pre, n_frames_post)

target_colors = {0: [0.5, 0.5, 0.5], 1: [0.2, 0.6, 1.0], 2: [1.0, 0.3, 0.3]}
markers = {0: '.', 1: 'o', 2: 's'}

for (sp, acq), f1 in all_f1.groupby(['species', 'acquisition']):
    fig, axn = plt.subplots(1, 2, figsize=(10, 5))
    #print(acq)
    #%
    #fig, axn = plt.subplots(1, 2, figsize=(10, 5))
    switch_positions = switch_df[switch_df['acquisition']==acq]['switch_frame'].values #switches[acq]
    print(f"Found {len(switch_positions)} switches for {acq}")
    
    frames = f1['frame'].values
    for ix in switch_positions[0:100]:
        switch_frame = frames[ix]
        curr_frames = np.arange(switch_frame - n_frames_pre, switch_frame + n_frames_post + 1)
        curr_df = f1[f1['frame'].isin(curr_frames)]

        # %
        #ix = 0
        #switch_ = switch_positions[ix]
        #switch_frame = frames[switch_]
        #print(switch_frame)
        #curr_frames = np.arange(switch_frame - n_frames_pre, switch_frame + n_frames_post)
        #curr_df = f1[f1['frame'].isin(curr_frames)]

        frame_range = curr_df['frame'].max() - curr_df['frame'].min()
        alpha = (curr_df['frame'].values - curr_df['frame'].min()) / max(frame_range, 1)
        alpha = 0.1 + 0.6 * alpha  # range [0.15, 1.0] so early points are still visible
        for tid, grp in curr_df.groupby('target_id'):
            mask = curr_df['target_id'] == tid
            ax=axn[0]
            ax.scatter(curr_df.loc[mask, 'pos_x'], curr_df.loc[mask, 'pos_y'],
                    c=[target_colors[tid]], alpha=alpha[mask],
                    marker=markers[tid], s=markersize, label=f'target {tid}')
            ax=axn[1]
            ax.scatter(curr_df.loc[mask, 'targ_rel_pos_y'], curr_df.loc[mask, 'targ_rel_pos_x'],
                    c=[target_colors[tid]], alpha=alpha[mask],
                    marker=markers[tid], s=markersize, label=f'target {tid}',
                    )
            ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
            ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
            ax.set_xlim([-xlim, xlim])
            ax.set_ylim([-xlim, xlim])
    axn[1].set_title(f'{sp}i, n={len(switch_positions)} switches', loc='left')

    for ax in axn:
        ax.set_aspect(1)

    figname = f'{sp}_{acq}_target_switches_min-chase-{min_chase_dur}s'
    fig.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')

# %%

# Plot switch frames for each species
# =====================================
alpha=0.1
markersize=30
plot_diagonals = False
#n_sec_pre=0.2
n_sec_pre_switch = 0.2
n_frames_pre_switch = int(n_sec_pre_switch * fps)
n_sec_pre = 0.5
n_sec_post = .1
n_frames_pre = int(n_sec_pre * fps)
n_frames_post = int(n_sec_post * fps)
print(n_frames_pre_switch)

marker = 'o'
xlim = [-600, 600]
ylim = [-600, 600]
# -------------------------------------

#for sp, n_ in switch_counts.groupby('species'):
counts = switch_df.groupby(['species', 'acquisition'], as_index=False).count().reset_index()
counts.rename(columns={'switch_frame': 'n_switches'}, inplace=True)
# Get mean and std of n_switches per species
mean_counts = counts.groupby('species', as_index=True)['n_switches'].mean()#@.reset_index()
std_counts = counts.groupby('species', as_index=True)['n_switches'].std()#.reset_index()


n_frames_pre = int(n_sec_pre * fps)
switch_frames_list = []
fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                        sharex=True, sharey=True)
for (sp, acq), f1 in all_f1.groupby(['species', 'acquisition']):
    switch_positions = switch_df[switch_df['acquisition']==acq]['switch_frame'].values #switches[acq]
    frames = f1['frame'].values
    if sp=='mel':
        ax=axn[0]
    else:
        ax=axn[1]

    # Plot lines for reference
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.1)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.1)

    # Plot scatter
    for ix in switch_positions: #[0:1000]:
        switch_frame = frames[ix]
        curr_frames = np.arange(switch_frame - n_frames_pre, switch_frame + n_frames_post)
        curr_df = f1[f1['frame'].isin(curr_frames)]

        switch_frame_set = curr_frames #[switch_frame-n_frames_pre_switch]
        switch_row = curr_df[curr_df['frame'].isin(switch_frame_set)]
        color = bg_color #species_colors[sp]
        # swap x, y so 0 is upward
        ax.scatter(switch_row['targ_rel_pos_y'], 
                   switch_row['targ_rel_pos_x'],
                c=color, alpha=alpha,
                marker=marker, s=markersize)

        switch_frames_list.append(switch_row.copy())
    ax.set_aspect(1)
    
    # plot a diaonal line across the plot
    if plot_diagonals:
        x = np.linspace(-600, 600, 100)
        y = x
        ax.plot(x, y, color=bg_color, linestyle='--', lw=0.1)
        ax.plot(x, -y, color=bg_color, linestyle='--', lw=0.1)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', pad=6, size=0)
    sns.despine(offset=4, ax=ax)

    ax.set_title(f"{sp}, n={mean_counts.loc[sp].round(2)}±{std_counts.loc[sp].round(2)} switches", loc='left')
    ax.axis('off')

switch_frames_df = pd.concat(switch_frames_list)
# custom legend
#legh = putil.custom_legend(['mel', 'yak'], [species_colors['mel'], species_colors['yak']])
#ax.legend(handles=legh, loc='upper left', bbox_to_anchor=(0.05, 0.95), 
#        frameon=False)

figname = f'all_switch_frames_mel-v-yak_min-chase-{min_chase_dur}s_direct-facing-angle-{direct_facing_angle}'
fig.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')

#ax.legend(loc='upper left', bbox_to_anchor=(1, 1), 
#        frameon=False)
#%%
# Print parameters
print('PARAMETERS:')
print('--------------------------------')
print(f'n_sec_pre: {n_sec_pre}')
print(f'n_sec_post: {n_sec_post}')
print(f'direct_facing_angle: {np.rad2deg(direct_facing_angle)}')
print(f'min_chase_dur: {min_chase_dur}')

# %%
# Plot hist of theta error
# =====================================
figsize_hist = (1, 1)
add_stats = False
# Keep only the row matching the new target's pair at each switch frame.
# target_id indicates which target the fly switched TO, so keep pair '0_{target_id}'.
if 'pair' in switch_frames_df.columns and 'target_id' in switch_frames_df.columns:
    plotd = switch_frames_df[
        switch_frames_df.apply(lambda r: r['pair'] == f"0_{int(r['target_id'])}"
                               if r['target_id'] in [1, 2] else False, axis=1)
    ].copy()
else:
    plotd = switch_frames_df.drop_duplicates(subset=['species', 'acquisition', 'frame']).copy()
hist_var = 'theta_error_deg_abs'
plotd['theta_error_deg_abs'] = np.abs(plotd['theta_error_deg'])
print(f"plotd: {len(plotd)} switch frames (from {len(switch_frames_df)} total rows, "
      f"keeping pair matching new target)")
# Make a histogram of target position
fig, ax = plt.subplots(figsize=figsize_hist)
sns.histplot(data=plotd, x=hist_var, ax=ax,
            hue='species', palette=species_colors,
            common_norm=False, stat='probability', 
            #cumulative=True, bins=50,
            fill=False, element='step')
ax.set_xticks(np.arange(0, 100, 20))
ax.set_xlabel('Position of new target (deg)')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), 
            frameon=False)
for sp in ['mel', 'yak']:
    med = plotd[plotd['species']==sp][hist_var].median()
    ax.axvline(med, color=species_colors[sp], linestyle='--', lw=1.2,
               label=f'{sp} median={med:.1f}')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), frameon=False)
# Statistical tests on theta_error_deg_abs distributions
mel_vals = plotd[plotd['species']=='mel'][hist_var].dropna().values
yak_vals = plotd[plotd['species']=='yak'][hist_var].dropna().values

ks_stat, ks_p = spstats.ks_2samp(mel_vals, yak_vals)
lev_stat, lev_p = spstats.levene(mel_vals, yak_vals, center='median')
mw_stat, mw_p = spstats.mannwhitneyu(mel_vals, yak_vals, alternative='two-sided')

print(f"--- {hist_var} ---")
print(f"  mel: median={np.median(mel_vals):.1f}, IQR=[{np.percentile(mel_vals, 25):.1f}, {np.percentile(mel_vals, 75):.1f}], n={len(mel_vals)}")
print(f"  yak: median={np.median(yak_vals):.1f}, IQR=[{np.percentile(yak_vals, 25):.1f}, {np.percentile(yak_vals, 75):.1f}], n={len(yak_vals)}")
print(f"  KS test (any difference):    stat={ks_stat:.4f}, p={ks_p:.4g}")
print(f"  Levene test (spread):        stat={lev_stat:.4f}, p={lev_p:.4g}")
print(f"  Mann-Whitney U (location):   stat={mw_stat:.1f}, p={mw_p:.4g}")

if add_stats:
    stats_text = (
        f"mel: med={np.median(mel_vals):.1f}, IQR=[{np.percentile(mel_vals,25):.1f}, {np.percentile(mel_vals,75):.1f}], n={len(mel_vals)}\n"
        f"yak: med={np.median(yak_vals):.1f}, IQR=[{np.percentile(yak_vals,25):.1f}, {np.percentile(yak_vals,75):.1f}], n={len(yak_vals)}\n"
        f"KS: p={ks_p:.2g}  Levene: p={lev_p:.2g}  MW: p={mw_p:.2g}"
    )
    fig.text(0.0, -0.2, stats_text, ha='left', va='top',
             fontsize=5.5, family='monospace', color='0.4',
             transform=ax.transAxes)

sns.despine(offset=4, ax=ax)
ax.tick_params(axis='both', which='major', pad=6, size=0)
ax.set_box_aspect(1)
ax.set_title('Error in angle to new target (deg)')
figname= f'hist_theta_error_mel-v-yak_min-chase-{min_chase_dur}s_direct-facing-angle-{direct_facing_angle}'
fig.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')
fig.savefig(os.path.join(figdir, f'{figname}.pdf'), bbox_inches='tight')
# %%

#%% Per-acquisition summary stats to avoid pseudoreplication
theta_thresh = 25 #30  # degrees
per_acq_list = []
for (sp, acq), grp in plotd.groupby(['species', 'acquisition']):
    vals = grp[hist_var].dropna().values
    per_acq_list.append({
        'species': sp,
        'acquisition': acq,
        'median': np.median(vals),
        'std': np.std(vals),
        'iqr': np.percentile(vals, 75) - np.percentile(vals, 25),
        'frac_below_thresh': (vals < theta_thresh).mean(),
        'n': len(vals),
    })
per_acq = pd.DataFrame(per_acq_list)

#print(per_acq)

# Mann-Whitney on per-acquisition medians (n=6 vs n=7)
mel_medians = per_acq[per_acq['species']=='mel']['median'].values
yak_medians = per_acq[per_acq['species']=='yak']['median'].values
mw_acq_stat, mw_acq_p = spstats.mannwhitneyu(mel_medians, yak_medians, alternative='two-sided')
print(f"\n--- Per-acquisition Mann-Whitney (median theta error) ---")
print(f"Is the typical theta error for mel and yak different?")
print(f"  mel (n={len(mel_medians)}): {mel_medians.round(1)}")
print(f"  yak (n={len(yak_medians)}): {yak_medians.round(1)}")
print(f"  U={mw_acq_stat:.1f}, p={mw_acq_p:.4g}")

# Mann-Whitney on per-acquisition IQR (spread)
mel_iqrs = per_acq[per_acq['species']=='mel']['iqr'].values
yak_iqrs = per_acq[per_acq['species']=='yak']['iqr'].values
mw_iqr_stat, mw_iqr_p = spstats.mannwhitneyu(mel_iqrs, yak_iqrs, alternative='two-sided')
print(f"\n--- Per-acquisition Mann-Whitney (IQR theta error) ---")
print(f"Is the spread of theta error for mel and yak different?")
print(f"  mel: {mel_iqrs.round(1)}")
print(f"  yak: {yak_iqrs.round(1)}")
print(f"  U={mw_iqr_stat:.1f}, p={mw_iqr_p:.4g}")

# Mann-Whitney on per-acquisition frac below threshold (precision)
mel_frac = per_acq[per_acq['species']=='mel']['frac_below_thresh'].values
yak_frac = per_acq[per_acq['species']=='yak']['frac_below_thresh'].values
mw_frac_stat, mw_frac_p = spstats.mannwhitneyu(mel_frac, yak_frac, alternative='two-sided')
print(f"\n--- Per-acquisition Mann-Whitney (frac < {theta_thresh} deg) ---")
print(f"Is the precision of theta error for mel and yak different?")
print(f"  mel: {mel_frac.round(3)}")
print(f"  yak: {yak_frac.round(3)}")
print(f"  U={mw_frac_stat:.1f}, p={mw_frac_p:.4g}")

# Plot per-acquisition summaries
summary_vars = ['median', 'iqr', 'frac_below_thresh']
pvals = {'median': mw_acq_p, 'iqr': mw_iqr_p, 'frac_below_thresh': mw_frac_p}

def sig_str(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

fig, axn = plt.subplots(1, len(summary_vars), figsize=(3*len(summary_vars), 3))
for i, var in enumerate(summary_vars):
    ax = axn[i]
    for si, sp in enumerate(per_acq['species'].unique()):
        vals = per_acq[per_acq['species']==sp][var].values
        jitter_x = si + np.random.uniform(-0.15, 0.15, size=len(vals))
        ax.bar(si, np.mean(vals), width=0.5, color=species_colors[sp],
               alpha=0.3, edgecolor=species_colors[sp])
        ax.scatter(jitter_x, vals, color=species_colors[sp],
                   alpha=0.8, s=40, zorder=5)
    ax.set_xticks(range(len(per_acq['species'].unique())))
    ax.set_xticklabels(per_acq['species'].unique())
    ax.set_xlabel('')
    ylabel = var.replace('_', ' ')
    if var == 'frac_below_thresh':
        ylabel = f'frac < {theta_thresh} deg'
    ax.set_ylabel(ylabel)
    # Significance annotation
    p = pvals[var]
    y_min, y_max = ax.get_ylim()
    bar_y = y_max + (y_max - y_min) * 0.05
    ax.plot([0.2, 0.2, 0.8, 0.8], [bar_y, bar_y + (y_max-y_min)*0.03,
            bar_y + (y_max-y_min)*0.03, bar_y], color='0.3', lw=1, clip_on=False)
    ax.text(0.5, bar_y + (y_max-y_min)*0.05, f'{sig_str(p)} (p={p:.3f})',
            ha='center', va='bottom', fontsize=8, color='0.3', clip_on=False)
    ax.set_ylim(y_min, bar_y + (y_max-y_min)*0.18)
    sns.despine(offset=4, ax=ax, bottom=True)
    ax.tick_params(axis='both', which='major', pad=6, size=0)
    ax.set_box_aspect(1)
fig.suptitle(f'Per-acquisition {hist_var}', fontsize=10)
fig.tight_layout()
figname = f'per_acq_{hist_var}_mel-v-yak_min-chase-{min_chase_dur}s'
fig.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')

#%% Per-acquisition: fraction of switches ABOVE theta_thresh
# (is one group switching at larger theta values than the other?)
per_acq['frac_above_thresh'] = 1 - per_acq['frac_below_thresh']

mel_above = per_acq[per_acq['species']=='mel']['frac_above_thresh'].values
yak_above = per_acq[per_acq['species']=='yak']['frac_above_thresh'].values
mw_above_stat, mw_above_p = spstats.mannwhitneyu(mel_above, yak_above, alternative='two-sided')
print(f"\n--- Per-acquisition Mann-Whitney (frac >= {theta_thresh} deg) ---")
print(f"  mel: {mel_above.round(3)}")
print(f"  yak: {yak_above.round(3)}")
print(f"  U={mw_above_stat:.1f}, p={mw_above_p:.4g}")

# Verify: recompute directly from plotd to check consistency
mel_check = np.array([((grp[hist_var] >= theta_thresh).mean())
                      for _, grp in plotd[plotd['species']=='mel'].groupby('acquisition')])
yak_check = np.array([((grp[hist_var] >= theta_thresh).mean())
                      for _, grp in plotd[plotd['species']=='yak'].groupby('acquisition')])
_, p_check = spstats.mannwhitneyu(mel_check, yak_check, alternative='two-sided')
print(f"\n--- Consistency check at {theta_thresh} deg ---")
print(f"  left plot (1-frac_below):  mel={mel_above.round(3)}, yak={yak_above.round(3)}, p={mw_above_p:.4g}")
print(f"  sweep recompute (>=thr):   mel={mel_check.round(3)}, yak={yak_check.round(3)}, p={p_check:.4g}")

# Sweep across thresholds to see at which angle the species diverge
thresholds = np.sort(np.unique(np.concatenate([np.arange(10, 100, 10), [theta_thresh]])))
sweep_pvals = []
sweep_means = {'mel': [], 'yak': []}
sweep_sems = {'mel': [], 'yak': []}
for thr in thresholds:
    mel_f = np.array([((grp[hist_var] >= thr).mean())
                      for _, grp in plotd[plotd['species']=='mel'].groupby('acquisition')])
    yak_f = np.array([((grp[hist_var] >= thr).mean())
                      for _, grp in plotd[plotd['species']=='yak'].groupby('acquisition')])
    _, pv = spstats.mannwhitneyu(mel_f, yak_f, alternative='two-sided')
    sweep_pvals.append(pv)
    sweep_means['mel'].append(np.mean(mel_f))
    sweep_means['yak'].append(np.mean(yak_f))
    sweep_sems['mel'].append(np.std(mel_f) / np.sqrt(len(mel_f)))
    sweep_sems['yak'].append(np.std(yak_f) / np.sqrt(len(yak_f)))
    print(f"  thr={thr:3d}  mel(n={len(mel_f)}): {mel_f.round(3)}  "
          f"yak(n={len(yak_f)}): {yak_f.round(3)}  p={pv:.4f}")

print(f"\nAcquisitions per species in plotd:")
for sp, gdf in plotd.groupby('species'):
    acqs = gdf['acquisition'].unique()
    print(f"  {sp}: {len(acqs)} acquisitions, {len(gdf)} frames")
    for a in acqs:
        print(f"    {a}: {len(gdf[gdf['acquisition']==a])} frames")

fig, axn = plt.subplots(1, 2, figsize=(7, 3)) #(5, 1.5))

# Left: bar + strip for frac >= theta_thresh
ax = axn[0]
for si, sp in enumerate(per_acq['species'].unique()):
    vals = per_acq[per_acq['species']==sp]['frac_above_thresh'].values
    jitter_x = si + np.random.uniform(-0.15, 0.15, size=len(vals))
    ax.bar(si, np.mean(vals), width=0.5, color=species_colors[sp],
           alpha=0.3, edgecolor=species_colors[sp])
    ax.scatter(jitter_x, vals, color=species_colors[sp],
               alpha=0.8, s=10, zorder=5)
ax.set_xticks(range(len(per_acq['species'].unique())))
ax.set_xticklabels(per_acq['species'].unique())
ax.set_ylabel(f'frac >= {theta_thresh} deg')
ax.set_xlabel('')
y_min, y_max = ax.get_ylim()
bar_y = y_max + (y_max - y_min) * 0.05
ax.plot([0.2, 0.2, 0.8, 0.8], [bar_y, bar_y + (y_max-y_min)*0.03,
        bar_y + (y_max-y_min)*0.03, bar_y], color='0.3', lw=1, clip_on=False)
ax.text(0.5, bar_y + (y_max-y_min)*0.05, f'{sig_str(mw_above_p)} (p={mw_above_p:.3f})',
        ha='center', va='bottom', fontsize=8, color='0.3', clip_on=False)
ax.set_ylim(y_min, bar_y + (y_max-y_min)*0.18)
sns.despine(offset=4, ax=ax, bottom=True)
ax.tick_params(axis='both', which='major', pad=6, size=0)
ax.set_box_aspect(1)

# Right: mean frac >= threshold per species, with significance markers
ax = axn[1]
for sp in ['mel', 'yak']:
    means = np.array(sweep_means[sp])
    sems = np.array(sweep_sems[sp])
    ax.plot(thresholds, means, '.-', color=species_colors[sp], label=sp)
    ax.fill_between(thresholds, means - sems, means + sems,
                    color=species_colors[sp], alpha=0.15)
# Annotate p-values at each threshold
y_top = ax.get_ylim()[1]
for ti, (thr, pv) in enumerate(zip(thresholds, sweep_pvals)):
    color = 'k' if pv < 0.05 else '0.6'
    weight = 'bold' if pv < 0.05 else 'normal'
    ax.text(thr, y_top * 0.98, f'{pv:.3f}', ha='center', va='top',
            fontsize=6, color=color, fontweight=weight, rotation=45)
ax.set_xlabel('theta threshold (deg)')
ax.set_ylabel('mean frac >= threshold')
ax.legend(frameon=False, fontsize=8)
sns.despine(offset=4, ax=ax)
ax.tick_params(axis='both', which='major', pad=6, size=0)
ax.set_box_aspect(1)

fig.tight_layout()
figname = f'per_acq_frac_above_thresh_{hist_var}_mel-v-yak_min-chase-{min_chase_dur}s'
fig.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')
fig.savefig(os.path.join(figdir, f'{figname}.pdf'), bbox_inches='tight')

#%%

# Left: bar + strip for frac >= theta_thresh
fig, ax = plt.subplots(figsize=(1, 1))
for si, sp in enumerate(per_acq['species'].unique()):
    vals = per_acq[per_acq['species']==sp]['frac_above_thresh'].values
    jitter_x = si + np.random.uniform(-0.15, 0.15, size=len(vals))
    ax.bar(si, np.mean(vals), width=0.5, color=species_colors[sp],
           alpha=0.3, edgecolor=species_colors[sp])
    ax.scatter(jitter_x, vals, color=species_colors[sp],
               alpha=0.8, s=10, zorder=5)
ax.set_xticks(range(len(per_acq['species'].unique())))
ax.set_xticklabels(per_acq['species'].unique())
ax.set_ylabel(f'frac >= {theta_thresh} deg')
ax.set_xlabel('')
y_min, y_max = ax.get_ylim()
bar_y = y_max + (y_max - y_min) * 0.05
ax.plot([0.2, 0.2, 0.8, 0.8], [bar_y, bar_y + (y_max-y_min)*0.03,
        bar_y + (y_max-y_min)*0.03, bar_y], color='0.3', lw=1, clip_on=False)
ax.text(0.5, bar_y + (y_max-y_min)*0.05, f'{sig_str(mw_above_p)} (p={mw_above_p:.3f})',
        ha='center', va='bottom', fontsize=8, color='0.3', clip_on=False)
ax.set_ylim(y_min, bar_y + (y_max-y_min)*0.18)
ax.set_yticks(np.linspace(0, 0.8, 3))
sns.despine(offset=4, ax=ax, bottom=True)
ax.tick_params(axis='both', which='major', pad=6, size=0)
ax.set_box_aspect(2)
figname = f'bar_per_acq_frac_above_thresh_{hist_var}_mel-v-yak_min-chase-{min_chase_dur}s_direct-facing-angle-{direct_facing_angle}'
fig.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')
fig.savefig(os.path.join(figdir, f'{figname}.pdf'), bbox_inches='tight')

print(f"\n--- Per-acquisition frac >= {theta_thresh} deg (bar+strip plot) ---")
print(f"  mel (n={len(mel_above)}): mean={np.mean(mel_above):.3f}, median={np.median(mel_above):.3f}, "
      f"IQR=[{np.percentile(mel_above,25):.3f}, {np.percentile(mel_above,75):.3f}]")
print(f"  yak (n={len(yak_above)}): mean={np.mean(yak_above):.3f}, median={np.median(yak_above):.3f}, "
      f"IQR=[{np.percentile(yak_above,25):.3f}, {np.percentile(yak_above,75):.3f}]")
print(f"  Mann-Whitney U={mw_above_stat:.1f}, p={mw_above_p:.4g} ({sig_str(mw_above_p)})")
print(f"  Interpretation: fraction of switches occurring at theta_error >= {theta_thresh} deg "
      f"({'differs' if mw_above_p < 0.05 else 'does not differ'} between mel and yak)")

#%%
# Plot binned theta error:
# =====================================
binned_var = 'theta_error_deg_abs'
bin_size = 20
max_dist = 20
start_bin = 0 #-180
end_bin = 180
bin_size = 20
plotd['binned_theta_error'] = pd.cut(plotd[binned_var],
                                bins=np.arange(start_bin, end_bin, bin_size),
                                labels=np.arange(start_bin+bin_size/2,
                                                    end_bin-bin_size/2, bin_size))    
# Get average ang vel across bins, use circular mean for angular variable
#avg_theta_error = all_f1.groupby(['species', 'acquisition', 'binned_theta_error', 
#                        ])['theta_error_deg'].apply(lambda x: spstats.circmean(x, high=np.pi, low=-np.pi, nan_policy='omit')).reset_index()

fig, ax = plt.subplots(figsize=(5, 4))
# Histogram, counts per binned theta error
counts = plotd.groupby(['species', 'binned_theta_error', 'acquisition']).size().reset_index(name='count')
counts.rename(columns={'binned_theta_error': 'theta_error_deg'}, inplace=True)

sns.pointplot(data=counts, ax=ax,
            x='theta_error_deg', y='count',
            hue='species', palette=species_colors, 
            errorbar='se')#, marker='o') #errwidth=0.5)
#ax.set_xticks(np.linspace(start_bin, end_bin, 7))

#%%
# Count N switches per acquisition and species from dict
#n_switches = dict()
n_ = []
for i, (acq, switch_positions) in enumerate(switches.items()):
    curr_species = plotd[plotd['acquisition']==acq]['species'].unique()[0]
    df_ = pd.DataFrame(data={'acquisition': acq, 
            'n_switches': len(switch_positions),
            'species': curr_species}, index=[i])
    n_.append(df_)
switch_counts = pd.concat(n_)
print(switch_counts)

fig, ax = plt.subplots(figsize=(3,3))
sns.barplot(data=switch_counts, ax=ax,
        x='species', y='n_switches', hue='species', 
        palette=species_colors, dodge=False, 
        fill=False, errorbar='se')
# plot spread
sns.stripplot(data=switch_counts, ax=ax,
        x='species', y='n_switches', hue='species', 
        palette=species_colors, jitter=True, alpha=0.8,
        legend=False, s=10)
ax.set_ylabel('N switches')
ax.set_xlabel('')
sns.despine(offset=2, ax=ax, bottom=True)
ax.tick_params(axis='both', which='major', pad=6, size=0)
ax.set_box_aspect(1)

ax.set_title('N switches per acquisition and species')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
ax.set_box_aspect(1)
sns.despine(offset=2, ax=ax, bottom=True)
ax.tick_params(axis='both', which='major', pad=6, size=0)

# Custom legend
legh = putil.custom_legend(['mel', 'yak'], [species_colors['mel'], species_colors['yak']])
ax.legend(handles=legh, loc='upper left', bbox_to_anchor=(1, 1), 
        frameon=False)
# do stats:
res = spstats.mannwhitneyu(switch_counts[switch_counts['species']=='mel']['n_switches'],
                           switch_counts[switch_counts['species']=='yak']['n_switches'])
print(res)
figname = f'switch_counts_min-chase-{min_chase_dur}s_direct-facing-angle-{direct_facing_angle}'
fig.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')
# %%
