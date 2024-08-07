#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import glob
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl

import matplotlib as mpl

import plotting as putil
import dlc as dlc
import utils as util
import cv2
import importlib

import parallel_pursuit as pp

import relative_metrics as rem

plot_style='dark'
putil.set_sns_style(style=plot_style, min_fontsize=12)
bg_color='w' if plot_style=='dark' else 'k'
#%%


# %%
#srcdir = '/Volumes/Julie/2d-projector-analysis/processed'
srcdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/JAABA'

#% Load jaaba-traansformed data
jaaba_fpath = os.path.join(srcdir, 'jaaba_transformed_data_transf.pkl')
assert os.path.exists(jaaba_fpath), "File not found: {}".format(jaaba_fpath)
jaaba = pd.read_pickle(jaaba_fpath)   

#%% Load feat/trk combined data
ft_fpath = os.path.join(srcdir, 'flytracker.pkl')
assert os.path.exists(ft_fpath), "File not found: {}".format(ft_fpath)
ft = pd.read_pickle(ft_fpath)

#%% COMBINE JAABA AND FLYTRACKER DATA
jaa_cols = [c for c in jaaba.columns if c in ft.columns]

ftjaaba = pd.merge(ft[ft['id']==0], jaaba, on=jaa_cols, how='left')
ftjaaba = ftjaaba.reset_index(drop=True)

# replace spaces in column names with underscores
ftjaaba.columns = [c.replace(' ', '_') for c in ftjaaba.columns]
ft.columns = [c.replace(' ', '_') for c in ft.columns]

# %% 
all_fnames = jaaba['filename'].unique()


#%% -------------------------------------
# LOOK AT SINGLE FILE
# --------------------------------------
fnames = [f for f in all_fnames if '20240222' in f 
          and 'fly7_Dmel' in f
          and '8x8' in f]
fnames = [f for f in all_fnames if '20240216' in f 
          and 'fly3_Dmel' in f
          and '4x4' in f]

assert len(fnames)>0, "No files found"
print(fnames)
fname = fnames[0]

#%% get video info for giacomo jaaba data
acq = fname
minerva_base2 = '/Volumes/Giacomo/JAABA_classifiers/projector/changing_dot_size_speed'
found_vids = glob.glob(os.path.join(minerva_base2, '{}*'.format(acq), 'movie.avi'))
assert len(found_vids)>0, "No video found for acq {}".format(acq)
video_fpath = found_vids[0]
#cap = pdlc.get_video_cap_tmp(acq)
cap = cv2.VideoCapture(video_fpath) 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)

#%% Get FlyTracker info (TRK/FEAT)
fps = 60.
trk_ = ft[ft['filename']==fname].copy().reset_index(drop=True)
trk_['sec'] = trk_['frame'] / fps
print(trk_['id'].unique())

#%% Do transformations to egocentric -- need BOTH M/F
importlib.reload(rem)
# do transformations
#nan_ix = trk_[trk_.isnull()].index.tolist()

# if using flytracker, flip ORI:
#trk_['pos_y'] = frame_height - trk_['pos_y']
trk_['ori'] = -1*trk_['ori']
df = rem.do_transformations_on_df(trk_, frame_width, frame_height)
#df.loc[nan_ix, :] = np.nan
#%% CHECK TRANSFORMATIONS

fly1 = df[df['id']==0].copy().reset_index(drop=True)
fly2 = df[df['id']==1].copy().reset_index(drop=True)
ix = 9400
fig = rem.plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)

#%
importlib.reload(rem)
fig = rem.check_rotation_transform(ix, df, cap) # frame_width, frame_height)

#%%
#flydf = df[df['id']==0].copy()

jaa_ = jaaba[jaaba['filename']==fname].copy().reset_index(drop=True)    
jaa_.shape, fly1.shape

merge_cols = [c for c in jaa_.columns if c not in fly1.columns]
flydf = pd.concat([fly1, jaa_[merge_cols]], axis=1) #pd.merge(flydf, jaa_, on=merge_cols, how='left')

#%%
#df = smooth_and_calculate_velocity_circvar(df, smooth_var='ori', 
#                                vel_var='ang_vel', time_var='sec')

#%% -------------------------------------
# THETA ERRORS
# ---------------------------------------
stim_hz_vals = jaaba['stim_hz'].unique()
stimhz_palette = dict((k, v) for k, v in zip(stim_hz_vals, 
                        sns.color_palette('viridis', n_colors=len(stim_hz_vals))))

#%% Filter BOUTS
min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
max_facing_angle = np.deg2rad(45)
min_vel = 5
max_dist_to_other = 20
min_dist_to_other = 3

#min_ang_vel = 0.04
#max_ang_vel = 24.8

passdf = flydf[ (flydf['stim_hz']>0) 
               #& (flydf['facing_angle']<=max_facing_angle)
               #& (flydf['targ_pos_theta']>=min_pos_theta) 
               #& (flydf['targ_pos_theta']<=max_pos_theta)
               #& (flydf['vel']>=min_vel)
               #& (flydf['dist_to_other'] <= max_dist_to_other)
               #& (flydf['dist_to_other'] >= min_dist_to_other)
               #& (flydf['ang_vel']<=max_ang_vel)
               #& (flydf['ang_vel']>=min_ang_vel)
               & (flydf['chasing']>20)
               ]
#%
# get start/stop indices of consecutive rows
consec_bouts = pp.get_indices_of_consecutive_rows(passdf)

# filter duration?
min_bout_len = 0.25
fps = 60.
incl_bouts = pp.filter_bouts_by_frame_duration(consec_bouts, min_bout_len, fps)
print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(consec_bouts), min_bout_len))

# find turn starts
turn_start_frames = [c[0] for c in incl_bouts]
print(len(turn_start_frames))

# PLOT -------
nsec_pre = 10
d_list = []
for ix in turn_start_frames: #[0::2]: #0::20]:
    start_ = ix #nsec_pre*fps
    stop_ = ix + nsec_pre*fps
    d_list.append(passdf.loc[start_:stop_])
plotted_passdf = pd.concat(d_list)

fig, axn = pl.subplots(1, 2,#figsize=(10,5), sharex=True, sharey=True,
                       subplot_kw={'projection': 'polar'})
pp.plot_allo_vs_egocentric_pos(plotted_passdf, axn, huevar='stim_hz', cmap='viridis',
                        palette_dict=stimhz_palette, 
                        plot_com=True, bg_color=bg_color) 
pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)

putil.label_figure(fig, acq)
pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)

putil.label_figure(fig, acq)
# save?


#%% ONE FLY: Does targ_pos_theta increase with increasing stim. vel?
#plotdf = flydf[flydf['chasing']>20]

fig, ax =pl.subplots()
sns.pointplot(data=plotted_passdf , x='stim_hz', y='targ_pos_theta', ax=ax,
              hue='stim_size', palette='viridis') #legend=0)
ax.legend_.remove()
ax.set_box_aspect(1)
putil.label_figure(fig, acq)


#%% ONE FLY: Does fly ang vel increase with increasing targ_pos_theta?
fig, ax = pl.subplots()
plotted_passdf['targ_pos_theta_binned'] = pd.cut(plotted_passdf['targ_pos_theta'], bins=10) 

sns.pointplot(data=plotted_passdf , ax=ax, 
                x='targ_pos_theta_binned', y='fly_ang_vel',
                hue='stim_hz', palette=stimhz_palette) 
                #edgecolor='none', s=10, legend=0, alpha=0.7)

#%% ----------------------------
# Does angular vel. INCREASE with stim_hz? with theta-error?
# ----------------------------
flydf['fly_ang_acc'] = flydf['fly_ang_vel'].diff() / flydf['sec_diff'].mean()
flydf['targ_ang_vel'] = flydf['targ_pos_theta'].diff() / flydf['sec_diff'].mean()
flydf['targ_ang_acc'] = flydf['targ_ang_vel'].diff() / flydf['sec_diff'].mean()

#%%
# Look at moments of high ang vel.
min_ang_vel = 3
passdf = flydf[ (flydf['stim_hz']>0) 
               & (flydf['fly_ang_vel']>=min_ang_vel)
               & (flydf['chasing']>20)
               ]

# get start/stop indices of consecutive rows
high_ang_vel_bouts = pp.get_indices_of_consecutive_rows(passdf)
print(len(high_ang_vel_bouts))
# filter duration?
min_bout_len = 0.25
fps = 60.
incl_bouts = pp.filter_bouts_by_frame_duration(high_ang_vel_bouts, min_bout_len, fps)
print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), 
                                        len(high_ang_vel_bouts), min_bout_len))

# find turn starts
turn_start_frames = [c[0] for c in incl_bouts] #high_ang_vel_bouts]

#%%
nframes_win = 2*fps
start_ix = turn_start_frames[1] #- nframes_win
stop_ix = start_ix + nframes_win #*2

plotdf = flydf.loc[start_ix:stop_ix]
# 
ang_peaks = plotdf[plotdf['fly_ang_acc']>200]
high_ang_bouts = pp.get_indices_of_consecutive_rows(ang_peaks)

# find turn starts
high_ang_start_frames = [c[0] for c in high_ang_bouts] #onsec_bouts]
print(len(high_ang_start_frames))

#%%
targ_color = 'r'
fly_color = 'cornflowerblue'
accel_color = [0.6]*3

fig, axn = pl.subplots(3, 1, figsize=(8,6), sharex=True)

ax=axn[0]
ax.plot(plotdf['targ_pos_theta'], targ_color)
# color y-axis spine and ticks red
ax.spines['left'].set_color(targ_color)
ax.set_ylabel('targ_pos_theta', color=targ_color)

ax2 = ax.twinx()
ax2.plot(plotdf['fly_ang_vel'], fly_color)
# color y-axis spine and ticks blue
ax2.spines['left'].set_color(fly_color)
ax2.set_ylabel('fly_ang_vel', color=fly_color)

ax=axn[1]
ax.plot(plotdf['targ_ang_vel'], targ_color)
ax.set_ylabel('targ_ang_vel', color=targ_color)
ax2 = ax.twinx()
ax2.plot(plotdf['fly_ang_vel'], fly_color)
ax2.set_ylabel('fly_ang_vel', color=fly_color)

ax=axn[2]
ax.plot(plotdf['fly_ang_acc'], fly_color)
ax.set_ylabel('fly_ang_acc', color=fly_color)
ax2 = ax.twinx()
ax2.plot(plotdf['targ_ang_acc'], targ_color)
ax2.set_ylabel('targ_ang_acc', color=targ_color)

for i in high_ang_start_frames:
    ax.plot(i, flydf.loc[i, 'fly_ang_acc'], bg_color, marker='o', markersize=5)

axn[0].set_title('{}, frames: {}-{}'.format(acq, start_ix, stop_ix), loc='left', 
                 fontsize=8)


nframes_win = 0.25*fps
ixs = []
fig, ax = pl.subplots()
for ix in high_ang_start_frames:
    start_ = ix - nframes_win
    stop_ = ix + nframes_win
    plotdf.loc[start_:stop_, 'tmp_ix'] = np.arange(0, plotdf.loc[start_:stop_].shape[0])
    sns.lineplot(data=plotdf.loc[start_:stop_], x='tmp_ix', y='targ_pos_theta', ax=ax,
                 color=accel_color, lw=1, alpha=0.75)
    print(ix)
    ixs.append(plotdf.loc[start_:stop_].index.tolist())
ax.axvline(x=nframes_win, color=bg_color, linestyle='--')

#%%


#%% ------------------------------
# MAKE VIDEO OF BOUT
# --------------------------------
#%% make video of bout?
#tmp_outdir = '/Users/julianarhee/Documents/rutalab/projects/predictive_coding'
from matplotlib.animation import FuncAnimation
from matplotlib import animation

curr_frames = pd.Series(flydf.loc[start_ix:stop_ix]['frame'].values)

f1 = flydf.loc[start_ix:stop_ix]
f2 = fly2.loc[start_ix:stop_ix]

vmin, vmax = f1[huevar].min(), 10 #f1[huevar].max()
#frame_num = curr_frames.values[0]
ix = 50
# get list of colors for scatter plot based on angular size using viridis cmap and normalizing to range of angular size values
huevar = 'ang_vel' #'targ_ang_size'
#vmin, vmax = f1[huevar].min(), f1[huevar].max()
#hue_norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
print(vmin, vmax)
ylim = np.ceil(f1['targ_pos_radius'].max()) + 50
# Create a Normalize object to map the data values to the range [0, 1]
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) #f1[hue_var].min(), 
                            #vmax=0.3) #f1[hue_var].max())
# Create a ScalarMappable object with the Viridis colormap and the defined normalization
sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)

# Get a list of colors corresponding to the data values
colors = [sm.to_rgba(value) for value in f1[huevar].values]

fig = pl.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='polar')
ax2.set_ylim([0, ylim])

#while im is None:
#for ix in range(len(curr_frames)):
cap.set(1, curr_frames.iloc[ix])
ret, im = cap.read()
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(ix)

p1 = ax1.imshow(im, aspect='equal', cmap='gray')
ax1.invert_yaxis()
p2 = ax2.scatter(f1.iloc[:ix]['targ_pos_theta'],
                    f1.iloc[:ix]['targ_pos_radius'],
                    c=f1.iloc[:ix][huevar],                       
                    vmin=vmin, vmax=vmax,
                    s=20, #f1.iloc[:ix]['targ_ang_size'],
                    cmap='viridis' #f1.loc[:frame_num]['targ_ang_size'],
                    )
#%
def init():
    p1.set_data(np.zeros((frame_height, frame_width)))
    # set x, y data for scatter plot
    p2.set_data([], [])
    # set sizes
    #p2.set_sizes([], [])
    p2.set_array([])
    return (p1, p2,)

def update_figure(ix): #, p1, p2, cap, f1):        
    # udpate imshow
    cap.set(1, curr_frames.iloc[ix])
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)
    p1.set_data(im)
    #p1.set_title('Frame {}'.format(frame_num))
    
    # Update subplot 2 (plot)
    xys_ = f1.iloc[:ix][['targ_pos_theta', 'targ_pos_radius']].values
    # update x, y
    p2.set_offsets(xys_)
    # uppdate size
    #p2.set_sizes(f1.iloc[:ix]['targ_ang_size'] * 100) 
    # update color
    p2.set_array(f1.iloc[:ix][huevar])

    # Adjust layou+ 1t
    #pl.tight_layout()

frame_numbers = np.arange(0, len(curr_frames)) #curr_frames.values #range(1, 11)  # List of frame numbers

# Create the animation
anim = FuncAnimation(fig, update_figure, frames=frame_numbers, 
                    interval=1000//fps)

# Save the animation as a movie file
video_outrate = 30
figdir = '/Users/julianarhee/Documents/rutalab/projects/predictive_coding'
save_movpath = os.path.join(figdir, 'anim_{}_frames-{}-{}.mp4'.format(acq, start_ix, stop_ix))

#                    'anim_frames-{}-{}_nsec-pre-{}_hue-{}_{}hz.mp4'.format(b_[0], b_[1], nsec_pre,
#                                                                                huevar, video_outrate))

anim.save(save_movpath, fps=video_outrate, extra_args=['-vcodec', 'libx264'])





#%%
nsec_pre = 10

stim_hz_vals = jaaba['stim_hz'].unique()
stimhz_palette = putil.get_palette_dict(jaaba, 'stim_hz', 'viridis')
#stimhz_palette = dict((k, v) for k, v in zip(stim_hz_vals, 
#                        sns.color_palette('viridis', n_colors=len(stim_hz_vals))))
d_list = []
for ix in turn_start_frames: #[0::2]: #0::20]:
    start_ = ix #nsec_pre*fps
    stop_ = ix + nsec_pre*fps
    d_list.append(passdf.loc[start_:stop_])
plotted_passdf = pd.concat(d_list)

fig, axn = pl.subplots(1, 2,#figsize=(10,5), sharex=True, sharey=True,
                       subplot_kw={'projection': 'polar'})
pp.plot_allo_vs_egocentric_pos(plotted_passdf, axn, huevar='stim_hz', cmap='viridis',
                        palette_dict=stimhz_palette, 
                        plot_com=True, bg_color=bg_color) 
pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)

putil.label_figure(fig, acq)
# save?




# %% -------------------------------------
# AGGREGATE ACROSS ALL FILES
# ----------------------------------------
#%% Do transformations to egocentric -- need BOTH M/F
create_new=False
if create_new:
    d_list = []
    for fn, trk_ in ft.groupby('filename'):
        # get video dims
        found_vids = glob.glob(os.path.join(minerva_base2, '{}*'.format(fn), 'movie.avi'))
        assert len(found_vids)==1, "No video found for acq {}".format(fn)
        video_fpath = found_vids[0]
        cap = cv2.VideoCapture(video_fpath) 
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        trk_['sec'] = trk_['frame'] / fps
        # if using flytracker, flip ORI:
        trk_['ori'] = -1*trk_['ori']
        df_ = rem.do_transformations_on_df(trk_, frame_width, frame_height)

        # get jaaba
        fly1 = df_[df_['id']==0]
        jaa_ = jaaba[jaaba['filename']==fn].copy().reset_index(drop=True)   
        assert jaa_.shape[0]>0, "No jaaba data found for {}".format(fn) 
        assert jaa_.shape[0] == fly1.shape[0], "jaaba and flytracker have different lengths"

        # combine ft and jaaba    
        merge_cols = [c for c in jaa_.columns if c not in fly1.columns]
        fly_ = pd.concat([fly1, jaa_[merge_cols]], axis=1) #pd.merge(flydf, jaa_, on=merge_cols, how='left')

        d_list.append(fly_)
    transft = pd.concat(d_list)


# %% targ_pos_theta as a function of stimulus Hz, split by SIZE and SPECIES

#sp = 'mel-sP1>CsChrimson'
plotdf = transft[(transft['stim_hz']>0.025)
                 & (transft['stim_hz']<1)
                 & (transft['chasing']>20)].copy()
plotdf['facing_angle_deg'] = np.rad2deg(plotdf['facing_angle'])
fig, ax = pl.subplots()
sns.lineplot(data=plotdf, x='stim_hz', y='targ_pos_theta', ax=ax,
              hue='stim_size', palette='colorblind', style='species')
ax.set_box_aspect(1)
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)
#sns.scatterplot(data=flydf, x='epoch', 
#                y='facing_angle', ax=ax) #palette='viridis')   


# %% For each stimsize, plot fly's ang_vel as a function of targ_pos_theta

curr_stim_levels = plotdf['stim_hz'].unique()

df_ = plotdf[plotdf['species']=='yak-sP1>CsChrimson']

fig, axn = pl.subplots(len(curr_stim_levels), 1, 
                       sharex=True, sharey=True, figsize=(4, 10))
for i, (stim, d_) in enumerate(df_.groupby('stim_hz')):
    ax=axn[i]
    sns.scatterplot(data=d_, ax=ax,
                x='targ_pos_theta', y='fly_ang_vel',
                hue='stim_hz', palette=stimhz_palette, 
                edgecolor='none', s=3, legend=0, alpha=0.5)

#%% Bin targ pos theta

# bin targ_pos_theta
nbins=15
#bins = np.arange(0, nbins)
transft['targ_pos_theta_binned'] = pd.cut(transft['targ_pos_theta'], bins=nbins)
transft['targ_pos_theta_leftbin'] = [v.left if isinstance(v, pd.Interval) else v for v in transft['targ_pos_theta_binned']]

# %% plot fly ang vel as a function of BINNED targ pos theta 

plotdf = transft[(transft['stim_hz']>0.025)
                 & (transft['stim_hz']<1)
                #(transft['species']==sp)
                 & (transft['chasing']>20)].copy()

fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=true, sharey=true)
for i, (sp, df_) in enumerate(plotdf.groupby('species')):
    ax=axn[i]
    sns.pointplot(data=df_, ax=ax,
                    x='targ_pos_theta_leftbin', y='fly_ang_vel',
                    hue='stim_hz', palette=stimhz_palette) # , legend=0)
                    #edgecolor='none', s=3, legend=0, alpha=0.5)
    ax.set_title(sp, loc='left')
    if i==0:
        ax.legend_.remove()
    else:
        sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=false)
    

#%% HISTOGRAM: targ_pos_theta for each stim_hz

fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
for i, (sp, df_) in enumerate(plotdf.groupby('species')):
    ax=axn[i]
    ax.set_title(sp, loc='left')
    sns.histplot(data=plotdf, x='targ_pos_theta', hue='stim_hz', 
             palette=stimhz_palette, ax=ax,
             stat='probability', common_norm=False, 
            cumulative=False, element='step', fill=False)  #kind='step')
    if i==0:
        ax.legend_.remove()
    else:
        sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)
   
   # ax.set_aspect(1)
