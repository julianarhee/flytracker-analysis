#!/usr/bin/env 
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:10:00 2020

@author: julianarhee
@description: import DLC tracks and do calculations
"""
#%%
import os
import glob
import yaml
import re

import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl

import matplotlib as mpl

import libs.plotting as putil
import libs.dlc as dlc
import libs.utils as util
import cv2
import importlib

import transform_data.relative_metrics as rem
import analyses.dlc.src.plot_dlc_frames as pdlc

plot_style='white'
putil.set_sns_style(style=plot_style)
bg_color='w' if plot_style=='dark' else 'k'

#%% functions

#def get_acq_from_dlc_fpath(fpath):
#    return '_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])

#%%
rootdir = '/Users/julianarhee/DeepLabCut'
projectname = 'projector-1dot-jyr-2024-02-18'
#%% Look at 1 data file
analyzed_dir = pdlc.get_dlc_analysis_dir(projectname=projectname)
acq_prefix = '20240214-1025_f1_*sz10x10'
#acq_prefix = '20231121*fly2*113942'
fpath = pdlc.get_fpath_from_acq_prefix(analyzed_dir, acq_prefix)
acq = dlc.get_acq_from_dlc_fpath(fpath) #'_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])
print(acq)
video_fname = os.path.split(fpath.split('DLC')[0])[-1]
print(video_fname)
# load dataframe
df0 = pd.read_hdf(fpath)
scorer = df0.columns.get_level_values(0)[0]

# get video info
#cap = pdlc.get_video_cap_tmp(acq)
minerva_base = '/Volumes/Julie/2d-projector'
vids = glob.glob(os.path.join(minerva_base, '20*', '{}*.avi'.format(acq_prefix)))
video_fpath = vids[0]
cap = cv2.VideoCapture(video_fpath)
print(video_fpath)
# get video info
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)

# load config file
cfg = pdlc.load_dlc_config(projectname=projectname)

# get fig id 
fig_id = os.path.split(fpath.split('DLC')[0])[-1]

#%%
importlib.reload(dlc)
# %% DLC
flyid = 'fly' # double check in the plots for abdomen lengths
dotid = 'single'
fps = 60  # Hz
max_jump = 6
pcutoff=0.8

# load _eh.h5
#trk = pd.read_hdf(fpath) #os.path.join(pathname, filename))
#scorer = trk.columns.get_level_values(0)[0]
#tstamp = np.linspace(0, len(trk) * 1 / fps, len(trk))
#nframes = len(trk)

# get dataframes
flydf = dlc.load_trk_df(fpath, flyid='fly', fps=fps, 
                        max_jump=max_jump, cop_ix=None,
                        filter_bad_frames=True, pcutoff=pcutoff)
dotdf = dlc.load_trk_df(fpath, flyid='single', fps=fps, 
                        max_jump=max_jump, cop_ix=None, 
                        filter_bad_frames=False, pcutoff=pcutoff)
# set nans
nan_rows = flydf.isna().any(axis=1)
dotdf.loc[nan_rows] = np.nan

flydf, dotdf = dlc.get_interfly_params(flydf, dotdf, cop_ix=None)

# add speed info
#dotdf, flydf = dlc.add_speed_epochs(dotdf, flydf, video_fname)


#%% process for relative metrics
# assign IDs like FlyTracker DFs
flydf['id'] = 0
dotdf['id'] = 1
trk_ = pd.concat([flydf, dotdf], axis=0)

mm_per_pix = 3 / trk_['body_length'].mean()
trk_['vel'] = trk_['lin_speed'] * mm_per_pix

# %% rename columns to get RELATIVE pos info
trk_ = trk_.rename(columns={'centroid_x': 'pos_x',
                           'centroid_y': 'pos_y',
                           'heading': 'ori', # should rename this
                           'body_length': 'major_axis_len',
                           'inter_wing_dist': 'minor_axis_len',
                           'time': 'sec'
                           }) 
#%% do transformations
importlib.reload(rem)
#trk_ = util.center_coordinates(trk_, frame_width, frame_height) 
df = rem.do_transformations_on_df(trk_, frame_width, frame_height) #, fps=fps)
df['ori_deg'] = np.rad2deg(df['ori'])

df['targ_pos_theta'] = -1*df['targ_pos_theta']

#%%
import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate2(p, angle, origin=(0, 0)): #degrees=0):
    #angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

importlib.reload(util)
#%% center x- and y-coordinates
trk_ = util.center_coordinates(trk_, frame_width, frame_height) 

# separate fly1 and fly2
flyid1=0
flyid2=1
fly1 = trk_[trk_['id']==flyid1].copy().reset_index(drop=True)
fly2 = trk_[trk_['id']==flyid2].copy().reset_index(drop=True)

# translate coordinates so that focal fly is at origin
fly1, fly2 = util.translate_coordinates_to_focal_fly(fly1, fly2)

# rotate coordinates so that fly1 is facing 0 degrees (East)
fly1, fly2 = util.rotate_coordinates_to_focal_fly(fly1, fly2)

polarcoords = util.cart2pol(fly2['rot_x'], fly2['rot_y']) 
fly1['targ_pos_radius'] = polarcoords[0]
fly1['targ_pos_theta'] = polarcoords[1]
fly2['targ_pos_radius'] = polarcoords[0]
fly2['targ_pos_theta'] = polarcoords[1]

fly1['targ_rel_pos_x'] = fly2['rot_x']
fly1['targ_rel_pos_y'] = fly2['rot_y']
fly2['targ_rel_pos_x'] = fly2['rot_x']
fly2['targ_rel_pos_y'] = fly2['rot_y']


#%%

ix = 2262 #7764 #2262 # 7064 #7764 # 2262
flypos = df0.xs(flyid, level='individuals', axis=1)
flypos = dlc.remove_jumps(flypos, max_jump)
bad_ixs = flypos[ flypos[ flypos[flypos.columns[flypos.columns.get_level_values(2)=='likelihood']] < pcutoff].any(axis=1)].index.tolist()
flypos.loc[bad_ixs, :] = np.nan

bpt1 = flypos.xs('abdomentip', level='bodyparts', axis=1).to_numpy() #copy()
bpt2 = flypos.xs('head', level='bodyparts', axis=1).to_numpy() #.copy()

ys = bpt2[:, 1] - bpt1[:, 1]
xs =bpt2[:, 0], bpt1[:, 0] 
angs = np.arctan2(ys, xs)

x1 = bpt1[ix]
x2 = bpt2[ix]
print(x1, x2)

print(np.arctan2(x2[1]-x1[1], x2[0]-x1[0]))


#%%
importlib.reload(dlc)
#flydf0 = dlc.get_fly_params(flypos, cop_ix=None)


flydf = pd.DataFrame(dlc.get_bodypart_angle(flypos, 'abdomentip', 'head'),
                                columns=['ori'])
print(flydf.loc[ix])
print(fly1.loc[ix]['ori'])

#%%
importlib.reload(pdlc)
#ix = 7064 #7764 # 2262
# 7064 #2262 #5585 # 7764 #2262 #7062 #2262

# load dataframe
df0 = pd.read_hdf(fpath)
# load config file
cfg = pdlc.load_dlc_config(projectname=projectname)

fig = pdlc.plot_skeleton_on_image([ix, ix+1], df0, cap, cfg, 
                            pcutoff=0.01, animal_colors={'fly': 'm', 'single': 'c'})

#%% TRANSFORMS
id_colors = ['r', 'b']
old = False#False #True

cap.set(1, ix)
ret, im = cap.read()
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

fig = pl.figure(figsize=(12,5)) #pl.subplots(1, 2)
ax = fig.add_subplot(131)
ax.imshow(im, cmap='gray')
ax.invert_yaxis()

for i, d_ in trk_.groupby('id'):
    print('pos:', i, d_[d_['frame']==ix]['pos_x'], d_[d_['frame']==ix]['pos_y'])
    ax.plot(d_.iloc[ix]['pos_x'], d_.iloc[ix]['pos_y'], 
            marker='o', color=id_colors[i], markersize=3)

#ax=fig.add_subplot(152) #projection='polar')
#for i, d_ in trk_.groupby('id'):
#    print('ctr:', i, d_[d_['frame']==ix]['ctr_x'], d_[d_['frame']==ix]['ctr_y'])
#    ax.plot(d_.iloc[ix]['ctr_x'], d_.iloc[ix]['ctr_y'], 
#            marker='o', color=id_colors[i], markersize=3)
##ax.invert_yaxis()
#ax.set_aspect(1)
#ax.set_title('center frame')
#
#ax = fig.add_subplot(153)
#for i, d_ in enumerate([fly1, fly2]):
#    print('trans:', i, d_.loc[ix]['trans_x'], d_.loc[ix]['trans_y'])
#    ax.plot(d_.loc[ix]['trans_x'], d_.loc[ix]['trans_y'], 
#            marker='o', color=id_colors[i], markersize=3)
##ax.invert_yaxis()
#ax.set_aspect(1)
#ax.set_title('trans')

ax = fig.add_subplot(132) #, projection='polar')
for i, d_ in enumerate([fly1, fly2]):
    #print('rot:', i, d_.iloc[ix]['rot_x'], d_.iloc[ix]['rot_y'])
    if old:
        ax.plot(d_.loc[ix]['rot_x'], d_.loc[ix]['rot_y'], 
           marker='o', color=id_colors[i], markersize=3)
    else:
        pt = np.array([d_.iloc[ix]['trans_x'], d_.iloc[ix]['trans_y']])
        #ang = rotation_angs[ix]        
        #rx, ry = rotate([0, 0], pt, ang)
        ang = -1*fly1.loc[ix]['ori'] 
        rotmat = np.array([[np.cos(ang), -np.sin(ang)],
                          [np.sin(ang), np.cos(ang)]])
        #rx, ry = (rotmat @ pt.T).T
        rx, ry = rotate2(pt, ang) #[0, 0], pt, ang)
        print('rot:', i, rx, ry)
        ax.plot(rx, ry,marker='o', color=id_colors[i], markersize=3)
ax.set_aspect(1)
ax.set_title('rot')

# POLAR
ax = fig.add_subplot(133, projection='polar')
for i, d_ in enumerate([fly1, fly2]):
    print('polar:', i, d_.iloc[ix]['targ_pos_theta'], d_.iloc[ix]['targ_pos_radius'])
    if i==0:
        ax.plot(0, 0, 'r*')
    if old:
        ax.plot(d_.iloc[ix]['targ_pos_theta'], d_.iloc[ix]['targ_pos_radius'], 
            marker='o', color=id_colors[i], markersize=3)
    else:
        #ang = fly1.iloc[ix]['ori'] #* -1
        pt = [d_.loc[ix]['trans_x'], d_.loc[ix]['trans_y']]
        #ang = rotation_angs[ix]  
        #rx, ry = rotate((0,0), pt, ang)      
        #rx, ry = rotate2(pt, ang) #[0, 0], pt, ang)
        rad, th = util.cart2pol(rx, ry)
        ax.plot(th, rad, marker='o', color=id_colors[i], markersize=3)
ax.set_aspect(1)
ax.set_title('polar')
#ax.set_theta_direction(-1)

fig.suptitle('{}, ang={:.2f}'.format(ix, np.rad2deg(ang)))

#%% OLD WAY
importlib.reload(rem)
fig = rem.plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)

#%%