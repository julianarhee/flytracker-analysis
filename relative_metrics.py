#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:47:00 2020
@author: julianarhee
@email: juliana.rhee@gmail.com  
"""
#%%
import os
import glob
import cv2
import numpy as np
import pandas as pd
import pylab as pl  
import seaborn as sns
import utils as util
import matplotlib as mpl
import pickle as pkl
import argparse

#%%
def plot_frame_check_affines(ix, fly1, fly2, cap, frame_width=None, frame_height=None):
    '''
    Plot frame and rotations with markers oriented to fly's heading. IX is FRAME NUMBER.

    Arguments:
        ix -- _description_
        fly1 -- _description_
        fly2 -- _description_
        cap -- _description_

    Keyword Arguments:
        frame_width -- _description_ (default: {None})
        frame_height -- _description_ (default: {None})

    Returns:
        _description_
    '''
    if frame_width is None:
        frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # set fly oris as arrows
    fly_marker = '$\u2192$' # https://en.wikipedia.org/wiki/Template:Unicode_chart_Arrows
    m_ori = np.rad2deg(fly1[fly1['frame']==ix]['rot_ori'])
    f_ori = np.rad2deg(fly2[fly2['frame']==ix]['rot_ori'])
    marker_m = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_m._transform = marker_m.get_transform().rotate_deg(m_ori)
    marker_f = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_f._transform = marker_f.get_transform().rotate_deg(f_ori)
    #print(np.rad2deg(fly1.loc[ix]['ori'])) #m_ori)
    #print(f_ori)

    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    fig = pl.figure(figsize=(8,4))
    ax = fig.add_subplot(121) # axn = pl.subplots(1, 2)
    ax.imshow(im, cmap='gray')
    ax.set_title("Frame {}".format(ix), fontsize=8, loc='left')
    ax.invert_yaxis()

    #ax = fig.add_subplot(142) 
    ax.plot(fly1[fly1['frame']==ix]['pos_x'], fly1[fly1['frame']==ix]['pos_y'], 'r*')
    ax.plot(fly2[fly2['frame']==ix]['pos_x'], fly2[fly2['frame']==ix]['pos_y'], 'bo')
    ax.set_aspect(1)
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    #ax.invert_yaxis()

    ax = fig.add_subplot(122)
    ax.set_title('centered and rotated to focal (*)', fontsize=8, loc='left') 
    # make a markerstyle class instance and modify its transform prop
    ax.plot([0, float(fly1[fly1['frame']==ix]['rot_x'].iloc[0])], 
            [0, float(fly1[fly1['frame']==ix]['rot_y'].iloc[0])], 'r', 
            marker=marker_m, markerfacecolor='r', markersize=10) 
    ax.plot([fly2[fly2['frame']==ix]['rot_x']], [fly2[fly2['frame']==ix]['rot_y']], 'b',
            marker=marker_f, markerfacecolor='b', markersize=10) 
    ax.set_aspect(1)
    ax.set_xlim(0-frame_width, frame_width)
    ax.set_ylim(0-frame_height, frame_height)
    #ax.invert_yaxis()

    return fig

def check_rotation_transform(ix, trk_, cap, id_colors=['r', 'b']):
    '''
    Note that ix should be frame.

    Arguments:
        ix -- _description_
        trk_ -- _description_
        cap -- _description_

    Keyword Arguments:
        id_colors -- _description_ (default: {['r', 'b']})

    Returns:
        _description_
    '''
    # get image
    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    fig = pl.figure(figsize=(12,5)) #pl.subplots(1, 2)
    # plot frame
    ax = fig.add_subplot(131)
    ax.imshow(im, cmap='gray')
    ax.invert_yaxis()
    # plot positions
    for i, d_ in trk_.groupby('id'):
        print('pos:', i, d_[d_['frame']==ix]['pos_x'], d_[d_['frame']==ix]['pos_y'])
        ax.plot(d_[d_['frame']==ix]['pos_x'], d_[d_['frame']==ix]['pos_y'], 
                marker='o', color=id_colors[i], markersize=3)

    fly1 = trk_[trk_['id']==0].copy().reset_index(drop=True)
    fly2 = trk_[trk_['id']==1].copy().reset_index(drop=True)
    # plot rotated positions, male faces EAST on cartesian
    ax = fig.add_subplot(132) #, projection='polar')
    for i, d_ in enumerate([fly1, fly2]):
        #print('rot:', i, d_.iloc[ix]['rot_x'], d_.iloc[ix]['rot_y'])
        pt = np.squeeze(np.array(d_[d_['frame']==ix][['trans_x', 'trans_y']].values))

        print(pt.shape)
        #ang = rotation_angs[ix]        
        #rx, ry = rotate([0, 0], pt, ang)
        ang = -1*fly1[fly1['frame']==ix]['ori'] 
        rotmat = np.array([[np.cos(ang), -np.sin(ang)],
                            [np.sin(ang), np.cos(ang)]])
        #rx, ry = (rotmat @ pt.T).T
        rx, ry = util.rotate_point(pt, ang) #[0, 0], pt, ang)
        print('rot:', i, rx, ry)
        ax.plot(rx, ry,marker='o', color=id_colors[i], markersize=3)
    ax.set_aspect(1)
    ax.set_title('rot')

    # POLAR
    ax = fig.add_subplot(133, projection='polar')
    for i, d_ in enumerate([fly1, fly2]):
        if i==0:
            ax.plot(0, 0, 'r*')
        #ang = fly1.iloc[ix]['ori'] #* -1
        pt = [d_[d_['frame']==ix]['trans_x'], 
              d_[d_['frame']==ix]['trans_y']]
        #ang = rotation_angs[ix]  
        #rx, ry = rotate((0,0), pt, ang)      
        #rx, ry = rotate2(pt, ang) #[0, 0], pt, ang)
        rad, th = util.cart2pol(rx, ry)
        ax.plot(th, rad, marker='o', color=id_colors[i], markersize=3)
    ax.set_aspect(1)
    ax.set_title('polar')
    #ax.set_theta_direction(-1)
    #print(ang)
    fig.suptitle('{}, ang={:.2f}'.format(ix, np.rad2deg(float(ang))))

    return fig




def plot_frame_target_projection(ix, fly1, fly2, cap, xvar='pos_x', yvar='pos_y'):
    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    # get vector between male and female
    xi = fly2.loc[ix][xvar] - fly1.loc[ix][xvar] 
    yi = fly2.loc[ix][yvar] - fly1.loc[ix][yvar]

    # get vector orthogonal to male's vector to female
    ortho_ = [yi, -xi] #ortho_hat = ortho_ / np.linalg.norm(ortho_)

    # project female heading vec onto orthog. vec
    f_ori = fly2.loc[ix]['ori']
    f_len = fly2.loc[ix]['major_axis_len']
    fem_vec = util.get_heading_vector(f_ori, f_len) #np.array([x_, y_])
    #female_hat = fem_vec / np.linalg.norm(fem_vec)
    vproj_ = util.proj_a_onto_b(fem_vec, ortho_)

    # plot
    fig, axn =pl.subplots(1, 2, figsize=(8, 4))
    ax = axn[0]
    ax.imshow(im, cmap='gray')
    ax.set_title("Frame {}".format(ix)) 
    # plot original positions
    x0, y0 = fly1.loc[ix][[xvar, yvar]]
    x1, y1 = fly2.loc[ix][[xvar, yvar]]
    # plot vector between
    ax.plot([x0, x0+xi], [y0, y0+yi])
    # plot orthogonal
    ax.plot([x1, x1+ortho_[0]], [y1, y1+ortho_[1]], 'orange')
    ax.set_aspect(1)
    # plot female heading
    ax.plot([x1, x1+fem_vec[0]], [y1, y1+fem_vec[1]], 'magenta')
    # plot proj
    ax.plot([x1, x1+vproj_[0]], [y1, y1+vproj_[1]], 'cyan')
    ax.plot([x1, x1-vproj_[0]], [y1, y1-vproj_[1]], 'cyan')

    # plot the vectors only
    ax=axn[1]
    ax.plot([0, xi], [0, yi], 'b')
    ax.plot([0, fem_vec[0]], [0, fem_vec[1]], 'magenta')
    ax.plot([0, ortho_[0]], [0, ortho_[1]], 'orange')
    ax.invert_yaxis()
    ax.plot([0, vproj_[0]], [0, vproj_[1]], 'cyan')
    #ax.plot([0, proj_[0]], [0, proj_[1]], 'magenta')
    ax.set_aspect(1)

    # check
    #diff = (vproj_ - np.array([fem_vec[0], fem_vec[1]]))
    #print(np.dot(diff, ortho_)) #(diff[0] * ortho_hat[0] ) + (diff[1] * ortho_hat[1])

    #ppm = calib_['PPM']
    #print(np.sqrt(xi**2 + yi**2)/ppm)
    #print(feat_.loc[ix]['dist_to_other']) 

    # len of projected female
    fem_sz = np.sqrt(vproj_[0]**2 + vproj_[1]**2) * 2
    dist_to_other = np.sqrt(xi**2 + yi**2)
    fem_sz_deg = 2*np.arctan(fem_sz/(2*dist_to_other))
    ax.set_title('Targ is {:.2f} deg. vis. ang'.format(np.rad2deg(fem_sz_deg)))

    return fig

def get_target_sizes_df(fly1, fly2, xvar='pos_x', yvar='pos_y'):
    '''
    For provided df of tracks (FlyTracker), calculates the size of target in deg. 

    Arguments:
        fly1 -- df of tracks.mat for fly1 (male or focal fly) 
        fly2 -- df of tracks.mat for fly2 (female or target fly)

    Keyword Arguments:
        xvar -- position var to use for calculating vectors (default: {'pos_x'})
        yvar -- same as xvar (default: {'pos_y'})

    Returns:
        fly2 -- returns fly2 with new column 'size_deg'
    '''
    fem_sizes = []
    for ix in fly1.index.tolist():
        xi = fly2.loc[ix][xvar] - fly1.loc[ix][xvar] 
        yi = fly2.loc[ix][yvar] - fly1.loc[ix][yvar]
        f_ori = fly2.loc[ix]['ori']
        f_len_maj = fly2.loc[ix]['major_axis_len']
        f_len_min = fly2.loc[ix]['minor_axis_len']
        # take into account major/minor axes of ellipse
        fem_sz_deg_maj = util.calculate_female_size_deg(xi, yi, f_ori, f_len_maj)
        fem_sz_deg_min = util.calculate_female_size_deg(xi, yi, f_ori, f_len_min)
        fem_sz_deg = np.max([fem_sz_deg_maj, fem_sz_deg_min])
        fem_sizes.append(fem_sz_deg)

    fly2['targ_ang_size'] = fem_sizes
    fly2['targ_ang_size_deg'] = np.rad2deg(fly2['targ_ang_size'])
    # copy same info for f1
    fly1['targ_ang_size'] = fem_sizes
    fly1['targ_ang_size_deg'] = np.rad2deg(fly2['targ_ang_size'])

    return fly1, fly2


def get_relative_velocity(df_, win=1, 
                          value_var='dist_to_other', time_var='sec'):
    '''
    Calculate relative velocity between two flies, relative metric (one fly).
    If using FlyTracker feat.mat, dist_to_other is in mm, and time is sec.

    Arguments:
        fly1 -- feat_ dataframe for fly1

    Keyword Argumentsprint(figdir, figname)

:
        value_var -- relative dist variable to calculate position diff (default: {'dist_to_other'})
        time_var -- time variable to calculate time diff (default: {'sec'})
    '''
    # fill nan of 1st value with 0
    df_['{}_diff'.format(value_var)] = df_[value_var].interpolate().diff().fillna(0)# if dist incr, will be pos, if distance decr, will be neg
    df_['{}_diff'.format(time_var)] = df_[time_var].interpolate().diff().fillna(0) # if dist incr, will be pos, if distance decr, will be neg

    df_['rel_vel'] = df_['{}_diff'.format(value_var)] / (win*df_['{}_diff'.format(time_var)].mean())
    df_['rel_vel_abs'] = df_['{}_diff'.format(value_var)].abs() / (win*df_['{}_diff'.format(time_var)].mean())

    return df_


#%%
def get_copulation_ix(acq):
    cop_ele = {
        '20231213-1103_fly1_eleWT_5do_sh_eleWT_5do_gh': 52267,
        '20231213-1154_fly3_eleWT_6do_sh_eleWT_5do_gh': 17243,
        '20231214-1051_fly2_eleWT_3do_sh_eleWT_3do_gh': 61541,
        '20231223-1117_fly1_eleWT_5do_sh_eleWT_5do_gh': 55582,
        '20231226-1137_fly2_eleWT_4do_sh_eleWT_4do_gh': 13740,
        '20240105-1007_fly1_eleWT_3do_sh_eleWT_3do_gh': 5051, 
        '20240109-1039_fly1_eleWT_4do_sh_eleWT_4do_gh': 177300
    }

    local_dir = '/Users/julianarhee/Documents/rutalab/projects/courtship'
    fname = 'courtship-free-behavior (Responses) - Form Responses 1.csv'
    meta_fpath = os.path.join(local_dir, fname)
    meta = pd.read_csv(meta_fpath)

    if 'ele' in acq:
        match_ = [v for v in cop_ele.keys() if v.startswith(acq)]
        if len(match_)==0:
            print("No match: {}".format(acq))
            cop_ix = np.nan
        else:
            cop_ix = cop_ele[match_[0]]
    else:
        match_ = [v for v in meta['logfile'] if v.startswith(acq)]
        if len(match_)==0: #,  "{} not found".format(acq)
            print("NO match: {}".format(acq))
            cop_ix = np.nan
        else:
            #cop_ix = meta[meta['logfile']==match_[0]]['FlyTracker: copulation index']
            cop_ix = float(meta.loc[meta['logfile']==match_[0], 'FlyTracker: copulation index'])

    return cop_ix

def get_video_cap(acqdir, movie_fmt='avi'):
    vids = util.get_videos(acqdir, vid_type=movie_fmt)
    alt_movie_fmt = 'mp4' if movie_fmt=='avi' else 'avi'
    try:
        assert len(vids)>0, "Found no video in directory: {}".format(vids)
        vids = [vids[-1]]
    except AssertionError as e:
        vids = util.get_videos(acqdir, vid_type=alt_movie_fmt)
        assert len(vids)==1, "Found more than one video in directory: {}".format(vids)  

    vidpath = vids[0]
    cap = cv2.VideoCapture(vidpath)
    return cap


def do_transformations_on_df(trk_, frame_width, frame_height, 
                             feat_=None,
                             cop_ix=None, flyid1=0, flyid2=1):
    if feat_ is None:
        assert 'dist_to_other' in trk_.columns, "No feat df provided. Need dist_to_other."

    # center x- and y-coordinates
    trk_ = util.center_coordinates(trk_, frame_width, frame_height) 

    # separate fly1 and fly2
    fly1 = trk_[trk_['id']==flyid1].copy().reset_index(drop=True)
    fly2 = trk_[trk_['id']==flyid2].copy().reset_index(drop=True)

    # translate coordinates so that focal fly is at origin
    fly1, fly2 = util.translate_coordinates_to_focal_fly(fly1, fly2)

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    fly1, fly2 = util.rotate_coordinates_to_focal_fly(fly1, fly2)

    # add polar conversion
    # FLIP y-axis? TODO check this
    polarcoords = util.cart2pol(fly2['rot_x'], fly2['rot_y']) 
    fly1['targ_pos_radius'] = polarcoords[0]
    fly1['targ_pos_theta'] = polarcoords[1]
    fly2['targ_pos_radius'] = polarcoords[0]
    fly2['targ_pos_theta'] = polarcoords[1]

    fly1['targ_rel_pos_x'] = fly2['rot_x']
    fly1['targ_rel_pos_y'] = fly2['rot_y']
    fly2['targ_rel_pos_x'] = fly2['rot_x']
    fly2['targ_rel_pos_y'] = fly2['rot_y']

    #% copulation index - TMP: fix this!
    if cop_ix is None or np.isnan(cop_ix):
        cop_ix = len(fly1)
        copulation = False
    else:
        copulation = True
    cop_ix = int(cop_ix)

    #% Get all sizes and aggregate trk df
    fly1, fly2 = get_target_sizes_df(fly1, fly2, xvar='pos_x', yvar='pos_y')

    # recombine trk df
    trk = pd.concat([fly1.iloc[:cop_ix], fly2.iloc[:cop_ix]], axis=0).reset_index(drop=True)#.sort_index()
    trk['copulation'] = copulation

    # Get relative velocity and aggregate feat df
    if feat_ is not None:
        f_list = []
        for fi, df_ in feat_.groupby('id'):
            df_ = get_relative_velocity(df_, win=1, 
                                value_var='dist_to_other', time_var='sec')
            f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
        feat = pd.concat(f_list, axis=0).reset_index(drop=True) #.sort_index()
        feat['copulation'] = copulation
        print(trk.iloc[-1].name, feat.iloc[-1].name)
        df = pd.concat([trk, 
                feat.drop(columns=[c for c in feat.columns if c in trk.columns])], axis=1)
        assert df.shape[0]==trk.shape[0], "Bad merge: {}, {}".format(feat.shape, trk.shape)
    else:
        f_list = []
        assert 'dist_to_other' in trk.columns, "No feat df provided. Need dist_to_other."
        for fi, df_ in trk.groupby('id'):
            df_ = get_relative_velocity(df_, win=1, 
                                value_var='dist_to_other', time_var='sec')
            f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
        df = pd.concat(f_list, axis=0).reset_index(drop=True) #.sort_index()

    #acq = os.path.split(acqdir)[-1]

    return df

def get_metrics_relative_to_focal_fly(acqdir, fps=60, cop_ix=None,
                                      movie_fmt='avi', flyid1=0, flyid2=1,
                                      plot_checks=False,
                                      savedir=None):
    '''
    Load -feat.mat and -trk.mat, do some processing, save processed df to savedir.

    Arguments:
        acqdir -- _description_

    Keyword Arguments:
        fps -- _description_ (default: {60})
        cop_ix -- _description_ (default: {None})
        movie_fmt -- _description_ (default: {'avi'})
        flyid1 -- _description_ (default: {0})
        flyid2 -- _description_ (default: {1})
        plot_checks -- _description_ (default: {False})
        savedir -- _description_ (default: {None})
    '''
    # check output dir
    if savedir is None:
        print("No save directory provided. Saving to acquisition directory.")
        savedir = acqdir
    # load flyracker data
    calib_, trk_, feat_ = util.load_flytracker_data(acqdir, fps=fps)

    # get video file for plotting/sanity checks
#    vids = util.get_videos(acqdir, vid_type=movie_fmt)
#    alt_movie_fmt = 'mp4' if movie_fmt=='avi' else 'avi'
#    try:
#        assert len(vids)>0, "Found no video in directory: {}".format(vids)
#        vids = [vids[-1]]
#    except AssertionError as e:
#        vids = util.get_videos(acqdir, vid_type=alt_movie_fmt)
#        assert len(vids)==1, "Found more than one video in directory: {}".format(vids)  
#
#    vidpath = vids[0]
#    cap = cv2.VideoCapture(vidpath)
    cap = get_video_cap(acqdir, movie_fmt=movie_fmt)

    # N frames should equal size of DCL df
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print(frame_width, frame_height) # array columns x array rows
    # switch ORI
    trk_['ori'] = -1*trk_['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
    df_ = do_transformations_on_df(trk_, frame_width, frame_height, 
                                   feat_=feat_, cop_ix=cop_ix,
                                   flyid1=0, flyid2=1)

    # save
    #% save
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        df_fpath = os.path.join(savedir, '{}_df.pkl'.format(acq))
        with open(df_fpath, 'wb') as f: 
            pkl.dump(df_, f)
        print('Saved: {}'.format(df_fpath))

    #% plot - sanity checks
    if plot_checks:
        fly1 = df_[df_['id']==flyid1]
        fly2 = df_[df_['id']==flyid2]
        # check affine transformations for centering and rotating male
        ix = 6500 #5000 #2500 #590
        fig = plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)
        fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)

        # check projections for calculating size based on distance and angle
        ix = 100 #213527 #5000 #2500 #590
        for ix in [100, 3000, 5000]:
            fig = plot_frame_target_projection(ix, fly1, fly2, cap, 
                                            xvar='pos_x', yvar='pos_y')
            fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)

    return df_



def load_processed_data(acqdir, savedir=None, load=True):
    '''
    Load processed feat and trk dataframes (pkl files) from savedir.

    Arguments:
        acq_dir -- _description_

    Keyword Arguments:
        savedir -- _description_ (default: {None})

    Returns:
        _description_
    '''
    feat_=None; trk=None;
    if savedir is None:
        savedir = acqdir

    acq = os.path.split(acqdir)[-1]
    df_fpath = os.path.join(savedir, '{}_df.pkl'.format(acq))
    #feat_fpath = os.path.join(savedir, '{}_feat.pkl'.format(acq))
    #trk_fpath = os.path.join(savedir, '{}_trk.pkl'.format(acq))

    if load:
        with open(df_fpath, 'rb') as f:
            df_ = pkl.load(f) 
        print('Loaded: {}'.format(df_fpath))
#        with open(feat_fpath, 'rb') as f:
#            feat_ = pkl.load(f) 
#        print('Loaded: {}'.format(feat_fpath))
#
#        with open(trk_fpath, 'rb') as f:
#            trk = pkl.load(f)
#        print('Loaded: {}'.format(trk_fpath))

    else:
        df_ = os.path.exists(df_fpath)
#        feat_ = os.path.exists(feat_fpath)
#        trk = os.path.exists(trk_fpath)

    return df_ #feat_, trk
    


#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process FlyTracker data for relative metrics.')
    parser.add_argument('--savedir', type=str, help='Directory to save processed data.')    
    parser.add_argument('--movie_fmt', type=str, default='avi', help='Movie format (default: avi).')
    parser.add_argument('--flyid1', type=int, default=0, help='ID of focal fly (default: 0).')  
    parser.add_argument('--flyid2', type=int, default=1, help='ID of target fly (default: 1).') 
    parser.add_argument('--plot_checks', type=bool, default=False, help='Plot checks (default: False).')    
    parser.add_argument('--viddir', type=str, default='/Volumes/Julie/38mm_dyad/courtship-videos/38mm_dyad', help='Root directory of videos (default: /Volumes/Julie/38mm_dyad/courtship-videos/38mm_dyad).')   
    parser.add_argument('--new', type=bool, default=False, help='Create new processed data (default: False).')
     
    args = parser.parse_args()
    
    #rootdir = '/Volumes/Julie'
    #assay = '38mm_dyad'
    #viddir = '/Volumes/Julie/courtship-videos/38mm_dyad'
    #savedir = '/Volumes/Julie/free-behavior-analysis/FlyTracker/38mm_dyad/processed'
    #movie_fmt = 'avi'
    #flyid1 = 0
    #flyid2 = 1
    #plot_checks = False

    #savedir = os.path.join(rootdir, 'free-behavior-analysis', 'FlyTracker', 
    #                       assay, 'processed')

#    session = '20240109'
#    flynum = 'fly'
#    acqdir = glob.glob(os.path.join(viddir, '{}*{}*'.format(session, flynum)))[0]
#    get_metrics_relative_to_focal_fly(acqdir,
#                                      savedir=savedir,
#                                      movie_fmt=movie_fmt, 
#                                      flyid1=flyid1, flyid2=flyid2,
#                                      plot_checks=False,
#                                      rootdir=rootdir) 
    #viddir = '/Volumes/Giacomo/free_behavior_data'
    #savedir = '/Volumes/Julie/free-behavior-analysis/FlyTracker/38mm_dyad/processed'

    #viddir = '/Volumes/Giacomo/JAABA_classifiers/projector/changing_dot_size_speed'
    #savedir = '/Volumes/Julie/2d-projector-analysis/FlyTracker/processed_mats'
    #flyid1=0
    #flyid2=1
    #movie_fmt = 'avi'

    #%% 
    viddir = args.viddir 
    savedir = args.savedir
    movie_fmt = args.movie_fmt
    flyid1 = args.flyid1
    flyid2 = args.flyid2

    found_mats = glob.glob(os.path.join(viddir,  '20*', '*', '*feat.mat'))
    print('Found {} processed videos.'.format(len(found_mats)))
    #%%
    #fp = found_mats[0]

    for fp in found_mats:
        acq = os.path.split(os.path.split(fp.split(viddir+'/')[-1])[0])[0]
        #acq = os.path.split(acq_viddir)[0]
        acqdir = os.path.join(viddir, acq)
        print(acq)
        create_new = args.new
        if not create_new:
            df_ = load_processed_data(acqdir, load=False, savedir=savedir)
            if df_ is False: 
                create_new=True #assert ft is True, "No feat df found, creating now."

        if create_new:
            cop_ix = get_copulation_ix(acq)
            get_metrics_relative_to_focal_fly(acqdir,
                                        savedir=savedir,
                                        movie_fmt=movie_fmt, 
                                        flyid1=flyid1, flyid2=flyid2,
                                        plot_checks=False)

#%%
#            get_metrics_relative_to_focal_fly(acqdir,
#                                        savedir=savedir,
#                                        movie_fmt='avi', 
#                                        flyid1=0, flyid2=1,
#                                        plot_checks=False)
#
