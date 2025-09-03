# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:28:54 2023

plots
Based on code by Mauricio Girardi Schappo

@author: Kelly
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg #for plotting the image
import numpy as np
import scipy.signal as sig
from modules import lib_process_data_to_mat as plib
from modules import lib_plot_mouse_trajectory as pltlib
import modules.config as cfg
from modules.cv_arena_hole_detect import detect_arena_circle, detect_arena_hole_coords, transpose_coords

def get_hole_coords(data, update_data = True, recalc=False):
    ''''gets hole coord data from file, If it doesn't exist, calculate it.'''
    
    if hasattr(data, 'r_arena_holes') and recalc == False: #checks if hole coords have already been calculated and we don't want to recalculate
        r_holes = data.r_arena_holes
    else:
        arena_circle, gray = detect_arena_circle(os.path.join(cfg.ROOT_DIR, 'data', 'BackgroundImage', data.bkgd_img),
                                 mask_sensitivity=60.) #40 for single trial, 60 for others
        holes = detect_arena_hole_coords(arena_circle, gray)
        r_holes, arena_coords = transpose_coords(holes, arena_circle, gray.shape, data.img_extent)
        
        data.r_arena_holes = r_holes
        data.arena_circle = arena_coords
        
        if update_data: data.Update()
    return r_holes


def find_trajectory_hole_intersections(trialdata, r_holes, idx_end = False, hole_radius = 5.):
    
    if isinstance(idx_end, bool) or idx_end > len(trialdata.r_center): #check if we are cropping trajectory
        idx_end = len(trialdata.r_center)
    
    if len(trialdata.r_nose) > 1.: bodypoint = trialdata.r_nose #try to use the nose points if they exist
    else: bodypoint = trialdata.r_center
    
    idx_inter = [] #list of indices where intersection happens (one arena hole per list item)
    r_inter    = [] # list of intersected coordinates of the trajectory (coords correspond to idx_inter)
    k_hole     = np.zeros(r_holes.shape[0],dtype=bool) #key for which hole it was
    
    for i,r0 in enumerate(r_holes): # counting the keys and coords for each hole
        tind = np.nonzero(np.linalg.norm(bodypoint[:idx_end,:] - r0,axis=1) < hole_radius)[0] #linalg.norm calculates magnitude vector between coords and holes
        if len(tind) > 0:
            idx_inter.append(tind)
            r_inter.append(bodypoint[tind])
            k_hole[i] = True #true if this hole is intersected
        else: #intersections not found
            idx_inter.append(None)
            r_inter.append(None)
    return idx_inter, r_inter, k_hole


def find_slowing_down_near_hole(velocity, idx_inter): #depreciated!

    # y = velocity
    # peaks, _ = find_peaks(-y, height=-60, distance = 3.) #find minima, no hole checks above 60cm/s

    threshold = np.nanmin(velocity) + 0.2 * (np.nanmax(velocity) - np.nanmin(velocity))
    f = (velocity[:-1]-threshold)*(velocity[1:]-threshold) # f<=0 -> crossing of velocity threshold;
    # idx_traj_slow = np.nonzero(f <= 0)[0] # index of all threshold crossings
    idx_traj_slow = np.nonzero(np.logical_and( f <= 0 , velocity[1:]<=threshold))[0] #index of only downward crossings
    
    #try replacing peaks with idx_traj_slow
    k_holes_slow = [] #keys of holes where slowing occured
    idx_traj_slow_holes = [] #index to get timepoints when slowing occured in vicinity of hole
    for i,k in enumerate(idx_inter):
        if k is not None:
            if len(np.intersect1d(idx_traj_slow, k)) > 0:
                k_holes_slow.append(i)
                idx_traj_slow_holes.append(np.intersect1d(idx_traj_slow, k))
        else:
            k_holes_slow.append(None)
            idx_traj_slow_holes.append(None)
            
    return idx_traj_slow, k_holes_slow, idx_traj_slow_holes

def curvature(A,B,C):
    """Calculates the Menger curvature from three Points, given as numpy arrays.
    Source: https://stackoverflow.com/questions/55526575/python-automatic-resampling-of-data
    """

    # Pre-check: Making sure that the input points are all numpy arrays
    if any(x is not np.ndarray for x in [type(A),type(B),type(C)]):
        print("The input points need to be a numpy array, currently it is a ", type(A))

    # Augment Columns
    A_aug = np.append(A,1)
    B_aug = np.append(B,1)
    C_aug = np.append(C,1)

    # Caclulate Area of Triangle
    matrix = np.column_stack((A_aug,B_aug,C_aug))
    area = 1/2*np.linalg.det(matrix)

    # Special case: Two or more points are equal 
    if np.all(A == B) or  np.all(B == C) or np.all(A == C):
        curvature = 0
    else:
        curvature = 4*area/(np.linalg.norm(A-B)*np.linalg.norm(B-C)*np.linalg.norm(C-A))

    # Return Menger curvature
    return curvature

def get_traj_curvatures(data, idx_end):
    try: x, y = data.r_nose[:idx_end,0], data.r_nose[:idx_end,1]
    except: x, y = data.r_center[:idx_end,0], data.r_center[:idx_end,1]
    
    curvature_list = np.empty(0) 
    for i in range(len(x)-2):
        # Get the three points
        A = np.array([x[i],y[i]])
        B = np.array([x[i+1],y[i+1]])
        C = np.array([x[i+2],y[i+2]])
    
        # Calculate the curvature
        curvature_value = abs(curvature(A,B,C))
        curvature_list = np.append(curvature_list, curvature_value)
    curvature_list = np.append(curvature_list, [0,0]) #add two trailing zeros so it is the same length as original coords
    return curvature_list

def peakdet(v, delta, x = None):
    """
    Made by endolith at https://gist.github.com/endolith/250860
    Based on MATLAB script by Eli Billauer at http://billauer.co.il/peakdet.html
    
    Returns two arrays, peaks and valleys
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        raise ValueError('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        raise ValueError('Input argument delta must be a scalar')
    
    if delta <= 0:
        raise ValueError('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)

def find_sharp_curve_near_hole(curvatures, idx_inter, delta = 1.):
    # threshold = np.nanmin(curvatures) + 0.2 * (np.nanmax(curvatures) - np.nanmin(curvatures))
    # peaks, _ = sig.find_peaks(curvatures, height=25., distance = 3.) #find maxima
    # peaks = sig.find_peaks_cwt(curvatures, widths = np.arange(0.5,5), gap_thresh = 20, noise_perc=5) #find maxima
    peaks, _ = peakdet(curvatures, delta) #forumula from matlab, usually delta = 1
    
    # f = (velocity[:-1]-threshold)*(velocity[1:]-threshold) # f<=0 -> crossing of velocity threshold;
    # idx_traj_slow = np.nonzero(f <= 0)[0] # index of all threshold crossings
    # idx_traj_slow = np.nonzero(np.logical_and( f <= 0 , velocity[1:]<=threshold))[0] #index of only downward crossings
    
    #try replacing peaks with idx_traj_slow
    k_holes_curve = [] #keys of holes where curving occured
    idx_traj_holes_curve = [] #index to get timepoints when curving occured in vicinity of hole
    for i,k in enumerate(idx_inter):
        if k is not None:
            if len(np.intersect1d(peaks, k)) > 0:
                k_holes_curve.append(i)
                idx_traj_holes_curve.append(np.intersect1d(peaks, k))
            else: #intersected hole but didn't check
                k_holes_curve.append(None)
                idx_traj_holes_curve.append(None)
        else: #didn't go to hole
            k_holes_curve.append(None)
            idx_traj_holes_curve.append(None)

    return peaks[:,0].astype(int), k_holes_curve, idx_traj_holes_curve


# from itertools import groupby
# from operator import itemgetter
def get_times(data, idx_inter, idx_traj_holes_curve):
    already_visited_holes = []
    k_times = [] # hole key and trajectory index
    for i, hole in enumerate(idx_inter):
        if hole is not None and idx_traj_holes_curve[i] is not None:
            for x in enumerate(hole):
                if x[0]-x[1] in already_visited_holes: #did we already group this sequence?
                    pass
                elif np.intersect1d(idx_traj_holes_curve[i], x[1]).size != 0: #does this group intersect with hole checking?
                    for h in idx_traj_holes_curve[i]:
                        if h == x[1]: 
                            k_times.append([i,x[1]]) #append both hole key and trajectory index
                            already_visited_holes.append(x[0]-x[1])
    k_times=np.array(k_times)
                
    
    times = []
    for i in k_times: #idx_traj_holes_curve or idx_traj_slow_holes
        times.append(data.time[i[1]])
    times = sorted(list(set(times))) #remove duplicates & sorts in decending order
    return times, k_times

#%%
import matplotlib.patches as patches
def plot_hole_checks(data, k_times, crop_at_target=True, crop_time = False, savefig=False):

    fig, ax = plt.subplots()
    
    # img = mpimg.imread(os.path.join(cfg.ROOT_DIR, 'data', 'BackgroundImage', data.bkgd_img))
    # ax.imshow(img, extent=data.img_extent) #plot image to match ethovision coordinates
    
    #draws arena
    Drawing_arena_circle = plt.Circle( (data.arena_circle[0], data.arena_circle[1]), 
                                          data.arena_circle[2] , fill = False )
    ax.add_artist( Drawing_arena_circle )
    
    for c in data.r_arena_holes:
        small_hole = plt.Circle( (c[0], c[1] ), 0.5 , fill = False ,alpha=0.5)
        ax.add_artist( small_hole )
        
    
    
    if crop_at_target and isinstance(crop_time, bool):
        idx_end = pltlib.coords_to_target(data.r_nose, data.target)
    elif crop_time == '2min' : 
        idx_end = np.searchsorted(data.time, 120.)
    elif crop_time == '5min' : idx_end = np.searchsorted(data.time, 300.)
    else:
        idx_end = len(data.r_nose)
        
    k_times = k_times[k_times[:,1] <= idx_end] #crop k_times
        
    colors_time_course = plt.get_cmap('cool') # plt.get_cmap('cool') #jet_r
    t_seq_hole = data.time[k_times[:,1]]/data.time[idx_end-1]
    t_seq_traj = data.time/data.time[idx_end-1]
        
    plt.plot(data.r_nose[:idx_end,0], data.r_nose[:idx_end,1], color='k')
    
    # ax.scatter(data.r_nose[:idx_target,0], data.r_nose[:idx_target,1], s=1.5, facecolors=colors_time_course(t_seq_traj[:idx_target])) #plot path with colours
    
    #plots hole checks
    
    ax.scatter(data.r_arena_holes[k_times[:,0]][:,0], data.r_arena_holes[k_times[:,0]][:,1], 
               s=50, marker = 'o', facecolors='none', edgecolors=colors_time_course(t_seq_hole), 
               linewidths=2.)
    
    #draw target
    target = plt.Circle((data.target), 2.5 , color='b', alpha=1)
    ax.add_artist(target)

    #draw entrance
    for i, _ in enumerate(data.r_nose):
        if np.isnan(data.r_nose[i][0]): continue
        else:
            first_coord = data.r_nose[i]
            break
    entrance = plt.Rectangle((first_coord-3.5), 7, 7, fill=False, color='k', alpha=0.8, lw=3)
    ax.add_artist(entrance)
    
    # Create a Rectangle patch
    # rect = patches.Rectangle((data.r_nose[0]), 7, 7, linewidth=5, edgecolor='k', facecolor='none')
    # ax.add_patch(rect)

    
    # #plots all holes
    # for hole in r_holes:
    #     target = plt.Circle((hole), 1., color='y')
    #     ax.add_artist(target)
    
    #plot only the holes that were intersected
    # for r0 in r_holes[k_hole]:
    #     target = plt.Circle(r0, 1., color='r') 
    #     ax.add_artist(target)
    
    # for r0 in r_holes[k_holes_slow]:
    #     target = plt.Circle(r0, 1., color='b') 
    #     ax.add_artist(target)
        
    #plot only the holes where curves peaked
    # for k in k_holes_curve:
    #     if k is not None:
    #         target = plt.Circle(r_holes[k], 3., fill = False, edgecolor = 'b', linewidth=1.2)
    #         ax.add_artist(target)
        
    
    ax.set_aspect('equal','box')
    ax.set_xlim([data.img_extent[2],data.img_extent[3]])
    ax.set_ylim([data.img_extent[2],data.img_extent[3]])
    ax.axis('off')
    
    if savefig == True:
        plt.savefig(cfg.ROOT_DIR+'/figures/HoleCheck_%s_M%s_%s.png'%(data.protocol_name, data.mouse_number, data.trial), dpi=600, bbox_inches='tight', pad_inches = 0)
        
    
    
    plt.show()
    
#%%
def plot_hole_check_heatmap(): #in progress, want to change circle size based on number of checks
    fig, ax = plt.subplots()
    
    #draws arena
    Drawing_arena_circle = plt.Circle( (data.arena_circle[0], data.arena_circle[1]), 
                                          data.arena_circle[2] , fill = False )
    ax.add_artist( Drawing_arena_circle )
    
    for c in data.r_arena_holes:
        small_hole = plt.Circle( (c[0], c[1] ), 0.5 , fill = False ,alpha=0.5)
        ax.add_artist( small_hole )
    
#%%

def plot_curvatures():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
    cutoff = 1000#len(data.time)
    
    x = data.time.reshape((-1, 1))
    # y = data.velocity
    y = curvature_list
    ax.plot(x[:cutoff], y[:cutoff],'-',c='k',alpha=0.5)
    
    # plt.hlines(246.17, 0, cutoff/25)
    
    x_slow = [i for i in peaks if i <= cutoff]
    plt.plot( x[x_slow], y[x_slow], "o") #plot peaks (slowdowns or curve inflections)
    
    
    x_slow_and_hole = [i for i in idx_traj_holes_curve[:,1] if i <= cutoff]
    colors_time_course = plt.get_cmap('jet_r')
    t_seq    = data.time[x_slow_and_hole]/data.time[-1]
    # idx_target = pltlib.coords_to_target(data.r_center, data.target)
    
    plt.scatter(data.time[x_slow_and_hole], np.zeros(len(x_slow_and_hole)), color=colors_time_course(t_seq))
    
    # plt.vlines(data.time[idx_target], 0, 25, linestyles='--', color='r') #target reached!
    
    plt.show()

#%%    
import matplotlib.animation as animation 

def plot_animated_traj(savefig = False):
    fig, ax = plt.subplots()
    ax.set_aspect('equal','box')
    ax.set_xlim([data.img_extent[2],data.img_extent[3]])
    ax.set_ylim([data.img_extent[2],data.img_extent[3]])
    
    #draws arena
    Drawing_arena_circle = plt.Circle( (data.arena_circle[0], data.arena_circle[1]), 
                                          data.arena_circle[2] , fill = False )
    ax.add_artist( Drawing_arena_circle )
    for c in data.r_arena_holes:
        small_hole = plt.Circle( (c[0], c[1] ), 0.7 , fill = False ,alpha=0.5)
        ax.add_artist( small_hole )
        
    # target = plt.Circle( (data.target[0], data.target[1] ), 2.5 , 
    #                     lw = 2, fill = False , edgecolor = 'g', alpha=0.5)
    # target_r = plt.Circle( (data.target_reverse[0], data.target_reverse[1]), 2.5 , 
    #                       lw = 2, fill = False , edgecolor = 'b', alpha=0.5)
    # ax.add_artist(target)
    # ax.add_artist(target_r)
    
    line, = ax.plot([], [], lw=2) 
    
    scat = ax.scatter([], [],
                      s=80, lw=2, edgecolors='r', marker = 'o',
                      facecolors='none')
    
    
    # initialization function 
    def init(): 
        # creating an empty plot/frame 
        line.set_data([], []) 
        # scat.set_offsets([0,0])
        return line, #scat
    
    # lists to store x and y axis points 
    xdata, ydata = [], [] 
    xdata_hole, ydata_hole = [], [] 
    
    # animation function 
    def animate(i): 
        
        if len(data.r_nose) > 1.:
            # x, y values to be plotted 
            x = data.r_nose[i,0]
            y = data.r_nose[i,1]
        else:
            x = data.r_center[i,0]
            y = data.r_center[i,1]
        
        if np.intersect1d(k_times[:,1], i).size != 0.:
            for h in k_times:
                if h[1]==i:
                    x_h = data.r_arena_holes[h[0]][0]
                    y_h = data.r_arena_holes[h[0]][1]
                    xdata_hole.append(x_h)
                    ydata_hole.append(y_h)
                    scat.set_offsets(np.c_[xdata_hole, ydata_hole])
            
            # target = plt.Circle(r_holes[i], 3., fill = False, edgecolor = 'b', linewidth=1.2)
            # tar = ax.add_artist(target)
    	
        # appending new points to x, y axes points list 
        xdata.append(x) 
        ydata.append(y) 
        
        
        line.set_data(xdata, ydata)     
        
        return line, scat
    	
    # hiding the axis details 
    plt.axis('off') 
    
    # call the animator	 
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
    							frames=2100, interval=30, blit=True, repeat=False) 
    if savefig:
        anim.save(f'{cfg.ROOT_DIR}/figures/animation_{data.exp}_{data.mouse_number}_{data.trial}.mp4')   
    plt.show()
    
def main_wrap_get_time(data):
    idx_inter, r_inter, k_hole = find_trajectory_hole_intersections(data, data.r_arena_holes, idx_end=False, hole_radius = 2.)
    curvature_list = get_traj_curvatures(data, len(data.time))
    peaks, k_holes_curve, idx_traj_holes_curve = find_sharp_curve_near_hole(curvature_list, idx_inter, delta = 1.)
    time, k_times = get_times(data, idx_inter, idx_traj_holes_curve)
   
    return time, k_times
    
#%%

if __name__ == '__main__':
    data = plib.TrialData()
    data.Load('2025-08-22', '60', '7')

    #velocity based detection
    # idx_inter, r_inter, k_hole = find_trajectory_hole_intersections(data, r_holes, 18000, hole_radius = 4.)
    # idx_traj_slow, k_holes_slow, idx_traj_slow_holes = find_slowing_down_near_hole(data.velocity, idx_inter)

    #curvature based detection
    idx_inter, r_inter, k_hole = find_trajectory_hole_intersections(data, data.r_arena_holes, idx_end=False, hole_radius = 2.)
    curvature_list = get_traj_curvatures(data, len(data.time))
    peaks, k_holes_curve, idx_traj_holes_curve = find_sharp_curve_near_hole(curvature_list, idx_inter, delta = 1.)
    time, k_times = get_times(data, idx_inter, idx_traj_holes_curve)
    
    plot_hole_checks(data, k_times, crop_at_target=True, crop_time=False, savefig=False)
    # plot_animated_traj(savefig=False)
    