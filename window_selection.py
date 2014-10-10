# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:43:00 2014

@author: Raimondas Zemblys
@email: raimondas.zemblys@humlab.lu.se

Script for finding the eye samples that fall within a target presentations selection windows.

Acknowledgement
===============
This script has been produced during my scholarship period at Lund University, 
thanks to a Swedish Institute scholarship.

"""
INCLUDE_TRACKERS = (
                    #'dpi',
#                    'eyefollower', 
                    'eyelink', 
#                    'eyetribe', 
#                    'hispeed1250', 
#                    'hispeed240',
#                    'red250', 
#                    'red500', 
#                    'redm', 
#                    't60xl', 
#                    'tx300', 
#                    'x2'
)


INCLUDE_SUB = 'ALL'
INCLUDE_SUB = [15]

INPUT_FILE_ROOT = r"/media/Data/EDQ/data_npy/"
GLOB_PATH_PATTERN = INPUT_FILE_ROOT+r"*/*.npy"

STIM_OUTPUT_DIR = INPUT_FILE_ROOT

SAVE_RAW_TXT = False
SAVE_RAW_NPY = True

SAVE_STIM_TXT = True
SAVE_STIM_NPY = True

window_skip=0.2
analysis_win_sizes = [0.1, 0.175] #in seconds
#analysis_win_sizes = [0.1]

selection_algorithms = ['fiona', 'dixon1', 'dixon2', 'dixon3', 'jeff']
#selection_algorithms = [ 'fiona']

import os, sys
import glob
import re
import time


import numpy as np
import matplotlib.pylab as plt
plt.ion()

from constants import (et_nan_values, stim_dtype, stim_pos_mappings)

from edq_shared import (getFullOutputFolderPath, nabs, 
                        save_as_txt, rolling_window, add_field,
                        parseTrackerMode
)
  
    
def analyseit(fpath, include_sub, include_trackers):
    """

    :param fpath:
    :return:
    """
    tracker_type, sub = getInfoFromPath(fpath)
    if include_sub == 'ALL':
        return (tracker_type in include_trackers)
    else:
        return (tracker_type in include_trackers) & (sub in include_sub)
    
def getInfoFromPath(fpath):
    """

    :param fpath:
    :return:
    """
    if fpath.lower().endswith(".npy"):
        fpath, fname = os.path.split(fpath)
    return fpath.rsplit(os.path.sep, 3)[-1], np.uint(re.split('_|.npy', fname)[-2])

def rolling_measures(data, eye, units, win_size):
    
    #inter-sample distances
    isd = np.diff(zip(data['_'.join((eye, units, 'x'))], 
                      data['_'.join((eye, units, 'y'))]), axis=0)
    isd = np.vstack((isd[:,0], isd[:,1], np.hypot(isd[:,0], isd[:,1]))).T
    

    measures=dict()
    
    #RMS
    measures['_'.join((eye, units, 'RMS', 'x'))] = np.sqrt(np.mean(rolling_window(isd[:,0], win_size-1)**2, 1))
    measures['_'.join((eye, units, 'RMS', 'y'))] = np.sqrt(np.mean(rolling_window(isd[:,1], win_size-1)**2, 1))
    measures['_'.join((eye, units, 'RMS'))] = np.sqrt(np.mean(rolling_window(isd[:,2], win_size-1)**2, 1))
    
    rolling_data_x = rolling_window(data['_'.join((eye, units, 'x'))], win_size)
    rolling_data_y = rolling_window(data['_'.join((eye, units, 'y'))], win_size)
    
    #STD  
    measures['_'.join((eye, units, 'STD', 'x'))] = np.std(rolling_data_x, axis=1)
    measures['_'.join((eye, units, 'STD', 'y'))] = np.std(rolling_data_y, axis=1)
    measures['_'.join((eye, units, 'STD'))] = np.hypot(measures['_'.join((eye, units, 'STD', 'x'))],
                                                       measures['_'.join((eye, units, 'STD', 'y'))]   
                                              )
    
    #ACC
    fix = [np.median(rolling_data_x, axis=1),
           np.median(rolling_data_y, axis=1)]

    measures['_'.join((eye, units, 'ACC', 'x'))] = fix[0] - data[stim_pos_mappings[units]+'x'][:-win_size+1]
    measures['_'.join((eye, units, 'ACC', 'y'))] = fix[1] - data[stim_pos_mappings[units]+'y'][:-win_size+1]
    measures['_'.join((eye, units, 'ACC'))] = np.hypot(measures['_'.join((eye, units, 'ACC', 'x'))],
                                                       measures['_'.join((eye, units, 'ACC', 'y'))]   
                                              )
    
    measures['_'.join((eye, units, 'fix', 'x'))] = fix[0]
    measures['_'.join((eye, units, 'fix', 'y'))] = fix[1]
    
    measures['_'.join((eye, units, 'range', 'x'))] = np.nanmax(rolling_data_x, axis=1)-np.nanmin(rolling_data_x, axis=1)
    measures['_'.join((eye, units, 'range', 'y'))] = np.nanmax(rolling_data_y, axis=1)-np.nanmin(rolling_data_y, axis=1)
    
    return measures

def getGeometry(data):
    return np.mean((1/(np.degrees(2*np.arctan(data['screen_width']/(2*data['eye_distance'])))/data['display_width_pix']),
                    1/(np.degrees(2*np.arctan(data['screen_height']/(2*data['eye_distance'])))/data['display_height_pix'])))


DATA_FILES = [nabs(fpath) for fpath in glob.glob(GLOB_PATH_PATTERN) if analyseit(fpath, INCLUDE_SUB, INCLUDE_TRACKERS)]

stim_all = []

if __name__ == '__main__':
    for file_path in DATA_FILES:
        try:
#        if 1:
            t1 = time.time()
            DATA = np.load(file_path)
            et_model, _ = getInfoFromPath(file_path)
                       
            #Skips multissesion recordings
            #TODO: plot data from all sessions, manually select right one (could be done in hdf2wide step?)
            if (len(np.unique(DATA['session_id'])) > 1):
                with open('multi_session.info', 'a') as _f:
                    _f.write(file_path+'\n')
                continue
    
            ### Trackloss filter DEMO START  ###
            #TODO: + Filter off-screen, off-pshysical limit samples
            #      - Already implemented in hdf2wide, needs to be removed later on when merging
            for eye in ['left', 'right']:
                #trackloss        
                for _dir in ['x', 'y']:
                    trackloss = (DATA['_'.join((eye, 'gaze', _dir))] == et_nan_values[et_model][_dir])
                    DATA['_'.join((eye, 'gaze', _dir))][trackloss] = np.nan
                    DATA['_'.join((eye, 'angle', _dir))][trackloss] = np.nan
                    
            #Removes every second sample for LC Tech EyeFollower
            #TODO: come up with a better alternative
            if et_model == 'eyefollower':
                _r = range(0,len(DATA['time']), 2)
                DATA=DATA[_r]
            ### Trackloss filter DEMO END  ###
            
            analysis_win_sizes_sample = np.int16(np.array(analysis_win_sizes)*DATA['eyetracker_sampling_rate'][0]) #win size in samples
            
            ### Add fields to data file
            for win_size in analysis_win_sizes:
                for eye in ['left', 'right']:
                    for wsa in ['fiona', 'dixon1', 'dixon2', 'dixon3', 'jeff']: #Add all in order to be consistent across all files
                        measure_to_add = '_'.join((eye, wsa, np.str(np.int32(win_size*1000))))
                        DATA = add_field(DATA, [(measure_to_add, '|u1')])
                        
            for win_size_sample, win_size in zip(analysis_win_sizes_sample, analysis_win_sizes):

                print 'tracker: {et_model}, sub: {sub}, win: {win}'.format(et_model=et_model, sub=DATA['subject_id'][0], win=win_size_sample)
                
                measures=dict()
                for eye in parseTrackerMode(DATA['eyetracker_mode'][0]):   
                        
                    measures.update(rolling_measures(DATA, eye, 'gaze', win_size_sample))
                    measures.update(rolling_measures(DATA, eye, 'angle', win_size_sample))
                    
                    for wsa in selection_algorithms:
                        ### Analysis level data matrix: one line per target 
                        #TODO: call initStim from edq_shared
                        stim_change_ind = np.where(np.hstack((1,np.diff(DATA['trial_id']))) == 1)
                        stim_change_count = len(stim_change_ind[0])
                        
                        wsa_list=np.empty_like(DATA['eyetracker_model'][stim_change_ind])
                        wsa_list[:]=wsa
                        
                        stim = np.array(
                                   zip(
                                       DATA['eyetracker_model'][stim_change_ind],
                                       DATA['eyetracker_sampling_rate'][stim_change_ind],                 
                                       DATA['eyetracker_mode'][stim_change_ind],
                                       np.ones(stim_change_count) * getGeometry(DATA[0]), 
                                       DATA['operator'][stim_change_ind],
            
                                       DATA['subject_id'][stim_change_ind],
                                       DATA['trial_id'][stim_change_ind], 
                                       DATA['ROW_INDEX'][stim_change_ind], 
                                       DATA['dt'][stim_change_ind], 
                                       DATA['TRIAL_START'][stim_change_ind],
                                       DATA['TRIAL_END'][stim_change_ind],
                                       DATA['posx'][stim_change_ind],
                                       DATA['posy'][stim_change_ind],
                                       DATA['target_angle_x'][stim_change_ind],
                                       DATA['target_angle_y'][stim_change_ind],
                                       
                                       wsa_list.tolist(),
                                       np.ones(stim_change_count) * win_size,
                                       np.ones(stim_change_count) * window_skip,
                                       *np.zeros((63, stim_change_count)) * np.nan
                                   ), dtype=stim_dtype
                               )
                        
                        #Remove first and last stimuli
                        stim = stim[1:-1]
                        ###

                        for stim_ind, stim_row in enumerate(stim):
#                            analysis_range = (DATA['time'] >= _stim['TRIAL_START']+window_skip) \
#                                           & (DATA['time'] <= _stim['TRIAL_END']-window_analyse)
                            
                            analysis_range = (DATA['time'] >= stim_row['TRIAL_START']+window_skip) \
                                           & (DATA['time'] <= stim_row['TRIAL_START']+1-win_size)
                            analysis_range_full= (DATA['time'] >= stim_row['TRIAL_START']) \
                                               & (DATA['time'] <= stim_row['TRIAL_START']+1)
                            
                            #needs to be a number to identify starting sample of a window               
                            analysis_range=np.squeeze(np.argwhere(analysis_range==True))
                            analysis_range_full=np.squeeze(np.argwhere(analysis_range_full==True))
                                              
                            measures_rangeRMS = measures['_'.join((eye, 'angle', 'RMS'))][analysis_range]
                            measures_rangeACC = measures['_'.join((eye, 'angle', 'ACC'))][analysis_range]
                            measures_rangeSTD = measures['_'.join((eye, 'angle', 'STD'))][analysis_range]
                            
                            if wsa == 'fiona':
                                measures_range = measures_rangeRMS #Fiona RMS 
                            elif wsa == 'dixon1':
                                measures_range = measures_rangeACC*measures_rangeSTD #Dixons's measure No. 1
                            elif wsa == 'dixon2':
                                measures_range = measures_rangeACC+measures_rangeSTD #Dixons's measure No. 2
                            elif wsa == 'dixon3':
                                measures_range = measures_rangeACC**2+measures_rangeSTD**2 #Dixons's measure No. 3
                            elif wsa == 'jeff':
                                #Jeff's measure                        
                                std_thres = np.nanmin(measures_rangeSTD)*5
                                measures_rangeACC[measures_rangeSTD>std_thres] = np.nan
                                measures_range = measures_rangeACC
                            

                            if np.sum(np.isfinite(measures_range)) > 0: #handle all-nan slice
                                IND = analysis_range[np.nanargmin(measures_range)]
                                
                                #Window selection in raw data
                                export_range = (DATA['time'] >= DATA['time'][IND]) \
                                             & (DATA['time'] <= DATA['time'][IND]+win_size)
                                export_range=np.squeeze(np.argwhere(export_range==True))
                                         
                                DATA[measure_to_add][export_range]=1
                                
                                #save measures to stim
                                stim['_'.join((eye, 'gaze', 'ind'))][stim_ind] = IND
                                stim['_'.join((eye, 'angle', 'ind'))][stim_ind] = IND
                                
                                stim['_'.join((eye, 'window_onset'))][stim_ind] = DATA['time'][IND]-stim_row['TRIAL_START']
                                stim['_'.join((eye, 'sample_count'))][stim_ind] = len(export_range)
                                stim['_'.join((eye, 'actual_win_size'))][stim_ind] = DATA['time'][IND+win_size_sample]-DATA['time'][IND]
                                
                                stim['invalid_sample_count'][stim_ind] = np.sum(np.isnan(DATA[eye+'_gaze_x'][analysis_range_full]) |
                                                                                np.isnan(DATA[eye+'_gaze_y'][analysis_range_full])
                                                                         )
                                
                                for key in measures.keys():
                                    stim[key][stim_ind]=measures[key][IND]
                                             
                        #save stim
                        stim_all.extend(stim.tolist())
    
            print 'Analysis time:', time.time()-t1
            

            #Save output
            WIN_SELECT_OUTPUT = '{root}/{et_model}/{et_model}_win_select_{git}'.format(root=INPUT_FILE_ROOT, 
                                                                                       et_model=et_model,
                                                                                       git=getFullOutputFolderPath('/')[1:])
            if not os.path.exists(WIN_SELECT_OUTPUT):
                os.mkdir(WIN_SELECT_OUTPUT)
            
            fname = file_path.rsplit(os.path.sep)[-1][:-4]
            save_path = '{output_dir}/{fname}_win_select'.format(output_dir=WIN_SELECT_OUTPUT, fname=fname)
            
            if SAVE_RAW_NPY:
                t1 = time.time()
                np.save(save_path, DATA)
                print 'RAW_NPY saving time: ', time.time()-t1
            if SAVE_RAW_TXT:
                t1 = time.time()
                save_as_txt(save_path+'.txt', DATA)
                print 'RAW_TXT saving time: ', time.time()-t1
            
   
        except:
            with open('errors.info', 'a') as _f:
                _f.write(file_path+'\n')
                print 'Something wrong:)', DATA['subject_id'][0]
   
   
    stim_all = np.array(stim_all, dtype=stim.dtype) 
    if SAVE_STIM_NPY: 
       np.save('{output_dir}/edq_measures_{git}'.format(output_dir=STIM_OUTPUT_DIR, git=getFullOutputFolderPath('/')[1:]), stim_all)
    if SAVE_STIM_TXT:
       save_as_txt('{output_dir}/edq_measures_{git}.txt'.format(output_dir=STIM_OUTPUT_DIR, git=getFullOutputFolderPath('/')[1:]), stim_all)

              
sys.exit()

