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

OUTPUT_DIR = r"/media/Data/EDQ/data_win_select/"

SAVE_RAW_TXT = True
SAVE_RAW_NPY = True

SAVE_STIM_TXT = True
SAVE_STIM_NPY = True

window_skip=0.2
analysis_win_sizes = [0.1, 0.175] #in seconds
analysis_win_sizes = [0.1]

selection_algorithms = ['fiona', 'dixon1', 'dixon2', 'dixon3', 'jeff']
#selection_algorithms = [ 'fiona']

import os, sys
import glob
import re
import time


import numpy as np
import matplotlib.pylab as plt
plt.ion()

from constants import (et_nan_values)

from edq_shared import (getFullOutputFolderPath, nabs, 
                        save_as_txt, add_field,
                        detect_rollingWin,
                        filter_trackloss,
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


def getGeometry(data):
    return np.mean((1/(np.degrees(2*np.arctan(data['screen_width']/(2*data['eye_distance'])))/data['display_width_pix']),
                    1/(np.degrees(2*np.arctan(data['screen_height']/(2*data['eye_distance'])))/data['display_height_pix'])))

if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)
                
OUTPUT_DIR = getFullOutputFolderPath(OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

print 'OUTPUT_FOLDER:', OUTPUT_DIR

                
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
    
            DATA = filter_trackloss(DATA, et_model)
                    
            #Removes every second sample for LC Tech EyeFollower
            #TODO: come up with a better alternative
            if et_model == 'eyefollower':
                _r = range(0,len(DATA['time']), 2)
                DATA=DATA[_r]
                        
            ### Add fields to data file
            for win_size in analysis_win_sizes:
                for eye in ['left', 'right']:
                    for wsa in ['fiona', 'dixon1', 'dixon2', 'dixon3', 'jeff']: #Add all in order to be consistent across all files
                        variable_to_add = '_'.join((eye, wsa, np.str(np.int32(win_size*1000))))
                        DATA = add_field(DATA, [(variable_to_add, '|u1')])
                        
            for win_size in analysis_win_sizes:

                print 'tracker: {et_model}, sub: {sub}, win: {win}'.format(et_model=et_model, sub=DATA['subject_id'][0], win=win_size)
                
                args={
                      'win_size': win_size,
                      'window_skip': window_skip,
                      'wsa': selection_algorithms}  
                
                stim=detect_rollingWin(DATA, **args)
                stim_all.extend(stim.tolist())
    
            print 'Analysis time:', time.time()-t1
            

            #Save output
            TRACKER_OUTPUT_DIR = '{root}/{tracker}'.format(root=OUTPUT_DIR, tracker=et_model)
            if not os.path.exists(TRACKER_OUTPUT_DIR):
                os.mkdir(TRACKER_OUTPUT_DIR)
            
            fname = file_path.rsplit(os.path.sep)[-1][:-4]
            save_path = '{output_dir}/{fname}_win_select'.format(output_dir=TRACKER_OUTPUT_DIR, fname=fname)
            
            if SAVE_RAW_NPY:
                t1 = time.time()
                np.save(save_path, DATA)
                print 'RAW_NPY saving time: ', time.time()-t1
            if SAVE_RAW_TXT:
                t1 = time.time()
                save_as_txt(save_path+'.txt', DATA)
                print 'RAW_TXT saving time: ', time.time()-t1
   
        except:
            with open(OUTPUT_DIR+'/errors.info', 'a') as _f:
                _f.write(file_path+'\n')
                print 'Something went wrong:)', DATA['subject_id'][0]
   
   
    stim_all = np.array(stim_all, dtype=stim.dtype) 
    if SAVE_STIM_NPY: 
       np.save('{output_dir}/edq_measures_{git}'.format(output_dir=OUTPUT_DIR, git=getFullOutputFolderPath('/')[1:]), stim_all)
    if SAVE_STIM_TXT:
       save_as_txt('{output_dir}/edq_measures_{git}.txt'.format(output_dir=OUTPUT_DIR, git=getFullOutputFolderPath('/')[1:]), stim_all)

              
sys.exit()

