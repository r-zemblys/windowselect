# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:43:00 2014

@author: huml-rzm

Script analyses all datasets for selected trackers using 5 different window selection methods
and 2 window sizes - 100 ms and 175 ms
Results are saved as additional variables of 0 and 1 (selected window)

REQUIRES data in numpy format, produced by Sol's EDQ conversion script
"""
INCLUDE_TRACKERS = ('eyefollower', 'eyelink', 'eyetribe', 'hispeed1250', 'hispeed240',
              'red250', 'red500', 'redm', 't60xl', 'tx300', 'x2')
INCLUDE_TRACKERS = ('hispeed1250')


INCLUDE_SUB = 'ALL'
#INCLUDE_SUB = [131]

INPUT_FILE_ROOT = r"/media/Data/EDQ/data_npy/"

GLOB_PATH_PATTERN = INPUT_FILE_ROOT+r"*/*.npy"

SAVE_TXT = False
SAVE_NPY = True

import os, sys
import glob
import re
import time


import numpy as np
import matplotlib.pylab as plt
plt.ion()

from constants import (et_nan_values, stim_dtype, stim_pos_fields, win_sizes_config)



def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
def nabs(file_path):
    """

    :param file_path:
    :return:
    """
    return os.path.normcase(os.path.normpath(os.path.abspath(file_path)))
    
def analyseit(fpath):
    """

    :param fpath:
    :return:
    """
    tracker_type, sub = getInfoFromPath(fpath)
    if INCLUDE_SUB == 'ALL':
        return (tracker_type in INCLUDE_TRACKERS)
    else:
        return (tracker_type in INCLUDE_TRACKERS) & (sub in INCLUDE_SUB)
    
def getInfoFromPath(fpath):
    """

    :param fpath:
    :return:
    """
    if fpath.lower().endswith(".npy"):
        fpath, fname = os.path.split(fpath)
    return fpath.rsplit(os.path.sep, 3)[-1], np.uint(re.split('_|.npy', fname)[-2])

def parseTrackerMode(eyetracker_mode):
    if eyetracker_mode == 'Binocular':
        return ['left', 'right']
    else:
        return [eyetracker_mode.split(' ')[0].lower()]

def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError, "`A' must be a structured numpy array"
    b = np.zeros(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b

def rolling_measures(eye, units, win_size):
    
    #inter-sample distances
    isd = np.diff(zip(DATA['_'.join((eye, units, 'x'))], 
                      DATA['_'.join((eye, units, 'y'))]), axis=0)
    isd = np.vstack((isd[:,0], isd[:,1], np.hypot(isd[:,0], isd[:,1]))).T
    

    measures=dict()
    
    #RMS
    measures['_'.join((eye, units, 'RMS', 'x'))] = np.sqrt(np.mean(rolling_window(isd[:,0], win_size-1)**2, 1))
    measures['_'.join((eye, units, 'RMS', 'y'))] = np.sqrt(np.mean(rolling_window(isd[:,1], win_size-1)**2, 1))
    measures['_'.join((eye, units, 'RMS'))] = np.sqrt(np.mean(rolling_window(isd[:,2], win_size-1)**2, 1))
    
    #STD
    STD = [np.std(rolling_window(DATA['_'.join((eye, units, 'x'))], win_size), axis=1),
           np.std(rolling_window(DATA['_'.join((eye, units, 'y'))], win_size), axis=1)]
    STD.append(np.hypot(STD[0], STD[1]))    
    measures['_'.join((eye, units, 'STD', 'x'))] = STD[0]
    measures['_'.join((eye, units, 'STD', 'y'))] = STD[1]
    measures['_'.join((eye, units, 'STD'))] = STD[2]
    
    #ACC
    fix = [np.median(rolling_window(DATA['_'.join((eye, units, 'x'))], win_size), axis=1),
           np.median(rolling_window(DATA['_'.join((eye, units, 'y'))], win_size), axis=1)]
           

    ACC = [DATA[stim_pos_fields[units]+'x'][:-win_size+1]-fix[0],
           DATA[stim_pos_fields[units]+'y'][:-win_size+1]-fix[1],]
    
    ACC.append(np.hypot(ACC[0], ACC[1]))
    
    measures['_'.join((eye, units, 'ACC', 'x'))] = ACC[0]
    measures['_'.join((eye, units, 'ACC', 'y'))] = ACC[1]
    measures['_'.join((eye, units, 'ACC'))] = ACC[2]
    
    measures['_'.join((eye, units, 'fix', 'x'))] = fix[0]
    measures['_'.join((eye, units, 'fix', 'y'))] = fix[1]
    
    return measures

def getGeometry(data):
    return np.mean((1/(np.degrees(2*np.arctan(data['screen_width']/(2*data['eye_distance'])))/data['display_width_pix']),
                    1/(np.degrees(2*np.arctan(data['screen_height']/(2*data['eye_distance'])))/data['display_height_pix'])))

DATA_FILES = [nabs(fpath) for fpath in glob.glob(GLOB_PATH_PATTERN) if analyseit(fpath)]

window_skip=0.2
analysis_win_sizes = [0.1, 0.175]
#analysis_win_sizes = [0.1]

selection_algorithms = ['fiona', 'dixon1', 'dixon2', 'dixon3', 'jeff']
#selection_algorithms = [ 'jeff']

stim_export = []
if __name__ == '__main__':
    for file_path in DATA_FILES:
        try:
#        if 1:
            t1 = time.time()
            DATA = np.load(file_path)
            et_model, _ = getInfoFromPath(file_path)
            
            '''        
            TODO:
                +filter multiple session recordings
                check eyetracker_mode: does recorded data correspont to set mode?
            '''
            
            if (len(np.unique(DATA['session_id'])) > 1):
                with open('multi_session.info', 'a') as _f:
                    _f.write(file_path+'\n')
                continue
    
            ### Trackloss filter DEMO START  ###
            for eye in parseTrackerMode(DATA['eyetracker_mode'][0]):
                #trackloss        
                for _dir in ['x', 'y']:
                    trackloss = (DATA['_'.join((eye, 'gaze', _dir))] == et_nan_values[et_model][_dir])
                    DATA['_'.join((eye, 'gaze', _dir))][trackloss] = np.nan
                    DATA['_'.join((eye, 'angle', _dir))][trackloss] = np.nan
            ### Trackloss filter DEMO END  ###
            
            ### Eyefollower filter DEMO START  ###
            if et_model == 'eyefollower':
                _r = range(0,len(DATA['time']), 2)
                DATA=DATA[_r]
            
            ### Eyefollower filter DEMO END  ###
            
            
    #        analysis_win_sizes = np.arange(win_sizes_config[et_model][0], 
    #                                      (1-window_skip)*DATA['eyetracker_sampling_rate'][0], 
    #                                       win_sizes_config[et_model][1])
    #        analysis_win_sizes = np.arange(win_sizes_config[et_model][0], 
    #                                      (1-window_skip)*DATA['eyetracker_sampling_rate'][0], 
    #                                       1)
            
            
            analysis_win_sizes_fs = np.int16(np.array(analysis_win_sizes)*DATA['eyetracker_sampling_rate'][0])
            
            ### Additional fields
            for win_size, window_analyse in zip(analysis_win_sizes_fs, analysis_win_sizes):
                for eye in ['left', 'right']:
                    for mta in ['fiona', 'dixon1', 'dixon2', 'dixon3', 'jeff']:
                        measure_to_add = '_'.join((eye, mta, np.str(np.int32(window_analyse*1000))))
                        DATA = add_field(DATA, [(measure_to_add, '|u1')])
                        
            for win_size, window_analyse in zip(analysis_win_sizes_fs, analysis_win_sizes):
    #            window_analyse = (win_size) / DATA['eyetracker_sampling_rate'][0]
                print 'sub: {sub}, win: {win}'.format(sub=DATA['subject_id'][0], win=win_size)
                
                ### 
                stim_change_ind = np.where(np.hstack((1,np.diff(DATA['trial_id']))) == 1)
                stim_change_count = len(stim_change_ind[0])
    
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
                               
                               np.ones(stim_change_count) * win_size,
                               np.ones(stim_change_count) * window_skip,
                               *np.zeros((22, stim_change_count)) * np.nan
                           ), dtype=stim_dtype
                       )
                stim = stim[1:-1]
                ###
                
                measures=dict()
                for eye in parseTrackerMode(DATA['eyetracker_mode'][0]):
    #                for eye in ['left']:
#                    print np.nansum(DATA[eye+'_gaze_x'])
                    if sum(np.isnan(DATA[eye+'_gaze_x'])) == len(DATA[eye+'_gaze_x']):
                        with open('eye_select.err', 'a') as _f:
                            _f.write(file_path+'\n')
                            print 'Eye select error', DATA['subject_id'][0]    
                        
                    measures.update(rolling_measures(eye, 'gaze', win_size))
#                    measures.update(rolling_measures(eye, 'angle', win_size))
                    for mta in selection_algorithms:
                        measure_to_add = '_'.join((eye, mta, np.str(np.int32(window_analyse*1000))))        
                        
    #                    DATA = add_field(DATA, [(measure_to_add, '|u1')])
                        
                        ### wi-LRMS
        #                for _stim in [stim[6]]:
                        for stim_ind, _stim in enumerate(stim):
#                            analysis_range = (DATA['time'] >= _stim['TRIAL_START']+window_skip) \
#                                           & (DATA['time'] <= _stim['TRIAL_END']-window_analyse)
                            analysis_range = (DATA['time'] >= _stim['TRIAL_START']+window_skip) \
                                           & (DATA['time'] <= _stim['TRIAL_START']+1-window_analyse)
                                           
                            analysis_range=np.squeeze(np.argwhere(analysis_range==True))
                                              
                            measures_rangeRMS = measures['_'.join((eye, 'gaze', 'RMS'))][analysis_range]
                            measures_rangeACC = measures['_'.join((eye, 'gaze', 'ACC'))][analysis_range]
                            measures_rangeSTD = measures['_'.join((eye, 'gaze', 'STD'))][analysis_range]
                            
                            if mta == 'fiona':
                                measures_range = measures_rangeRMS #Fiona RMS 
                            elif mta == 'dixon1':
                                measures_range = measures_rangeACC*measures_rangeSTD #Dixons's measure No. 1
                            elif mta == 'dixon2':
                                measures_range = measures_rangeACC+measures_rangeSTD #Dixons's measure No. 2
                            elif mta == 'dixon3':
                                measures_range = measures_rangeACC**2+measures_rangeSTD**2 #Dixons's measure No. 3
                            else:
                                #Jeff's measure                        
                                std_thres = np.nanmin(measures_rangeSTD)*5
                                measures_rangeACC[measures_rangeSTD>std_thres] = np.nan
                                measures_range = measures_rangeACC
                            
#                            try:
                            if np.sum(np.isfinite(measures_range)) > 0: #handle all-nan slice
                                IND = analysis_range[np.nanargmin(measures_range)]
                                _IND = np.nanargmin(measures_range)
                                
#                                    IND_END = DATA['time'][IND]+window_analyse
                                
                                export_range = (DATA['time'] >= DATA['time'][IND]) \
                                             & (DATA['time'] <= DATA['time'][IND]+window_analyse)
                                
#                                    DATA[measure_to_add][IND:IND+win_size]=1
                                export_range=np.squeeze(np.argwhere(export_range==True))
                                
                                #Debuging                                
#                                if DATA['trial_id'][IND]==3:
#                                    STOP
                                ###            
                                
                                DATA[measure_to_add][export_range]=1
                                

                                _key = '_'.join((eye, 'gaze', 'ind'))
                                stim[_key][stim_ind] = IND
                                for key in ('ACC', 'RMS', 'STD', 'fix_x', 'fix_y'):
                                    _key = '_'.join((eye, 'gaze', key))
                                    stim[_key][stim_ind] = measures[_key][IND]
                                
                                #px2deg. TODO: add STD
                                for key in ('ACC', 'RMS'):
                                    _key_gaze = '_'.join((eye, 'gaze', key))
                                    _key_angle = '_'.join((eye, 'angle', key))
                                    stim[_key_angle][stim_ind] = stim[_key_gaze][stim_ind]/stim['px2deg'][stim_ind]
                            
#                            except:
#                                print 'Something wrong:)', DATA['subject_id'][0], stim_ind
            
                stim_export.extend(stim.tolist())
    
            print 'Analysis time:', time.time()-t1
    
    #Dixons measure
    #plt.figure()
    #plt.plot(measures_rangeACC, measures_rangeSTD)
    #plt.plot(measures_rangeACC[_IND], measures_rangeSTD[_IND], 'ro')
    
    #stim = np.array(stim_export, dtype=stim_dtype)
    #        np.save(file_path, DATA)
            
            t1 = time.time()
    #        row_names = DATA.dtype.names
    #        header_line = '\t'.join(row_names) + '\n'
    #        
    #        col_count = len(row_names)
    #        format_str = "{}\t" * col_count
    #        format_str = format_str[:-1] + "\n"
    #                
    #        txtf = open(file_path[:-4]+'_TD.txt', 'w')
    #        txtf.write(header_line)
    #        for s in DATA:
    #            txtf.write(format_str.format(*s))
    #        txtf.close()
            WIN_SELECT_OUTPUT = '{root}/{et_model}/{et_model}_win_select/'.format(root=INPUT_FILE_ROOT, et_model=et_model)
            if not os.path.exists(WIN_SELECT_OUTPUT):
                os.mkdir(WIN_SELECT_OUTPUT)
            
            fname = file_path.rsplit(os.path.sep)[-1][:-4]
            save_path = '{output_dir}/{fname}_win_select'.format(output_dir=WIN_SELECT_OUTPUT, fname=fname)
            
            if SAVE_NPY:
                np.save(save_path, DATA)
            if SAVE_TXT:
                np.savetxt(save_path+'.txt', DATA, fmt='%s', delimiter='\t', header='\t'.join(DATA.dtype.names))
            
            print 'Save time: ', time.time()-t1
        
        except:
            with open('errors.info', 'a') as _f:
                _f.write(file_path+'\n')
                print 'Something wrong:)', DATA['subject_id'][0]
        
        
#    np.save(et_model+'.npy', np.array(stim_export, dtype=stim_dtype))
#    np.savetxt(et_model+'.txt', np.array(stim_export, dtype=stim_dtype), fmt='%s', delimiter='\t', header='\t'.join(stim_dtype.names))
    
                
sys.exit()

