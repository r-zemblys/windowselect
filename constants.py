__author__ = 'Sol'
__version__ = 'RZ'
import numpy as np

et_nan_values = dict()
et_nan_values['eyefollower'] = {'x': 0.0, 'y': 0.0}
et_nan_values['eyelink'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['eyetribe'] = {'x': -840.0, 'y': 525.0}
et_nan_values['hispeed1250'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['hispeed240'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['red250'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['red500'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['redm'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['t60xl'] = {'x': -1.0, 'y': -1.0}
et_nan_values['tx300'] = {'x': -1.0, 'y': -1.0}
et_nan_values['x2'] = {'x': -1.0, 'y': -1.0}
et_nan_values['dpi'] = {'x': -10000, 'y': -10000}
   
stim_dtype = np.dtype([

    ('eyetracker_model', str, 32),
    ('eyetracker_sampling_rate', np.float32),
    ('eyetracker_mode', str, 16),
    ('px2deg', np.float32), 
    ('operator', str, 8),

    ('subject_id', np.uint8),
    ('trial_id', np.uint16),
    ('ROW_INDEX', np.uint8),
    ('dt', np.float32),
    ('TRIAL_START', np.float32),
    ('TRIAL_END', np.float32),
    ('posx', np.float32),
    ('posy', np.float32),
    ('target_angle_x', np.float32),
    ('target_angle_y', np.float32),
    
    ('wsa', str, 32),
    ('win_size', np.float32),
    ('window_skip', np.float32),
    
    ('invalid_sample_count', np.float32),

    ('left_gaze_ind', np.float32),
    ('left_window_onset', np.float32),
    ('left_sample_count', np.float32),
    ('left_actual_win_size', np.float32),
    ('left_gaze_ACC', np.float32),
    ('left_gaze_ACC_x', np.float32),
    ('left_gaze_ACC_y', np.float32),
    ('left_gaze_RMS', np.float32),
    ('left_gaze_RMS_x', np.float32),
    ('left_gaze_RMS_y', np.float32),
    ('left_gaze_STD', np.float32),
    ('left_gaze_STD_x', np.float32),
    ('left_gaze_STD_y', np.float32),
    ('left_gaze_fix_x', np.float32),
    ('left_gaze_fix_y', np.float32),
    ('left_angle_ind', np.float32),
    ('left_angle_ACC', np.float32),
    ('left_angle_ACC_x', np.float32),
    ('left_angle_ACC_y', np.float32),
    ('left_angle_RMS', np.float32),
    ('left_angle_RMS_x', np.float32),
    ('left_angle_RMS_y', np.float32),
    ('left_angle_STD', np.float32),
    ('left_angle_STD_x', np.float32),
    ('left_angle_STD_y', np.float32),
    ('left_angle_fix_x', np.float32),
    ('left_angle_fix_y', np.float32),
    

    ('right_gaze_ind', np.float32),
    ('right_window_onset', np.float32),
    ('right_sample_count', np.float32),
    ('right_actual_win_size', np.float32),
    ('right_gaze_ACC', np.float32),
    ('right_gaze_ACC_x', np.float32),
    ('right_gaze_ACC_y', np.float32),
    ('right_gaze_RMS', np.float32),
    ('right_gaze_RMS_x', np.float32),
    ('right_gaze_RMS_y', np.float32),
    ('right_gaze_STD', np.float32),
    ('right_gaze_STD_x', np.float32),
    ('right_gaze_STD_y', np.float32),
    ('right_gaze_fix_x', np.float32),
    ('right_gaze_fix_y', np.float32),
    ('right_angle_ind', np.float32),
    ('right_angle_ACC', np.float32),
    ('right_angle_ACC_x', np.float32),
    ('right_angle_ACC_y', np.float32),
    ('right_angle_RMS', np.float32),
    ('right_angle_RMS_x', np.float32),
    ('right_angle_RMS_y', np.float32),
    ('right_angle_STD', np.float32),
    ('right_angle_STD_x', np.float32),
    ('right_angle_STD_y', np.float32),
    ('right_angle_fix_x', np.float32),
    ('right_angle_fix_y', np.float32),
    ])
    
stim_pos_mappings=dict([
    ('angle','target_angle_'),
    ('gaze', 'pos'),
    ])
    
