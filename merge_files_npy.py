# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 19:25:47 2014

@author: huml-rzm
"""

import os, glob

import numpy as np

DATA_ROOT = r"/media/Data/EDQ/data_npy/"
ET_FOLDERS = ['eyefollower', 'eyelink', 'eyetribe', 'hispeed1250', 'hispeed240',
              'red250', 'red500', 'redm', 't60xl', 'tx300', 'x2']

ET_FOLDERS = ['hispeed1250']

SAVE_TXT = True
SAVE_NPY = True
SAVE_TXT_INDV = False

def nabs(file_path):
    return os.path.normcase(os.path.normpath(os.path.abspath(file_path)))

for et_name in ET_FOLDERS:
    et_npy_files = glob.glob(nabs('{root}/{tracker}/{tracker}_win_select/{tracker}*.npy'.format(root=os.path.join(DATA_ROOT), tracker=et_name)))
    for n, filename in enumerate(et_npy_files):
        DATA_TMP = np.load(filename)
        print '{n} out of {total} processed'.format(n=n+1, total=len(et_npy_files))
        if SAVE_TXT_INDV:
            print 'Saving individual txt...'
            np.savetxt(filename[:-4]+'.txt', DATA_TMP, fmt='%s', delimiter='\t', header='\t'.join(DATA_TMP.dtype.names))
        try:
            DATA = np.hstack((DATA, DATA_TMP))
        except:
            DATA = np.copy(DATA_TMP)
    if SAVE_NPY:
        print 'Saving npy...'
        np.save(et_name+'_all', DATA) 
    if SAVE_TXT:
        print 'Saving txt...'
        np.savetxt(et_name+'_all.txt', DATA, fmt='%s', delimiter='\t', header='\t'.join(DATA.dtype.names))
    del DATA
