# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 08:41:09 2017

@author: moricex
"""

# Dependencies
import os
import sys
import numpy as np
#import astropy.io.fits as fits
import time
import logging
import random

run_name = '20042017_dgp_500'
run_top_path = '/mnt/lustre/moricex/MGPICOLAruns/'+ run_name
save_dir='voxelised_oldnorm_500'
os.chdir(run_top_path) # Change directory
cwd=os.getcwd()
dirs=os.listdir(cwd)

logging.basicConfig(level=logging.INFO,\
                    format='%(asctime)s %(name)-20s %(levelname)-6s %(message)s',\
                    datefmt='%d-%m-%y %H:%M',\
                    filename=run_name+'.log',\
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter) # tell the handler to use this format
logger=logging.getLogger('') # add the handler to the root logger
logger.addHandler(console)

logger.info('Program start')
logger.info('------------')
logger.info('CWD is %s' %cwd)

run_dirs = [s for s in dirs if "run_" in s]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for run_i in range(len(run_dirs)):

    #find info file
    run_info_name = [s for s in os.listdir(cwd+'/run_'+str(run_i+1)) if ".txt" in s[-4:]]
    run_info = np.genfromtxt('run_'+str(run_i+1)+'/'+run_info_name[0])
    
    data = []
    hist_data=[]
    for i in range(np.int(run_info[0])+1):
        data.append(np.genfromtxt('run_'+str(run_i+1)+'/'+run_info_name[0][5:-3]+str(i),skip_header=1))
    data=data[0][:,0:3]
    # CODE HERE FOR IF BIGGER THAN ONE DO A VSTACK    
#	name:voxelised
#    weights = np.ones_like(data[:,0:1]) / np.float64(len(data))
#    hist_data = np.histogramdd(data, bins=64.0,normed=False,weights=weights[:,0])

#	name:voxelised_oldnorm
    hist_data = np.histogramdd(data, bins=64.0,normed=True)
    print('Saving run %s' %run_i)
    np.save(save_dir+'/run_'+str(run_i+1)+'.vox',hist_data)
