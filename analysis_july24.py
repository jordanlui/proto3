# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:40:28 2017

@author: Jordan
Analysis on IR Results
June 24 data recorded by Kevin Andrews


"""
from __future__ import division
from regression_IRBand import model, model_multi, LOOCV, singleRun
import numpy as np
import glob

segment = 0.70 # Section of data that we train on
seed = 0
number_randomize = 2 # Number of times we want to random shuffle
seeds = range(0,number_randomize)
scale_table = 428/500 # Table scaling in pixels/mm. Note this calibration of active area square, but should be consistent across the entire camera frame.
table_width = 500 # Table width in mm (test table)
select_sensors = '111111111'
select_sensors = '100011000'

allfiles = []

#%% Multi Analysis
#path = ['../Data/june23/1/','../Data/june23/2/','../Data/june23/3/','../Data/june23/4/','../Data/june23/5/','../Data/june23/6/','../Data/june23/7/']
#path = ['../Data/july24/A - Neutral Hand Position/', '../Data/july24/B - Pronation, 45 Degrees/', '../Data/july24/C - Supination, 45 Degrees/']
#for apath in path:
#
#    allfiles =  allfiles + (glob.glob(apath+'*.csv'))
    
#%% Single Run Analysis
#path = ['../Data/july24/A - Neutral Hand Position/']
jun17 = glob.glob('../Data/july17/*')
jun24 = glob.glob('../Data/july24/*')
allpaths = jun17 + jun24
#path = path[0]
#path = '../Data/july24/B - Pronation, 45 Degrees/'
#path = '../Data/july24/C - Supination, 45 Degrees/'
#allfiles =  (glob.glob(path+'*.csv'))
#singlefile = filelist[0]
#singleRun(filelist)
#test = load(singlefile)
#error, var = singleRun(allfiles,segment,seed,scale_table)


#%% Model LOOCV
## Run model in LOOCV config
for path in allpaths:
	allfiles =  (glob.glob(path+'/*.csv'))
	error,var = LOOCV(allfiles,seed,scale_table,select_sensors)

	print 'Mean error was %.3f mm, Error relative to table dimensions is %.3f ' % (np.mean(error), np.mean(error)/(table_width*scale_table))
