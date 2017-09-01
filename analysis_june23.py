# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:40:28 2017

@author: Jordan
Analysis on IR Results
June 23 data recorded by Kevin Andrews


"""
from __future__ import division
from regression_IRBand import randomize_data, split_xt, segment_data, prep_model, model, model_multi, LOOCV, singleRun
#import jlregression
import glob

segment = 0.70 # Section of data that we train on
seed = 0
number_randomize = 2 # Number of times we want to random shuffle
seeds = range(0,number_randomize)
scale_table = 428/500 # Table scaling in pixels/mm. Note this calibration of active area square, but should be consistent across the entire camera frame.
table_width = 500 # Table width in mm (test table)

#%% Prepare for multi file analysis
path = ['../Data/june23/1/','../Data/june23/2/','../Data/june23/3/','../Data/june23/4/','../Data/june23/5/','../Data/june23/6/','../Data/june23/7/']
allfiles = []

for apath in path:

    allfiles =  allfiles + (glob.glob(apath+'*.csv'))
    
#%% Single Run Analysis
#path = ['../Data/june23/1/']
#path = path[0]
#allfiles =  (glob.glob(path+'*.csv'))

#%% Model LOOCV
## Run model in LOOCV config
error,var = LOOCV(allfiles,seed,scale_table)
#error,var = singleRun(allfiles,segment,seed,scale_table) 

#%% LOOCV Result values

#print 'Variances: ', var
#print 'MSE:', error
print 'Mean error was %.3f mm, Error relative to table dimensions is %.3f ' % (np.mean(error), np.mean(error)/(table_width*scale_table))
