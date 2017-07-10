# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 15:48:57 2017

@author: Jordan

Data workspace loader
Loads the relevant x matrix and labels into memory for analysis

Goal: Load specific file paths into the x matrix.
"""
import numpy as np

def load(path = '../Data/june23/analysis/'):
    # Load data from july analysis
#    path = '../Data/june23/analysis/'
    
    x = np.genfromtxt(path+'x.csv',delimiter=',') # Load x matrix
    xx = np.genfromtxt(path+'xx.csv',delimiter=',') # Load xx matrix (contains patient,class,trial,data)
    return x,xx