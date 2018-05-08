# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 14:06:24 2018

@author: Jordan

Parse .txt files from LabVIEW containing IR Band data and ART mocap data.
Can also parse ART mocap .DRF data.
"""

from parseData import parseSave
import glob
#import numpy as np
#import matplotlib.pyplot as plt


#%% Grab files

path = '../Data/apr30/'
files = glob.glob(path+'log*.txt')
#files = glob.glob(path+'putty*.log')
#files = glob.glob(path+'*.drf')
#parseSave(files[3])

#%% Execute
#afile = files[1]
#parseSave(afile)
for afile in files:
	parseSave(afile)