# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 10:00:41 2017

@author: Jordan
Results analysis Nov 3
"""
from __future__ import division
import numpy as np
import glob, os

path = '../data/nov3/'
filename = 'forward2.csv'
coordPrefix = 'coords_'

timeDeltaThreshold = 0.1
IRFullPath = path + filename
coordFullPath = path + coordPrefix + filename
distChestTable = 18 # distance from chest to marker, in cm
scaleCameraTable = 73.298 / 5.0 # Calibration, pixels / cm

filelist = []
for file in glob.glob(path+'*.csv'):
	if os.path.basename(file) == 'coords_*.csv':
		# Do nothing, don't import this file
		print 'readme.txt detected, skipped'
	else:
		filelist.append(file)


def loadBothFiles(IRFullPath,coordFullPath):

	IR = np.genfromtxt(IRFullPath,delimiter=',')
	coord = np.genfromtxt(coordFullPath,delimiter=',')
	
	timeArduino = (IR[-1,1] - IR[0,1]) / 1000
	timeDevice = (IR[-1,-1] - IR[0,-1])
	timeCoord = coord[-1,0] - coord[0,0]
	
	print 'device times are', timeArduino, timeDevice, timeCoord
	
	coordTimes = coord[:,0]
	deviceTimes = IR[:,-1]
	
	if np.abs(deviceTimes[0] - coordTimes[0]) < timeDeltaThreshold:
		print 'start time is nearly synchronized'
		if len(coordTimes) > len(deviceTimes):
			print 'webcam length longer. will be truncated'
			coordTimesClipped = coordTimes[0:len(deviceTimes)]
			coordClipped = coord[0:len(deviceTimes)]
	
	timeDelta = coordTimesClipped - deviceTimes# Check overall time error between the two
	print 'time delta max, median, mean are',np.max(timeDelta), np.median(timeDelta), np.mean(timeDelta)
	return IR, coordClipped
# Main Loop
	
IR, coord = loadBothFiles(IRFullPath,coordFullPath) # Load the coord and IR data and combine together

# Spatial compensation
