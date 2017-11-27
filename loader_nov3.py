# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 10:00:41 2017

@author: Jordan
File Loading Nov 3
Basic data loading of this data, checking timestamps agree between coord file 
and proto data, and saving to file.
Some basic plotting is done as well to look for correlations between sensor data
and movements.

"""
from __future__ import division
import numpy as np
import glob, os
import matplotlib.pyplot as plt
from scipy import stats
import re

print(__doc__)

#%% Run parameters - change these to target different files
path = '../data/nov3/'
filename = 'swing3.csv'

#%% Calculated parameters
folderOut = re.findall('([a-z]+)[0-9]*\.csv',filename)[0]

try:
	num = re.findall('[a-z]+([0-9])+\.csv',filename)[0]
except:
	num = 1
		

coordPrefix = 'coords_'
#IRFullPath = path + filename
#coordFullPath = path + coordPrefix + filename

timeDeltaThreshold = 0.1
distChestTable = 18 # distance from chest to marker, in cm
scaleCameraTable = 73.298 / 5.0 # Calibration, pixels / cm

filelist = []

#%% Functions
for file in glob.glob(path+'*.csv'):
	if os.path.basename(file) == 'coords_*.csv':
		# Do nothing, don't import this file
		print 'readme.txt detected, skipped'
	else:
		filelist.append(file)
file = 0

def loadBothFiles(IRFullPath,coordFullPath):

	IR = np.genfromtxt(IRFullPath,delimiter=',')
	coord = np.genfromtxt(coordFullPath,delimiter=',')
	
	timeArduino = (IR[-1,1] - IR[0,1]) / 1000
	timeDevice = (IR[-1,-1] - IR[0,-1])
	timeCoord = coord[-1,0] - coord[0,0]
	
	print 'Arduino and proto record durations are', timeArduino, timeDevice, timeCoord
	
	coordTimes = coord[:,0]
	deviceTimes = IR[:,-1]
	
	if np.abs(deviceTimes[0] - coordTimes[0]) < timeDeltaThreshold: # Check time coordination
		print 'start time is nearly synchronized'
		if len(coordTimes) > len(deviceTimes):
			print 'webcam length longer. will be truncated'
			coordTimesClipped = coordTimes[0:len(deviceTimes)]
			coordClipped = coord[0:len(deviceTimes)]
	
	timeDelta = coordTimesClipped - deviceTimes# Check overall time error between the two
	print 'time delta max, median, mean are',np.max(timeDelta), np.median(timeDelta), np.mean(timeDelta)
	return IR, coordClipped

def fixAnchorPosition(coord):
	np.median(coord,axis=0)
	anchor = stats.mode((np.round(coord[:,1:5],0)))
	if anchor.mode[0][0] == anchor.mode[0][2] and anchor.mode[0][1] == anchor.mode[0][3]:
		anchorPos = (anchor.mode[0][0],anchor.mode[0][1])
		print 'Anchor located at (%i,%i), position agreed upon based on mode'%(int(anchorPos[0]),int(anchorPos[1]))
	else:
		print 'Anchor agreement not conclusive, positions at',anchor
		if anchor.count[0][0] + anchor.count[0][1] > anchor.count[0][2] + anchor.count[0][3]:
			anchorPos = (anchor.mode[0][0],anchor.mode[0][1])
		else:
			anchorPos = (anchor.mode[0][2],anchor.mode[0][3])
		
	#if np.sum(anchor.count[0:2]) > np.sum(anchor.count[2:]):
		
	
	oldCoord = coord[:,1:5]
	newCoord = np.zeros((oldCoord.shape))
	for i in range(len(oldCoord)):
		if np.sqrt( (oldCoord[i,0] - anchorPos[0])**2 + (oldCoord[i,1] - anchorPos[1])**2) < np.sqrt( (oldCoord[i,2] - anchorPos[0])**2 + (oldCoord[i,3] - anchorPos[1])**2):
			# Then Point 1 is the anchor. Don't change
			newCoord[i,:] = oldCoord[i,:]
		else:
			# Then Point 2 is anchor, and we want to swap
			newCoord[i,:] = np.array([oldCoord[0,2],oldCoord[0,3],oldCoord[0,0],oldCoord[0,1]])
			
	coord[:,1:5] = newCoord	
	return coord
def plotSensorsTime(time,distance,omron,sharp):
	fig1 = plt.figure(1)
	fig1, axarr = plt.subplots(3, sharex=True)
	axarr[0].plot(time,distance)
	axarr[0].set_title('Reach distance, cm')
	#axarr[1].plot(time,omron[:,4:12])
	axarr[1].plot(time,omron[:,:])
	axarr[1].set_title('Omron')
	axarr[2].plot(time,sharp[:,:])
	axarr[2].set_title('Sharp')
	fig1.suptitle('Sensor values with time for %s'%filename, fontsize=16)
	
def plotSpatial(coord):
	fig5 = plt.figure(5)

	plt.scatter(coord[:,1],coord[:,2], marker='.')
	plt.scatter(coord[:,3],coord[:,4], marker = '^')
	plt.title('Coordinates of markers',fontsize=16)
	
def plotSharpCorrelation(distance,sharp):
	fig2 = plt.figure(2)
	fig2, axarr = plt.subplots(4, sharex=True)
	axarr[0].scatter(distance,sharp[:,0])
	axarr[0].set_title('sharp1')
	axarr[1].scatter(distance,sharp[:,1])
	axarr[1].set_title('sharp2')
	axarr[2].scatter(distance,sharp[:,2])
	axarr[2].set_title('sharp3')
	axarr[3].scatter(distance,sharp[:,3])
	axarr[3].set_title('sharp4')
	fig2.suptitle('Sharp IR values with distance', fontsize=16)
	
def plotOmronCorrelation(distance,OmronVert,omronHoriz):
	fig3 = plt.figure(3)
	fig3, axarr = plt.subplots(4, sharex=True)
	for i in range(0,4):
		axarr[i].scatter(distance,np.mean(omronVert[i], axis = 1))
	
	fig3.suptitle('Omron columns with distance, columwise for %s'%filename, fontsize=16)
	
	fig4 = plt.figure(4)
	fig4, axarr = plt.subplots(4, sharex=True)
	for i in range(0,4):
		axarr[i].scatter(distance,np.mean(omronHoriz[i], axis = 1))
	
	fig4.suptitle('Omron rows with distance, columwise for %s'%filename, fontsize=16)
	
	fig4 = plt.figure(3)
	fig4, axarr = plt.subplots(4, sharex=True)
	for i in range(0,4):
		axarr[i].scatter(distance,np.mean(omronVert[i], axis = 1) / np.mean(omronVert[0],axis=1))
	
	fig4.suptitle('Omron columns with distance normalized to distal value, columwise for %s'%filename, fontsize=16)
	
def OmronFeatures(omron):
	omronVert = [] # Array of Omron columns, from distal to proximal
	for i in [0,4,8,12]:
		omronVert.append(omron[:,i:i+4])
	omronHoriz = []
	for i in range(0,4):
		omronHoriz.append(omron[:,(i,i+4,i+8,i+12)])
	return omronVert, omronHoriz
#%% Main Loop

def main():
	global omron, omronVert, omronHoriz
	IRFullPath = path + filename
	coordFullPath = path + coordPrefix + filename
	IR, coord = loadBothFiles(IRFullPath,coordFullPath) # Load the coord and IR data and combine together, truncate as necessary
	distance = coord[:,-1]
	sharp = IR[:,2:6]
	acc = IR[:,6:9]
	gyr = IR[:,9:12]
	omron = IR[:,12:28]
	time = IR[:,-1] # Time values since 1901
	time = time - min(time) # Relative time values in seconds


	coord = fixAnchorPosition(coord) # Fixing Coordinate jumping
	omronVert, omronHoriz = OmronFeatures(omron)
		
	plotSensorsTime(time,distance,omron,sharp) # Plot with time
	plotSpatial(coord) # Spatial coordinate plot

	plotSharpCorrelation(distance,sharp) # Sharp IR Correlation Plots
	#%% 
	plotOmronCorrelation(distance,omronVert,omronHoriz) # Omron Correlation Plots
	return omron, omronVert, omronHoriz
#%% Main Loop


main()

#%% Save to file
#pathOut = '../Analysis/nov3/' + folderOut + '/'
#t = distance
#t = np.reshape(t,(len(t),1))
#x = IR[:,2:-1]
#
#np.savetxt(pathOut + 't' + str(num) + '.csv',t,delimiter=',')
#np.savetxt(pathOut + 'x' + str(num) + '.csv',x,delimiter=',')
#np.savetxt(pathOut + 'XX' + str(num) + '.csv',np.hstack((x,t)),delimiter=',')

#%% Machine Learning analysis

#y_rbf, y_lin, y_poly = svmRegression(x,t)
