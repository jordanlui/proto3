# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 10:00:41 2017

@author: Jordan
Results analysis Nov 3
Basic data loading of this data, checking timestamps agree between coord file 
and proto data, and saving to file.
Some basic plotting is done as well to look for correlations between sensor data
and movements.
"""
from __future__ import division
import numpy as np
import glob, os
import matplotlib.pyplot as plt
from sklearn.svm import SVR

print(__doc__)
path = '../data/nov3/'
filename = 'forward3.csv'
coordPrefix = 'coords_'
IRFullPath = path + filename
coordFullPath = path + coordPrefix + filename

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

def svmRegression(X,y):
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)
    
    # #############################################################################
    # Look at the results
    lw = 2
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    return y_rbf, y_lin, y_poly
    

#%% Main Loop
	
IR, coord = loadBothFiles(IRFullPath,coordFullPath) # Load the coord and IR data and combine together
distance = coord[:,-1]
sharp = IR[:,2:6]
acc = IR[:,6:9]
gyr = IR[:,9:12]
omron = IR[:,12:28]
time = IR[:,-1] # Time values since 1901
time = time - min(time) # Relative time values in seconds
omronVert = [] # Array of Omron columns, from distal to proximal
for i in [0,4,8,12]:
	omronVert.append(omron[:,i:i+4])
omronHoriz = []
for i in range(0,4):
	omronHoriz.append(omron[:,(i,i+4,i+8,i+12)])
	#x = np.hstack((IR,coord)) # All values
#%% Plot Sensor values with time

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

#%% Spatial coordinate plot
fig5 = plt.figure(5)

plt.scatter(coord[:,1],coord[:,2], marker='.')
plt.scatter(coord[:,3],coord[:,4], marker = '^')
plt.title('Coordinates of markers',fontsize=16)

#%% Sharp IR Correlation Plots
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

#%% Omron Correlation Plots
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

#%% Save to file
#pathOut = '../Analysis/nov3/swing/'
#t = distance
#x = IR[:,2:-1]
#np.savetxt(pathOut + 't3.csv',t,delimiter=',')
#np.savetxt(pathOut + 'x3.csv',x,delimiter=',')

#%% Machine Learning analysis

#y_rbf, y_lin, y_poly = svmRegression(x,t)
