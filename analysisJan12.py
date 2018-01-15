# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 19:33:04 2018

@author: Jordan
"""

from __future__ import division
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob, os
from datetime import datetime

#%% Runtime parameters
path = ('../analysis/jan12/')
files = glob.glob(path + 'log*.csv')
#afile = files[12]
#%% Functions
def loadHybridData(data):
	time = data[:,0]
	omron = data[:,1:17]
	acc= data[:,17:20]
	gyr = data[:,20:23]
	quat = data[:,23:27]
	position = data[:,27:36]
	rotation = data[:,36:]
	
	# Create Secondary Features
	distances = [0 for i in range(3)]
	distances[0] = np.sqrt((position[:,0]-position[:,6])**2 + (position[:,1]-position[:,7])**2 + (position[:,2]-position[:,8])**2) # wrist to arm
	distances[1] = np.sqrt((position[:,3]-position[:,6])**2 + (position[:,4]-position[:,7])**2 + (position[:,5]-position[:,8])**2) # forearm to wrist
	distances[2] = np.sqrt((position[:,0]-position[:,3])**2 + (position[:,1]-position[:,4])**2 + (position[:,5]-position[:,8])**2) # upper arm to forearm
	
	y = distances[0]
	X = np.hstack((omron,acc,gyr,quat))
#	X = np.hstack((omron,acc))
	return X,y,distances,time,omron,acc,gyr,quat,position,rotation
#	return X,y,distances

def evalModel(X_train,y_train,X_test,y_test,C,gamma):
	clf = SVR(kernel='rbf',C=C, gamma=gamma, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	y_pred = clf.predict(X_test)
	error = np.sqrt((y_pred - y_test)**2)
	errorRel = []
	for y in y_test:
		if y != 0 :
			errorRel.append(error/y)
	
	errorMean = np.mean(error)
	errorRelative = (np.nanmean(errorRel)) * 100
	print 'absolute error is %.2f mm'%errorMean
	print 'overall relative error is %.2f %%'%errorRelative
	return y_pred, errorRelative, error

def SVR_Optimize(X_train,y_train):
	print("Optimize system for best C, gamma values")
#	t0 = datetime.now()
	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
				  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	clf = GridSearchCV(SVR(kernel='rbf'), param_grid)
#	clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
#	clf = clf.fit(X_train, y_train)
#	print("done in %0.3fs" % (datetime.now() - t0))
#	print("Best estimator found by grid search:")
	print(clf.get_params())
	return clf

#%% Main

# Reach Data
# Train data
afile = files[10]
data = np.genfromtxt(afile,delimiter=',')
data = data[int(0.25*len(data)):int(0.94*len(data)),:]
X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Test Data
afile = files[11]
data = np.genfromtxt(afile,delimiter=',')
data = data[int(0.2*len(data)):int(0.95*len(data)),:]
X_test,y_test,distances2,time2,omron2,acc2,gyr2,quat2,position2,rotation2 = loadHybridData(data)

## Raised Elbow Flex Data
## Train data
#afile = files[5]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.25*len(data)):int(0.94*len(data)),:]
#X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
#
## Test Data
#afile = files[6]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.2*len(data)):int(0.90*len(data)),:]
#X_test,y_test,distances2,time2,omron2,acc2,gyr2,quat2,position2,rotation2 = loadHybridData(data)
#
## Circle square movements
## Train data
#afile = files[13]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.25*len(data)):int(0.94*len(data)),:]
#X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
#
## Test Data
#afile = files[14]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.2*len(data)):int(0.90*len(data)),:]
#X_test,y_test,distances2,time2,omron2,acc2,gyr2,quat2,position2,rotation2 = loadHybridData(data)

## Shoulder Raising
## Train data
#afile = files[3]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.25*len(data)):int(0.92*len(data)),:]
#X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
#
## Test Data
#afile = files[4]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.2*len(data)):int(0.90*len(data)),:]
#X_test,y_test,distances2,time2,omron2,acc2,gyr2,quat2,position2,rotation2 = loadHybridData(data)

#%% Some Plots
#plt.figure()
#plt.plot(time,omron/10)
#plt.title('Omron Signal')
#plt.ylabel('Temperature (C)')

plt.figure()
plt.plot(time,distances[0],label='Wrist to upper arm')
plt.plot(time,distances[1],label='Forearm to wrist')
plt.plot(time,distances[2],label='Upper Arm to forearm')
plt.title('Distances')
plt.ylabel('Distance (mm)')
plt.legend()

plt.figure()
plt.plot(time2,distances2[0],label='Wrist to upper arm (file2)')
plt.plot(time2,distances2[1],label='Forearm to wrist')
plt.plot(time2,distances2[2],label='Upper Arm to forearm')
plt.title('Distances')
plt.ylabel('Distance (mm)')
plt.legend()


#%% Model prediction on wrist distance data
#clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
C = 1000
gamma = 'auto'
y_pred, errorRelative, error = evalModel(X_train,y_train,X_test,y_test,C,gamma)
plt.figure()
plt.scatter(y_test,y_pred)
plt.title('Distances')
plt.ylabel('Distance (mm)')
plt.legend()
plt.ylim(min(y_test),max(y_test))

plt.figure()
plt.hist(error)
plt.title('histogram of error')
plt.ylabel('Occurences')
plt.xlabel('Error (mm)')

print 'movement span was',(np.max(distances2,axis=1) - np.min(distances2,axis=1))
