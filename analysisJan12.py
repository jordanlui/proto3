# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 19:33:04 2018

@author: Jordan
"""

from __future__ import division
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob, os
from datetime import datetime
from scipy import signal
from scipy.signal import filtfilt

#%% Runtime parameters
path = ('../analysis/jan12/')
files = glob.glob(path + 'log*.csv')
#afile = files[12]
#%% Functions
def loadHybridData(data):
	# Function for loading the ART and proto data together
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

def evalModel(X_train,y_train,X_test,y_test,C,gamma,epsilon):
	clf = SVR(kernel='rbf',C=C, gamma=gamma, epsilon = epsilon, max_iter=-1, shrinking=True, tol=0.001)
#	clf = SVR(kernel='rbf',C=C, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
#	clf = SVR(kernel='linear',C=C, gamma=gamma, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
#	clf = SVR(kernel='poly',C=C, gamma=gamma, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
	clf.fit(X_train,y_train)
	print 'R2 fit is ',clf.score(X_test,y_test)
	y_pred = clf.predict(X_test)
	if normyparam: # If we used normalization, we can transform back to real units
		y_pred = (y_pred *(normyparam[1] - normyparam[2]) + normyparam[0])
		y_test = (y_test *(normyparam[1] - normyparam[2]) + normyparam[0])
	error = np.sqrt((y_pred - y_test)**2) # Error calculation
	errorRel = []
	for y in y_test:
		if y != 0 :
#			errorRel.append(error/y) # Relative error based on true value
			errorRel.append(error/(np.max(y_test) - np.min(y_test)))
	
	errorMean = np.nanmean(error)
	errorRelative = (np.nanmean(errorRel)) * 100
	print 'Mean absolute error is %.2f mm'%errorMean
	print 'Mean relative error is %.2f %%'%errorRelative
	return y_pred, errorRelative, error

def SVR_Optimize(X_train,y_train):
	print("Optimize system for best C, gamma values")
	C = [10**i for i in range(-2,0)]
	gamma = C
	epsilon= C
#	C = [0.01,0.1,1,10,100]
#	gamma = [0.01,0.1,1,10,100]
#	epsilon= [0.01,0.1,1,10,100]
	performance = [[],[]]
	for c in C:
		for g in gamma:
			for e in epsilon:
				y_pred, errorRelative, error = evalModel(X_train,y_train,X_test,y_test,c,g,e)
				performance[0].append(errorRelative)
				outputString = 'C=%.5f, gamma=%.5f, epsilon=%.4f'%(c,g,e)
				performance[1].append(outputString)
	return performance
	
def SVR_OptimizeGridSearch(X_train,y_train):
	print("Optimize system for best C, gamma values")

	#param_grid = {'C': [1, 10, 100, 1e3],
	#			  'gamma': [0.0001, 0.0005],
	#				'epsilon': [0.01, 0.1, 1, 10]}
	param_grid = {'C': [10**i for i in range(-5,3)],
	#			  'gamma': [0.0001, 0.0005],
					'epsilon': [10**i for i in range(-5,3)]}
	clf = GridSearchCV(SVR(verbose=True), param_grid, cv=3)
	clf.fit(X_train, y_train)
	#print("done in %0.3fs" % (time() - t0))
	print("Best estimator found by grid search:")
	print(clf.best_estimator_)
	print(clf.best_params_)
	return clf

def bpf(data,high,low,order,freq):
	plt.subplot(211)
	plt.plot(data)
	pltTitle = 'BPF: Order%i, HPF=%.2f Hz, LPF=%.2f'%(order,high,low)
	plt.title(pltTitle)
	freq = float(freq)
	[b, a] = signal.butter(order, [high*2/freq],btype='highpass')
	dataFilt = filtfilt(b,a,data,axis=0)
	
	b, a = signal.butter(order, [low*2/freq],btype='lowpass')
	dataFilt = filtfilt(b,a,data,axis=0)
	plt.subplot(212)
	plt.plot(dataFilt)
	return dataFilt
	
def hpf(data,high,order, freq):
	plt.subplot(211)
	plt.plot(data)
	freq = float(freq)
	pltTitle = 'HPF: Order%i, cutoff=%.2f Hz'%(order,high)
	plt.title(pltTitle)
	b, a = signal.butter(order, [high*2/freq],btype='highpass')
	dataFilt = filtfilt(b,a,data,axis=0)
	
	
	plt.subplot(212)
	plt.plot(dataFilt)
	return dataFilt
	
def smoothQuaternion(quat,tol=1e4):
	plt.subplot(211)
	plt.plot(quat)
	for i in range(len(quat)-1):
		row2 = quat[i+1,:]
		row1 = quat[i,:]
		if np.sum(np.abs(row2 - row1)) > tol:
			quat[i+1,:] = quat[i,:]
	plt.subplot(212)
	plt.plot(quat)
	return quat
#%% Main
high = 0.1 # HPF filt freq, in Hz
order = 5
low = 5
freq = 60

## Reach Data
## Train data
#afile = files[10]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.35*len(data)):int(0.80*len(data)),:]

#
## Test Data
#afile = files[11]
#data2 = np.genfromtxt(afile,delimiter=',')
#data2 = data2[int(0.30*len(data2)):int(0.75*len(data2)),:]


## Raised Elbow Flex Data
## Train data
#afile = files[5]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.25*len(data)):int(0.94*len(data)),:]
#
## Test Data
#afile = files[6]
#data2 = np.genfromtxt(afile,delimiter=',')
#data2 = data2[int(0.2*len(data2)):int(0.90*len(data2)),:]

#
## Circle square movements
## Train data
#afile = files[13]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.25*len(data)):int(0.94*len(data)),:]
##X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)
#
## Test Data
#afile = files[14]
#data2 = np.genfromtxt(afile,delimiter=',')
#data2 = data2[int(0.2*len(data2)):int(0.90*len(data2)),:]


# Shoulder Raising
# Train data
afile = files[3]
data = np.genfromtxt(afile,delimiter=',')
data = data[int(0.25*len(data)):int(0.92*len(data)),:]
#X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)


# Test Data
afile = files[4]
data2 = np.genfromtxt(afile,delimiter=',')
data2 = data2[int(0.2*len(data2)):int(0.90*len(data2)),:]
#X_test,y_test,distances2,time2,omron2,acc2,gyr2,quat2,position2,rotation2 = loadHybridData(data)

# Filter
data[:,17:23] = hpf(data[:,17:23],high, order, freq) # acc and gyro filtering
data2[:,17:23] = hpf(data2[:,17:23],high, order, freq) # acc and gyro filtering
#data[:,23:27] = smoothQuaternion(data[:,23:27],tol=1e4) # Quaternion
#data2[:,23:27] = smoothQuaternion(data2[:,23:27],tol=1e4) # Quaternion

# Parse out data
X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)
X_test,y_test,distances2,time2,omron2,acc2,gyr2,quat2,position2,rotation2 = loadHybridData(data2)

#%% Filter testing
#high = 0.1 # HPF filt freq, in Hz
#order = 5
#low = 5
#plt.subplot(211)
#plt.plot(acc)
#pltTitle = 'HPF: Order%i, cutoff=%.2f Hz'%(order,high)
#plt.title(pltTitle)
#b, a = signal.butter(order, [high*2/freq],btype='highpass')
#accFilt = filtfilt(b,a,acc,axis=0)
#
#
#plt.subplot(212)
#plt.plot(accFilt)
##plt.title(pltTitle)
#hpf(acc,high,order,freq)
#hpf(gyr,high,order,freq)
##bpf(quat,high,1,freq,9)
#%%
#import numpy.fft as fft

#from scipy.fftpack import fft
##fftIN = hpf(gyr,high,order,freq)
#fftIN = bpf(quat,high,10,freq,9)
#
#def fftplot(fftIN,freq,title='FFT plot'):
#	# FFT plot of columns of data
#	plt.figure()
#	for i in range(fftIN.shape[1]):
#		data = fftIN[:,i]
#		freq = float(freq)
#		yf = fft(data)
#		N = len(yf)
#		T = 1 / float(freq)
#		xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
#		plt.semilogy(xf,2.0/N*np.abs(yf[0:N/2]))
#	plt.title(title)
#	return plt
#fftplot(fftIN,freq,title='FFT quaternion')
#%% Normalize train and test data with train data stats
normXparam = [np.mean(X_train,axis=0), np.max(X_train,axis=0), np.mean(X_train,axis=0)]
normyparam = [np.mean(y_train), np.max(y_train), np.min(y_train)]
X_test = (X_test - np.mean(X_train,axis=0))/(np.max(X_train,axis=0) - np.min(X_train,axis=0))
y_test = (y_test - np.mean(y_train))/(np.max(y_train) - np.min(y_train))
X_train = (X_train - np.mean(X_train,axis=0))/(np.max(X_train,axis=0) - np.min(X_train,axis=0))
y_train = (y_train - np.mean(y_train))/(np.max(y_train) - np.min(y_train))


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

plt.figure()
plt.plot(distances[0],label='Wrist to upper arm')
plt.plot(distances2[0],label='Wrist to upper arm (file2)')
plt.title('Distances')
plt.ylabel('Distance (mm)')
plt.legend()

plt.figure()
plt.plot(time,distances[0],label='Wrist to upper arm')
for i in range(omron.shape[1]):
	plt.plot(time,omron[:,i])
plt.title('Distances and Thermal data')
plt.ylabel('Distance (mm)')
plt.legend()

plt.figure()
plt.plot(time,distances[0],label='Wrist to upper arm')
plt.plot(time,np.mean(omron,axis=1),label='Omron Average')
plt.title('Distances and average Thermal data')
plt.ylabel('Distance (mm)')
plt.legend()


fig, ax1 = plt.subplots()
ax1.plot(distances[0],label='Wrist to upper arm')
ax1.plot(distances2[0],label='Wrist to upper arm')
ax2 = ax1.twinx()
ax2.plot(np.mean(omron,axis=1),'b.',label='Omron Average')
ax2.plot(np.mean(omron2,axis=1),'g.',label='Omron Average')
plt.title('Distances and average Thermal data')
plt.ylabel('Distance (mm)')
ax1.legend(bbox_to_anchor=(0., 1.3, 0.5, .102))
ax2.legend(bbox_to_anchor=(0., 1.3, 1., .102))

#%% Model prediction on wrist distance data
#clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)

C = 0.01
gamma = 0.1
epsilon = 0.01
#performance = SVR_Optimize(X_train,y_train)
y_pred, errorRelative, error = evalModel(X_train,y_train,X_test,y_test,C,gamma,epsilon)

#%% Results Plotting

y_testmm = (y_test *(normyparam[1] - normyparam[2]) + normyparam[0])
#y_pred = (y_pred *(normyparam[1] - normyparam[2]) + normyparam[0])
#error = (error *(normyparam[1] - normyparam[2]) + normyparam[0])

plt.figure()
pltTitle = 'Relative Error %.2f%%. Predictions for C=%.3f, g=%.3f, e=%.2f'%(errorRelative,C,gamma,epsilon)
plt.scatter(y_testmm,y_pred)
plt.title(pltTitle)
plt.ylabel('Predicted Distance (mm)')
plt.xlabel('Real Distance (mm)')
plt.legend()
#plt.ylim((np.min((y_test,y_pred)))/plotPadding,np.max((y_test,y_pred)) * plotPadding)
#plt.xlim(np.min((y_test,y_pred))/plotPadding,np.max((y_test,y_pred)) * plotPadding)


plt.figure()
plt.hist(error)
plt.title('Histogram of error')
plt.ylabel('Occurences')
plt.xlabel('Error (mm)')

print 'movement span (mm) was in each axis was ',(np.max(distances2,axis=1) - np.min(distances2,axis=1))


#%%
