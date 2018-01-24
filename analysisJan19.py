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
plt.close("all")

#%% Runtime parameters
path = ('../analysis/jan19/')
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
	
def filesCV(files):
	# Cross validation across files
	return

def smooth(data,tol=500):
	# Smooth column of data
	# useful for distance data, which has some distance spikes
	for i in range(len(data)-1):
		row2 = data[i+1]
		row1 = data[i]
		if abs(row2-row1) > tol:
			data[i+1] = row1
	return data
def distanceCorrection(data,tol=500):
	# Threshold correction of data
	# useful for distance data, which has some distance spikes
	for i in range(len(data)-1):
		if data[i+1] > tol:
			data[i+1] = data[i] # Current decision: Freeze in place and take previous value
	return data

#%% Main
high = 0.1 # HPF filt freq, in Hz
order = 5
low = 5
freq = 60
dataTrain = []
# Reach Data
# Train data

afile = files[0]
data = np.genfromtxt(afile,delimiter=',')
data = data[int(0.3*len(data)):int(0.90*len(data)),:]
dataTrain.append(data)

afile = files[1]
data = np.genfromtxt(afile,delimiter=',')
data = data[int(0.2*len(data)):int(0.90*len(data)),:]
dataTrain.append(data)

# Test Data
afile = files[2]
data = np.genfromtxt(afile,delimiter=',')
data = data[int(0.2*len(data)):int(0.99*len(data)),:]
dataTrain.append(data)


afile = files[3]
data = np.genfromtxt(afile,delimiter=',')
data = data[int(0.23*len(data)):int(0.90*len(data)),:]
dataTrain.append(data)

## Raised Elbow Flex Data
## Train data
#afile = files[5]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.25*len(data)):int(0.94*len(data)),:]
#dataTrain.append(data)
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
#dataTrain.append(data)
# Test Data
#afile = files[14]
#data2 = np.genfromtxt(afile,delimiter=',')
#data2 = data2[int(0.2*len(data2)):int(0.90*len(data2)),:]


## Shoulder Raising
## Train data
#afile = files[3]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.25*len(data)):int(0.92*len(data)),:]
#dataTrain.append(data)
## Test Data
#afile = files[4]
#data2 = np.genfromtxt(afile,delimiter=',')
#data2 = data2[int(0.2*len(data2)):int(0.90*len(data2)),:]

## Random movements
## Train data
#afile = files[8]
#data = np.genfromtxt(afile,delimiter=',')
#data = data[int(0.25*len(data)):int(0.94*len(data)),:]
#dataTrain.append(data)
##X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)
#plt.figure()
#plt.plot(data[:,27:30])
#
## Test Data
#afile = files[9]
#data2 = np.genfromtxt(afile,delimiter=',')
#data2 = data2[int(0.2*len(data2)):int(0.90*len(data2)),:]


# Mash all training data together
#data = np.vstack((dataTrain))
data = dataTrain[1]
data2 = dataTrain[0]
# Filter
data[:,17:23] = hpf(data[:,17:23],high, order, freq) # acc and gyro filtering
data2[:,17:23] = hpf(data2[:,17:23],high, order, freq) # acc and gyro filtering
#data[:,23:27] = smoothQuaternion(data[:,23:27],tol=1e4) # Quaternion
#data2[:,23:27] = smoothQuaternion(data2[:,23:27],tol=1e4) # Quaternion

# Parse out data
X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(data)
X_test,y_test,distances2,time2,omron2,acc2,gyr2,quat2,position2,rotation2 = loadHybridData(data2)

# Physical constraints, kinematics, and anatomy
distanceTol = 450
distances = [distanceCorrection(i,tol=distanceTol) for i in distances]
distances2 = [distanceCorrection(i,tol=distanceTol) for i in distances2]

y_test = distanceCorrection(y_test,tol=distanceTol)
y_train = distanceCorrection(y_train,tol=distanceTol)


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
plt.ylabel('Distance (mm), file 1')
plt.legend()

plt.figure()
#plt.subplot(221)
plt.plot(time2,distances2[0],label='Wrist to upper arm (file2)')
plt.plot(time2,distances2[1],label='Forearm to wrist')
plt.plot(time2,distances2[2],label='Upper Arm to forearm')
plt.title('Distances')
plt.ylabel('Distance (mm), file 2')
plt.legend()

plt.figure()
#plt.subplot(222)
plt.plot(distances[0],label='Wrist to upper arm')
plt.plot(distances2[0],label='Wrist to upper arm (file2)')
plt.title('Distances')
plt.ylabel('Distance (mm)')
plt.legend()

plt.figure()
#plt.subplot(222)
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

#%% 3d plot of position data
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors=['red','blue','green']
for i in range(3):
	X = position[:,3*i]
	Y = position[:,3*i + 1]
	Z = position[:,3*i + 2]
	ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, color=colors[i])
#	ax.plot(X,Y,Z)
plt.title('path of each ART device')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#%% Model prediction on wrist distance data
#clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)

C = 0.01
gamma = 0.1
epsilon = 0.01
#performance = SVR_Optimize(X_train,y_train)
y_pred, errorRelative, error = evalModel(X_train,y_train,X_test,y_test,C,gamma,epsilon)


#%% Try Tree regression method
from sklearn import tree
modelTree = tree.DecisionTreeRegressor(max_depth=3)
modelTree = modelTree.fit(X_train,y_train)
y_predTree = modelTree.predict(X_test)
errorTree = y_predTree - y_test
modelTree.score
y_predTreemm = (y_predTree *(normyparam[1] - normyparam[2]) + normyparam[0])
y_testmm = (y_test *(normyparam[1] - normyparam[2]) + normyparam[0])
plt.figure()
plt.scatter(y_testmm,y_predTreemm)
plt.figure()
plt.plot(y_testmm,label='Test Data')
plt.plot(y_predTreemm, label='Prediction')
plt.legend()
plt.title('Prediction with tree method')
errorTree = np.sqrt((y_predTreemm - y_testmm)**2)
print 'mean error for tree method is %.2f mm'%(np.mean(errorTree))

#%% Cross Val
#accuracyResult = []
#for i in range(len(dataTrain)):
#	testData = dataTrain[i]
#	
#	testData[:,17:23] = hpf(testData[:,17:23],high, order, freq) # acc and gyro filtering	
#	X_test,y_test,distances2,time2,omron2,acc2,gyr2,quat2,position2,rotation2 = loadHybridData(testData)
#	trainingFiles = dataTrain[:i] + dataTrain[i+1:]
#	trainData = np.vstack((trainingFiles))
#	trainData[:,17:23] = hpf(trainData[:,17:23],high, order, freq) # acc and gyro filtering
#	
#	X_train,y_train,distances,time,omron,acc,gyr,quat,position,rotation = loadHybridData(trainData)
#	normXparam = [np.mean(X_train,axis=0), np.max(X_train,axis=0), np.mean(X_train,axis=0)]
#	normyparam = [np.mean(y_train), np.max(y_train), np.min(y_train)]
#	X_test = (X_test - np.mean(X_train,axis=0))/(np.max(X_train,axis=0) - np.min(X_train,axis=0))
#	y_test = (y_test - np.mean(y_train))/(np.max(y_train) - np.min(y_train))
#	X_train = (X_train - np.mean(X_train,axis=0))/(np.max(X_train,axis=0) - np.min(X_train,axis=0))
#	y_train = (y_train - np.mean(y_train))/(np.max(y_train) - np.min(y_train))
#	y_pred, errorRelative, error = evalModel(X_train,y_train,X_test,y_test,C,gamma,epsilon)
#	accuracyResult.append(errorRelative)
#	print errorRelative

#%% Results Plotting

y_testmm = (y_test *(normyparam[1] - normyparam[2]) + normyparam[0])
#y_pred = (y_pred *(normyparam[1] - normyparam[2]) + normyparam[0])
#error = (error *(normyparam[1] - normyparam[2]) + normyparam[0])
plotPadding = 1
plt.figure()
pltTitle = 'Relative Error %.2f%%. Predictions for C=%.3f, g=%.3f, e=%.2f'%(errorRelative,C,gamma,epsilon)
plt.scatter(y_testmm,y_pred)
plt.title(pltTitle)
plt.ylabel('Predicted Distance (mm)')
plt.xlabel('Real Distance (mm)')
plt.legend()
axisLim = (np.min(y_pred),np.max(y_pred))
#plt.ylim((np.min((y_testmm,y_pred)))/plotPadding,np.max((y_testmm,y_pred)) * plotPadding)
plt.xlim(axisLim)

# Remove error values beyond physical tolerances
lengthWristArm = 400 # Max length wrist to upper arm of Jordan (38-40cm)
errorCorrected = []
y_testmmCorrected = []
y_predCorrected = []
for i in range(len(error)):
	if error[i] < lengthWristArm:
		errorCorrected.append(error[i])	
		y_testmmCorrected.append(y_testmm[i])
		y_predCorrected.append(y_pred[i])
		

plt.figure()
plt.hist(error)
plt.title('Histogram of error')
plt.ylabel('Occurences')
plt.xlabel('Error (mm)')

plotPadding = 1
plt.figure()
pltTitle = 'Outliers rejected. Relative Error %.2f%%. Predictions for C=%.3f, g=%.3f, e=%.2f'%(errorRelative,C,gamma,epsilon)
plt.scatter(y_testmmCorrected,y_predCorrected)
plt.title(pltTitle)
plt.ylabel('Predicted Distance (mm)')
plt.xlabel('Real Distance (mm)')
plt.legend()
axisLim = (np.min(y_predCorrected),np.max(y_predCorrected))
#plt.ylim((np.min((y_testmm,y_pred)))/plotPadding,np.max((y_testmm,y_pred)) * plotPadding)
plt.xlim(axisLim)


plt.figure()
plt.hist(errorCorrected)
plt.title('Histogram of error, outliers rejected')
plt.ylabel('Occurences')
plt.xlabel('Error (mm)')

print 'movement span (mm) was in each axis was ',(np.max(distances2,axis=1) - np.min(distances2,axis=1))
print 'Mean error is %.2f mm',np.mean(error)
print 'Mean error (outliers removed) is %.2f mm',np.mean(errorCorrected)

#%%
