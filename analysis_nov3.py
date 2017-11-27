# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 23:17:45 2017

@author: Jordan

Machine Learning Analysis on nov 3 IR data

Code base from:
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
Loads multiple files and performs varying levels of Deep learning analysis on each file, printing out results for each

"""

from __future__ import division
#from analysisFunctions import *

import numpy
import numpy as np
#import pandas

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from time import time
import logging
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from scipy import stats
import scipy

scaleCameraTable = 73.298 / 5.0 # Scale for system, pixels/cm. From Nov 3?
#import time
import glob, os
import pickle, shelve
import matplotlib.pyplot as plt

print(__doc__)

#%% Load Datasets

files = []
path = 'forward'
files.append(glob.glob(path+'/XX*.csv'))
path = 'swing'
files.append(glob.glob(path+'/XX*.csv'))
path = 'lateral'
files.append(glob.glob(path+'/XX*.csv'))
path = 'diag'
files.append(glob.glob(path+'/XX*.csv'))
files = [singlefile for file in files for singlefile in file] # Flatten the list
XX = []
Xlist = []
Ylist = []

seed = 7 # fix random seed for reproducibility

for file in files:
    XXtemp = numpy.genfromtxt(file,delimiter=',')
    XX.append(XXtemp)
    Xtemp = XXtemp[:,:-5] # Remove last 5 columns (anchor, tag, distance)
    Ytemp = XXtemp[:,-5:]
    Ytemp = Ytemp/ scaleCameraTable # Change distances from pixels to cm
    
    Xlist.append(Xtemp)
    Ylist.append(Ytemp)
#XX = numpy.genfromtxt('forward/XX1.csv',delimiter=',')
numFeat = Xtemp.shape[1] # Should be 26



#%% Functions


def saveWorkspace(workspaceSavePath):
	my_shelf = shelve.open(workspaceSavePath,'n') # 'n' for new

	for key in dir():
	    try:
	        my_shelf[key] = globals()[key]
	    except TypeError:
	        #
	        # __builtins__, my_shelf, and imported modules can not be shelved.
	        #
	        print('ERROR shelving: {0}'.format(key))
	my_shelf.close()
	return

def restoreWorkspace(path):
	my_shelf = shelve.open(path)
	for key in my_shelf:
	    globals()[key]=my_shelf[key]
	my_shelf.close()
	
	
def SVR_Optimize(X_train,y_train):
	print("Optimize system for best C, gamma values")
	t0 = time()
	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), param_grid)
	clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
	clf = clf.fit(X_train, y_train)
	print("done in %0.3fs" % (time() - t0))
	print("Best estimator found by grid search:")
	print(clf.best_estimator_)
	return clf
		
def plotHistogram(x,comment):
	n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
#	plt.xlabel('Error Value (cm)')
	plt.ylabel('Occurrence')
	titleString = 'Histogram, ' + comment
	plt.title(titleString)
	#plt.grid(True)
	plt.show()
	return

def calcStats(x):
	outputString = 'max %.2f, min %.2f, mean: %.2f, median: %.2f, stdev: %.2f'%(np.max((x)), np.min((x)),np.mean((x)), np.median((x)), np.std(x))
	print outputString
	return np.max((x)), np.min((x)),np.mean((x)), np.median((x)), np.std(x)

def binPlot(x,y,bins=6):
	# Accepts two vectors, bins the data and plots
	# Also returns the values of the bins (x-axis)
	if len(x) == len(y):
		results = scipy.stats.binned_statistic(x,y,statistic='mean',bins=bins)
#		N = bins
		ind = []
		ind2 = []
		for i in range(len(results.bin_edges)-1):
			ind.append(np.mean((results.bin_edges[i:i+2]))) # Index label based on midpoint
			ind2.append('%.0f-%.0f'%(results.bin_edges[i],results.bin_edges[i+1])) # Index label string based on start-end indices
		ind = np.round(ind,decimals=1)
		width = 1
		fig, ax = plt.subplots()
		ax.bar(ind,results.statistic,width,color='r')
		ax.set_title('Binned error ',fontsize=16)
		ax.set_ylabel('Mean Error (cm)')
		ax.set_xlabel('Distance (cm)')
		plt.show()
		return results, ind, ind2
	else:
		print('error! mismatch length!')
		return
def boxPlot(results,y,ind):
	# Box plot of data
	bins = len(results.statistic)
	data = []
	for i in range(bins): # Loop through bins
		mask=results.binnumber-1 == i # Create mask
		binvalues = np.ma.array(y,mask=~mask) # Grab the values for that mask
		data.append(binvalues.compressed()) # Add values to a list
	
	fig,ax = plt.subplots()
	ax.boxplot(np.abs(data),labels=ind)
	ax.set_title('Error Distribution at different distances')
	ax.set_ylabel('Error values (cm)')
	ax.set_xlabel('Distance(cm)')
	plt.show()
	return

#%% Main Loop
#numFeat = 26
startTime = time()
rSquares = [] # R2 values
error = [] # Error values
errorRel = []
testData = []

for X,Y,file in zip(Xlist,Ylist,files):
#    numFeat = x.shape[1]
	print('Analysis on %s'%file)
	print( X.shape, Y.shape)
	n_features = X.shape[1]
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
	testData.append(y_test[:,-1])
	
	clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
	clf = clf.fit(X_train, y_train[:,-1])
	
	print("Predicting on the test set, SVR")
	t0 = time()
	y_pred = clf.predict(X_test)
	y_error = y_test[:,-1] - y_pred # Error in prediction, cm
	error.append(y_error) # Save error values for each trial
	errorRel.append(np.abs(y_error)/y_test[:,-1]) # Relative error values
	errorMean = np.mean(np.abs(y_error)) # Mean absolute error, cm
	print("done in %0.3fs" % (time() - t0))
	rSquared = clf.score(X_test,y_test[:,-1]) # R2 value
	rSquares.append(rSquared)
	  
print('Overall runtime was %.2f'%(time() - startTime))

#%% Results analysis
errorMeans = [np.mean(np.abs(i)) for i in error] # Mean error values for all trials
errorMedian = [np.median(np.abs(i)) for i in error] # Mean error values for all trials


allError = [i for trial in error for i in trial] # Errors in distance predictions
distances = [i for trial in testData for i in trial] # Actual distances
plotHistogram(allError, 'Error in cm') # Histogram of error
calcStats(np.abs(allError))

allErrorRel = [i for trial in errorRel for i in trial]
plotHistogram(allErrorRel,'Relative error') # Histogram of normalized/relative error
calcStats(np.abs(allErrorRel))

#%% Error as a function of distance
fig, ax = plt.subplots()
plt.scatter(distances,allError)
ax.set_title('Absolute Error')
ax.set_xlabel('Distance (cm)')
fig, ax = plt.subplots()
plt.scatter(distances,allErrorRel)
ax.set_title('Relative Error')
ax.set_xlabel('Distance (cm)')

stats.binned_statistic(distances,np.abs(allError),statistic='mean')

bins = 8
N = bins	
results,ind,ind2 = binPlot(distances,np.abs(allError),bins=bins) # Mean error at each binned distance
boxPlot(results, allError,ind) # Box plot of results

results,ind,ind2 = binPlot(distances,np.abs(allErrorRel),bins=bins) # Mean error at each binned distance
boxPlot(results, allErrorRel,ind) # Box plot of results

#%% Save workspace
#workspaceSavePath = path + '.out'
##saveWorkspace(workspaceSavePath) # Shelf method. Can be troublesome
#
import dill                            #pip install dill --user
filename = 'forwardSVR' + '_workspace.pkl'
dill.dump_session(filename)
#dill.load_session(filename) # To load 
#%% Try Dill Pickle
#restoreWorkspace('unsure.out')