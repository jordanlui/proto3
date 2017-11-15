# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 23:17:45 2017

@author: Jordan
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

Loads multiple files and performs varying levels of Deep learning analysis on each file, printing out results for each

"""
import numpy
#import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaleCameraTable = 73.298 / 5.0
import time
import glob, os
import pickle, shelve
import matplotlib.pyplot as plt


#%% Try Model with Adam's deep learning data example for housing data
#dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
#dataset = dataframe.values
### split into input (X) and output (Y) variables
#X = dataset[:,0:13]
#Y = dataset[:,13]

#%% Load Datasets
path = 'forward'
files = glob.glob('../Analysis/nov3/' + path+ '/XX*.csv')
XX = []
Xlist = []
Ylist = []

seed = 7 # fix random seed for reproducibility

for file in files:
    XXtemp = numpy.genfromtxt(file,delimiter=',')
    XX.append(XXtemp)
    Xtemp = XXtemp[:,:-1]
    Ytemp = XXtemp[:,-1]
    Ytemp = Ytemp/ scaleCameraTable
    
    Xlist.append(Xtemp)
    Ylist.append(Ytemp)
#XX = numpy.genfromtxt('forward/XX1.csv',delimiter=',')
#numFeat = Xtemp.shape[1] # Should be 26

#X = numpy.genfromtxt('forward/x3.csv',delimiter=',')
#Y = numpy.genfromtxt('forward/t3.csv',delimiter=',')
#X = XX[:,:-1]
#Y = XX[:,-1]
#Y = Y/scaleCameraTable # Reach distance in cm
#X = X[:,0:4]
#dataset = numpy.hstack((X,numpy.resize(Y,(len(Y),1))))
#numFeat = X.shape[1]

#%% Standard model
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(numFeat, input_dim=numFeat, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model



def runBaseline(X,Y):
    numpy.random.seed(seed)
    # evaluate model with standardized dataset
#    numFeat = X.shape[1]
    estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, random_state=seed)
    startTime = time.time()
    results = cross_val_score(estimator, X, Y, cv=kfold)
    runTime = time.time() - startTime
    #print('Runtime: %.2f s'%runTime)
    print("Standard Analysis. Results: %.2f (%.2f) MSE. Runtime: %.2f s" % (results.mean(), results.std(), runTime) )
    
    
    # evaluate model with standardized dataset
    numpy.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)
    startTime = time.time()
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    runTime = time.time() - startTime
    #print('Runtime: %.2f s'%runTime)
    print("Standardized: %.2f (%.2f) MSE. Runtime: %.2f s" % (results.mean(), results.std(), runTime))
    return results

#%% Deeper model
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(numFeat, input_dim=numFeat, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
	
def runLarge(X,Y):
	
	numpy.random.seed(seed)
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
	pipeline = Pipeline(estimators)
	kfold = KFold(n_splits=10, random_state=seed)
	startTime = time.time()
	results = cross_val_score(pipeline, X, Y, cv=kfold)
	runTime = time.time() - startTime
	#print('Runtime: %.2f s'%runTime)
	print('Larger: %.2f (%2.f) MSE. Runtime: %.2f s' %(results.mean(),results.std(), runTime))
	return results

#%% Look at a wider network topology
def wider_model():
    # Create model
    model = Sequential()
    model.add(Dense(numFeat+4, input_dim=numFeat, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    # Compile
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def runWide(X,Y):

	numpy.random.seed(seed)
	estimators = []
	estimators.append(('standardize',StandardScaler()))
	estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
	pipeline = Pipeline(estimators)
	kfold = KFold(n_splits=10, random_state=seed)
	startTime = time.time()
	results = cross_val_score(pipeline, X, Y, cv=kfold)
	runTime = time.time() - startTime
	#print('Runtime: %.2f s'%runTime)
	print('Wider: %.2f (%2.f) MSE. Runtime: %.2f s' %(results.mean(),results.std(), runTime))
	return results

def saveWorkspace(pathOut):
	my_shelf = shelve.open(pathOut,'n') # 'n' for new

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
#%% Main Loop
numFeat = 26
loopStartTime = time.time()
resultsStd = []
resultsLarge = []
resultsWide = []

for X,Y,file in zip(Xlist,Ylist,files):
#    numFeat = x.shape[1]
	numFeat = X.shape[1]
	print('Analysis on %s'%file)
	print( X.shape, Y.shape)
	resultsStd.append(runBaseline(X,Y))
	resultsLarge.append(runLarge(X,Y))
	resultsWide.append(runWide(X,Y))
    
print('Overall runtime was %.2f'%(time.time() - loopStartTime))

#%% Results analysis

print('Median Error for Standardized is',numpy.median(resultsStd,axis=1))
print('Median Error for Large is',numpy.median(resultsLarge,axis=1))
print('Median Error for Wide is',numpy.median(resultsWide,axis=1))

n_bins = 10

plt.figure()
#for result in resultsWide:
plt.hist(resultsWide,n_bins)
plt.title('Wide network for 3 trials of %s movement'%path)
plt.xlabel('Error (cm)')
plt.ylabel('Occurences')
idx = range(int(numpy.max(resultsWide)))

plt.show()

plt.figure()
#for result in resultsLarge:
plt.hist(resultsLarge)
plt.title('Large network 3 trials of %s movement'%path)
plt.xlabel('Error (cm)')
plt.ylabel('Occurences')
plt.show()

plt.figure()
#for result in resultsStd:
plt.hist(resultsStd)
plt.title('Standard 3 trials of %s movement'%path)
plt.xlabel('Error (cm)')
plt.ylabel('Occurences')
plt.show()


#%% Save workspace
workspaceSavePath = path + '.out'
#saveWorkspace(workspaceSavePath) # Shelf method

import dill                            #pip install dill --user
filename = path + '_workspace.pkl'
dill.dump_session(filename)
# dill.load_session(filename) # To load 
#%% Try Dill Pickle
restoreWorkspace('unsure.out')

