# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 23:17:45 2017

@author: Jordan
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
"""
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaleCameraTable = 73.298 / 5.0
import time

#%% Try Model with Adam's deep learning data example for housing data
#dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
#dataset = dataframe.values
### split into input (X) and output (Y) variables
#X = dataset[:,0:13]
#Y = dataset[:,13]

#%% Load Jordan Dataset
X = numpy.genfromtxt('../Analysis/nov3/forward/x1.csv',delimiter=',')
Y = numpy.genfromtxt('../Analysis/nov3/forward/t1.csv',delimiter=',')
Y = Y/scaleCameraTable # Reach distance in cm
#X = X[:,0:4]
dataset = numpy.hstack((X,numpy.resize(Y,(len(Y),1))))
numFeat = X.shape[1]

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


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
startTime = time.time()
results = cross_val_score(estimator, X, Y, cv=kfold)
runTime = time.time() - startTime
print('Runtime: %.2f s'%runTime)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


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
print('Runtime: %.2f s'%runTime)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

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

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
startTime = time.time()
results = cross_val_score(pipeline, X, Y, cv=kfold)
runTime = time.time() - startTime
print('Runtime: %.2f s'%runTime)
print('Larger: %.2f (%2.f) MSE' %(results.mean(),results.std()))

#%% Look at a wider network topology
def wider_model():
    # Create model
    model = Sequential()
    model.add(Dense(numFeat+4, input_dim=numFeat, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    # Compile
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
starTime = time.time()
results = cross_val_score(pipeline, X, Y, cv=kfold)
runTime = time.time() - startTime
print('Runtime: %.2f s'%runTime)
print('Wider: %.2f (%2.f) MSE' %(results.mean(),results.std()))

print('End of script \n\n')
