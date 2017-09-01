# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 15:54:37 2017

@author: Jordan
Library of code for analysis of results

Format of imported x matrix
[xcoord, ycoord, distance, sensor data]

"""
from __future__ import division
from  workspace_loader import load
from sklearn import datasets, linear_model
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#%% Parameters
#segment = 0.70 # Section of data that we train on
#seed = 0
#number_randomize = 2 # Number of times we want to random shuffle
#seeds = range(0,number_randomize)
#scale_table = 428/500 # Table scaling in pixels/mm. Note this calibration of active area square, but should be consistent across the entire camera frame.
#table_width = 500 # Table width in mm (test table)

#%% Functions
def randomize_data(x,seed):
	# Randomizes the data based on a seed value
	random.seed(a=seed)
	x = np.asarray(random.sample(x,len(x)))
	return x
def split_xt(xin):
	# Split data into xmatrix and t columns
	t = xin[:,0:2] # First two columns. ignore distance for now
	x = xin[:,3:]
	return x,t
def segment_data(x,seg_index):
	# Segment data based on chosen train/test segmentation

	# t values based on distance calculation
	#t_train = np.reshape(x[:seg_index,5],(seg_index,1))
	#t_test =  np.reshape(x[seg_index:,5],(m-seg_index,1))

	# t value based on actual x y coord
	x,t = split_xt(x) # Split into the given columns x and t
	t_train = t[:seg_index,:]
	t_test = t[seg_index:,:]
	
	x_train = x[:seg_index,:]
	x_test =  x[seg_index:,:]
	return x_train, x_test, t_train, t_test
	
def prep_model(x,seg_index,seed):
	# Shuffle and split data as desired in a full mix scenario
	# Don't use this when doing LOO shuffle based on folders!

	x = randomize_data(x,seed) # Shuffle the data
	x_train, x_test, t_train, t_test = segment_data(x,seg_index) # Split into test and train
	return x_train, x_test, t_train, t_test
def error_euclid(regr,x_test, t_test):
	# Error calculation, Euclidean distance between predicted and actual point
	# Regr is our ML model. x_test and t_test are the test data
	# This function uses model and x_test to calc prediction values and compare to t_test
	diff = np.sqrt(np.sum((regr.predict(x_test) - t_test) **2,axis=1)) # Euclidean error
	MSE = np.mean(diff) # Mean error (in pixels)
	variance = regr.score(x_test,t_test)
	return MSE, variance, diff
	
def error_shoulder(regr, x_test, t_test):
	# Error calculation with Euclidean distance from where the shoulder location has been estimated

	
def model(x_train, x_test, t_train, t_test, seed):
	# Run the analysis
	# Inputs: x, segment, random seeds
	# Output: MSE Error Value, Variance score
	
	regr = linear_model.LinearRegression(normalize=True) # Build model
	regr.fit(x_train, t_train)  # Fit model
	
#	MSE = np.mean((regr.predict(x_test) - t_test) **2)
	
	# Do error as euclidean
#	diff = np.sqrt(np.sum((regr.predict(x_test) - t_test) **2,axis=1)) # Euclidean error
#	MSE = np.mean(diff) # Mean error (in pixels)
#	variance = regr.score(x_test,t_test)
	MSE, variance, diff = error_euclid(regr,x_test,t_test) # Euclidean distance error calc
	return MSE, variance, diff
	
def model_multi(x_train,x_test,t_train,t_test,seed):
	# This function runs the model repeatedly based on number of random seeds and return the average MSE values and variances
	# Useful for single file analysis - not useful for LOOCV
	MSE = []
	variance = []
	for seed in seeds:
#		x_train, x_test, t_train, t_test = prep_model(x,seg_index,seed)
		error, var, diff = model(x_train, x_test, t_train, t_test,seed)
		MSE.append(error)
		variance.append(var)

	MSE_mean = np.mean(MSE)
	variance_mean = np.mean(variance)
	# Coefficients
	#print('Coefficients: \n', regr.coef_)
	## MSE
#	print 'Ran ',len(seeds),' randomizations'
#	print("MSE: %.2f" % MSE_mean)
#	print('Variance score: %.2f' %variance_mean)
	return MSE_mean, variance_mean
	
def LOOCV(path,seed=0,scale_table=1):
	# Runs cross validation routine
	# Accepts a list of files paths and performs LOOCV
	# Changed to output system error in mm
	

	error = []
	var = []
	
	for i in range(0,len(path)): # Iterate through the files
		x_train=[]
		x_test=[]
		
		single_path = path[i] # Grab a single file for testing
		rest_path = path[:i]+path[i+1:] # Rest of files are for training
		
		x_test = load(path=single_path) # Load test file into into x matrix
		x_test,t_test = split_xt(x_test) # Split to x and t
	
		
		for apath in rest_path:	# Loop through the training files
			xx_train = load(path=apath) 
			x_train.append(xx_train)	# Append into a master training list
	  
		x_train = np.vstack(x_train) # Convert list to a matrix
		x_train,t_train = split_xt(x_train) # Split to x and t arrays

		
		# Run the model
		MSE_mean, variance_mean,diff = model(x_train,x_test,t_train,t_test,seed)
		error.append(MSE_mean/scale_table)
		var.append(variance_mean)
	return error,var


def singleRun(path,segment=0.7,seed=0,scale_table=1):
	errors = [] # Error values in mm

	for path in path:
	#path = path[2]
		
		x = load(path)
		seg_index = int(segment*len(x))
		x_train, x_test, t_train, t_test = prep_model(x,seg_index,seed)
		MSE, variance, diff = model(x_train,x_test,t_train,t_test,seed)
		
		# Histogram on error distribution
		diff_mm = diff/scale_table # This is the mm value error
		error_mean = np.mean(diff_mm)
		errors.append(error_mean)
	
		print 'Results: Mean error (mm)', error_mean, 'min error (mm) is',np.min(diff_mm),'max error',np.max(diff_mm),'median',np.median(diff_mm)
		
		# Plots
#		plt.figure()
#		plt.hist(diff_mm,bins='auto')
#		plt.title('Histogram of error (mm)')
#		plt.ylabel('Occurrences')
#		plt.xlabel('Error value (mm)')
#		plt.show()
#		
#		# 3D plot of error
#		fig = plt.figure()
#		
#		ax = fig.add_subplot(111, projection='3d')
#		
#		ax.scatter(t_test[:,0], t_test[:,1], diff)
#		
#		plt.title('Error with position')
#		ax.set_xlabel('X')
#		ax.set_ylabel('Y')
#		ax.set_zlabel('Error')
	
	numfiles = len(path)
	print 'mean error across %d files was %f mm'%(numfiles,np.mean(errors))
#	print 'Number of random shuffles:',len(seeds)
	print 'Number of files',numfiles
	return 0

