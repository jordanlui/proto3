# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 14:46:42 2017

@author: Jordan

Makes plots of the Sharp and Omron Data, comparing to real distance values

"""

from __future__ import division
# Custom Functions
from analysisFunctions import OmronFeatures

import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import  SVR
from scipy import stats
import scipy
import pandas as pd
import glob, os
import pickle, shelve
import matplotlib.pyplot as plt



print(__doc__)

#%% Functions
def data2Frame(X,Y):
	M = len(X)
	ind = range(M)
	ind = np.reshape(ind,(M,1))
	sharp = X[:,0:4] # Sharp sensor data, not converted to cm
	omron = X[:,10:]
	realDistances = Y[:,-1] # Real distance cm
	realDistances = np.reshape(realDistances,(M,1))
#	omronVert, omronHoriz = OmronFeatures(omron) # Split to colums and rows
	omronMean = np.mean(omron,axis=1)
	omronMean = np.reshape(omronMean,(M,1))
	omronMedian = np.median(omron,axis=1)
	omronMedian = np.reshape(omronMedian,(M,1))
	data = np.hstack((ind,sharp,omronMean,omronMedian,realDistances,omron))
	df = pd.DataFrame(data,columns=['Index']+labels)
	return df

def plotRaw(df):
	# Plots Raw Sharp Data and Real distance data (cm)
	fig, ax = plt.subplots()
	df.Long1.plot(ax=ax,style='g--', legend=True).set_title('Raw Sharp sensor data')
	df.Short1.plot(ax=ax,style='b--',legend=True)
	df.Long2.plot(ax=ax,style='y--', legend=True)
	df.Short2.plot(ax=ax,style='m--',legend=True)
	
	ax.set_xlabel('Datapoints')
	ax.set_ylabel('Raw Data')
	ax2 = df.RealDist.plot(ax=ax,secondary_y=True, style='r-', legend=True) # Seconday Axis
	ax2.set_ylabel('Distance (cm)')
	return

def plotNorm(df):
	# Plots Raw Sharp Data, Omron, and Real distance data (cm), all normalized
	fig, ax = plt.subplots()
	df.Long1.plot(ax=ax,style='g--', legend=True).set_title('Normalized Sharp and Omron Data')
	df.Short1.plot(ax=ax,style='b--',legend=True)
	df.Long2.plot(ax=ax,style='y--', legend=True)
	df.Short2.plot(ax=ax,style='m--',legend=True)
	df.OmronMean.plot(ax=ax,style='c-',legend=True)
	df.OmronMedian.plot(ax=ax,style='k-',legend=True)
	
	ax.set_xlabel('Datapoints')
	ax.set_ylabel('Raw Data')
	ax2 = df.RealDist.plot(ax=ax,secondary_y=True, style='r-', legend=True) # Seconday Axis
	ax2.set_ylabel('Distance')
	return


#%% Export or load data
import dill                           
filename = 'allfiles' + '_workspace.pkl'
dill.load_session(filename)
#dill.dump_session(filename)

#%% Plot Analysis
labels= ['Long1','Short1','Long2','Short2','OmronMean','OmronMedian','RealDist','o1','o2','o3','o4','o5','o6','o7','o8','o9','o10','o11','o12','o13','o14','o15','016']
fileChoice = 0 # This should be forward1 - which I plotted once


for i in range(len(Xlist)):
	fileChoice = i
	X = Xlist[fileChoice]
	Y = Ylist[fileChoice]
	df = data2Frame(X,Y)
	checkLabelNames = list(df)
	df_norm = (df-df.mean()) / (df.max() - df.min())
	print('analysis on file: %s'%files[i])
#	plotRaw(df)
	df.plot.scatter(x='RealDist',y='OmronMedian')
	df.plot.scatter(x='RealDist',y='OmronMean')
#	plotNorm(df_norm)

