# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 23:07:48 2017

@author: Jordan
"""

from __future__ import division
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import glob,os

path = '../Analysis/nov3/'
xfiles = []
tfiles = []
for file in glob.glob(path+'x*.csv'):
    xfiles.append(file)
    
for file in glob.glob(path+'t*.csv'):
    tfiles.append(file)
filex = 'x.csv'
filet = 't.csv'
xfilesOrig = xfiles[:]
tfilesOrig = tfiles[:]
#%% Load
for i in range(0,3):
    xfiles = xfilesOrig[:]
    tfiles = tfilesOrig[:]
    X_test = np.genfromtxt(xfiles[i],delimiter=',')
    y_test = np.genfromtxt(tfiles[i],delimiter=',')
    xfiles.remove(xfiles[i])
    tfiles.remove(tfiles[i])
    
    X_train = []
    y_train = []
    for file in xfiles:
        X_train.append(np.genfromtxt(file,delimiter=','))
    for file in tfiles:
        y_train.append(np.genfromtxt(file,delimiter=','))
    
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    
    #%% Normalize
    X_train = X_train / np.linalg.norm(X_train)
    X_test = X_test / np.linalg.norm(X_train)
    
    #%% Models
    # #############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #svr_lin = SVR(kernel='linear', C=1e3)
    #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
    #y_lin = svr_lin.fit(X, y).predict(X)
    #y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
    print svr_rbf.score(X_test,y_test)
    y_pred = svr_rbf.predict(X_test)
    print np.mean(y_pred - y_test)
#%% #############################################################################
# Look at the results
#lw = 2
#plt.scatter(X[:,0], y, color='darkorange', label='data')
#plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
##plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
##plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()