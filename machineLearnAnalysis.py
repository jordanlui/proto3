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
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier

path = '../Analysis/nov3/forward/'
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

#%% Functions
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


#%% Single analysis
i = int(0)
X = np.genfromtxt(xfiles[i],delimiter=',')
y = np.genfromtxt(tfiles[i],delimiter=',')

# Normalize
#X = X / np.linalg.norm(X)
#y = y / np.linalg.norm(y)

#plt.figure()
#plt.plot(X)
#plt.plot(y)
# Shuffle
#X,y = unison_shuffled_copies(X,y)

seg = 20
X_test = X[:seg,:]
X_train = X[seg:,:]
y_test = y[:seg]
y_train = y[seg:]


#%% Support Vector Regression
model = SVR(kernel='rbf', C=1e1, gamma=0.001)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
model.fit(X_train, y_train).predict(X_test)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
print model.score(X_test,y_test)
y_pred = model.predict(X_test)
print np.mean(y_pred - y_test)

# Plot
fig1 = plt.figure()
plt.plot(y_test)
plt.plot(y_pred)
plt.title('Test and prediction single')

#%% Neural Network

numFeatures = 26

h = .02  # step size in the mesh

alphas = np.logspace(-5, 3, 5)
names = []
for i in alphas:
    names.append('alpha ' + str(i))

classifiers = []
for i in alphas:
    classifiers.append(MLPClassifier(alpha=i, random_state=1))



X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
#cm = plt.cm.RdBu
#cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#ax = plt.subplot(1, len(classifiers) + 1, i)
## Plot the training points
#ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
## and testing points
#ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
#ax.set_xlim(xx.min(), xx.max())
#ax.set_ylim(yy.min(), yy.max())
#ax.set_xticks(())
#ax.set_yticks(())
#i += 1

i=1
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
 # iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(3, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='black', s=25)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6, edgecolors='black', s=25)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1



#%% Load and do looped analysis LOOCV
#for i in range(0,3):
#    xfiles = xfilesOrig[:]
#    tfiles = tfilesOrig[:]
#    X_test = np.genfromtxt(xfiles[i],delimiter=',')
#    y_test = np.genfromtxt(tfiles[i],delimiter=',')
#    xfiles.remove(xfiles[i])
#    tfiles.remove(tfiles[i])
#    
#    X_train = []
#    y_train = []
#    for file in xfiles:
#        X_train.append(np.genfromtxt(file,delimiter=','))
#    for file in tfiles:
#        y_train.append(np.genfromtxt(file,delimiter=','))
#    
#    X_train = np.vstack(X_train)
#    y_train = np.hstack(y_train)
#    
#    #%% Normalize
##    normX = np.linalg.norm(X_train)
##    normy = np.linalg.norm(y_train)
##    X_train = X_train / normX
##    X_test = X_test / normX
##    y_train = y_train / normy
##    y_test = y_test / normy
#    
#    #%% Models
#    # #############################################################################
#    # Fit regression model
#    model = SVR(kernel='rbf', C=1e2, gamma=0.0001)
#    #svr_lin = SVR(kernel='linear', C=1e3)
#    #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#    model.fit(X_train, y_train).predict(X_test)
#    #y_lin = svr_lin.fit(X, y).predict(X)
#    #y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
#    print model.score(X_test,y_test)
#    y_pred = model.predict(X_test)
#    print np.mean(y_pred - y_test)
#    
#    # Plot
#    fig1 = plt.figure()
#    plt.plot(y_test)
#    plt.plot(y_pred)
#    plt.title('Test and prediction')
#%% #############################################################################
#fig1 = plt.figure()
#plt.plot(y_test)
#plt.plot(y_pred)
#plt.title('Test and prediction')

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