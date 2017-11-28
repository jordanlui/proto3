# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:17:40 2017

@author: Jordan

Test model ability to generalize results.
Initial trials show a poor performance, when looking at the simple forward movement

Nov 25:
R2 value is 0.94 and mean error 0.89
Test now with different data
R2 value is -0.41 and mean error 5.24

"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

print(__doc__)

#%% Load Data
import dill
filename = 'allfilesSVR' + '_workspace.pkl'
#dill.dump_session(filename)
dill.load_session(filename) # To load 

#%% Functions
def errors(error):
	print('Mean error %.4f, Mean Abs Error %.4f, Median Abs Error %.4f'%(np.mean(error),np.mean(np.abs(error)),np.median(np.abs(error))))
	return

#%% Main
# Train on one set of data and test it
X = [Xlist[0]] + Xlist[1:2]
y = [Ylist[0]] + Ylist[1:2]
X = np.vstack(X)
y = np.vstack(y)
Xall = np.vstack(Xlist)
yall = np.vstack(Ylist)[:,-1]

#y = np.array([i for j in y for i in j])
y = y[:,-1] # Grab distance values (webcam acquired)
if len(X) != len(y):
	print 'data error! Lengths not equal!'
	
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
clf = clf.fit(X_train, y_train) # Fit model with training data

y_pred = clf.predict(X_test)

error = y_pred - y_test

#print('Mean error %.4f, Mean Abs Error %.4f, Median Abs Error %.4f'%(np.mean(error),np.mean(np.abs(error)),np.median(np.abs(error))))
errors(error)
print('Score value is %.2f and mean error %.2f'%(clf.score(X_test,y_test), np.mean((clf.predict(X_test) - y_test))))

# Manual R2 value
#u = ((y_test - y_pred)**2).sum()
#v = ((y_test - y_test.mean())**2).sum()
#R2 = (1-u/v)
#print('Manual R2 value is',R2)
#%% Bring in other training data
print 'Test now with different data \n'
X_test2 = Xlist[2]
#X_test2 = np.vstack(X_test2)
y_test2 = Ylist[2][:,-1]
#y_test2 = [i for j in y_test2 for i in j]
#y_test2 = np.array(y_test2)

y_pred2 = clf.predict(X_test2)
error2 = y_pred2 - y_test2
errors(error2)
print('Score value and is %.2f and mean error %.2f'%(clf.score(X_test2,y_test2), np.mean(np.abs(clf.predict(X_test2) - y_test2))))

#%% Plot resuts
plt.figure()
plt.scatter(y_test,y_pred, label='Orig set')
plt.scatter(y_train,y_train, label='training')
plt.scatter(y_test2,y_pred2, label='Different set')
plt.legend()
plt.title('Plot of training, testing, and transfer learning values')

#%% Cross validation on all
#clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
scoreCV = cross_val_score(clf,Xall,yall, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoreCV.mean(), scoreCV.std() * 2))

#%% Cross Validation on select
#clf = SVR(kernel='rbf',C=1000, gamma=0.0001, epsilon = 0.1, max_iter=-1, shrinking=True, tol=0.001)
scoreCV = cross_val_score(clf,X,y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoreCV.mean(), scoreCV.std() * 2))