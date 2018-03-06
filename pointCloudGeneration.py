# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 21:38:44 2018

@author: Jordan


Point Cloud calculator
Currently runs slower than matlab due to bad matrix handling and some other factors
"""

import numpy as np 
import time

#%% Setup
Ll = 269
Lu = 343
DH = [[0,0,0,90],[90,0,0,90],[90,0,0,-90],[-90, Lu,0,0],[90,Ll,0,0],[90,0,0,-90],[90,0,0,90]]
DH = np.asarray(DH)

#%% Functions
def calcPointCloud(ROM,stepSize,DH):
	cnt = 0
	rotation = []
	elbow = []
	wrist = []
	angles = []
	for i in range(ROM[0][0],ROM[0][1],stepSize):
		for j in range(ROM[1][0],ROM[1][1],stepSize):
			for k in range(ROM[2][0],ROM[2][1],stepSize):
				for l in range(ROM[3][0],ROM[3][1],stepSize):
					for m in range(ROM[4][0],ROM[4][1],stepSize):
						q = np.transpose(np.array([i,k,j,l,0,0,m]))
						elbowPos, wristPos, wristRot = armPose(DH,q)
						cnt = cnt + 1
						rotation.append(wristRot)
						elbow.append(elbowPos)
						wrist.append(wristPos)
						angles.append([i,j,k,l,m])
	return rotation, elbow, wrist, angles

def armPose(DH,q):
	# Accepts angle values and DH parameters, returns arm pose
	DH[:,-1] = DH[:,-1] - q

	elbow = transformDH(DH[0:4,:])
	wrist = transformDH(DH[:,:])	
	wristRot = wrist[0:3,0:3]
	wristPos = wrist[0:3,-1]
	elbowPos = elbow[0:3,-1]
	return elbowPos, wristPos, wristRot
	
def transformDH(input):
	# Calculates DH Transform matrix for a given input
	# Accepts list or singular	
	if input.shape[0] == 1:
		T = transformMat(input)
	else:
		transforms = []
		for i in range(input.shape[0]):
			transforms.append(transformMat(input[i,:]))
		T = transforms[0]
		for i in range(1,input.shape[0]):
			T = np.matmul(T,transforms[i])
	return T

def transformMat(input):
	# Accepts line of DH inputs and calculates transform matrix
	alpha = input[0]
	a = input[1]
	d = input[2]
	theta = input[3]
	T = np.array(([cosd(theta),-sind(theta),0,a],[sind(theta)*cosd(alpha),cosd(theta)*cosd(alpha),-sind(alpha),-sind(alpha)*d],[sind(theta)*sind(alpha),cosd(theta)*sind(alpha),cosd(alpha),cosd(alpha)*d],[0,0,0,1]))
	return T
	
def cosd(theta):
	return np.cos(theta * np.pi / 180)
def sind(theta):
	return np.sin(theta * np.pi / 180)

#%% Run
angleRangeShen = [[-60,180],[-40,120],[-30,120],[0,150],[0,180]]
angleRangeRosen = [[-10,62],[-14,134],[-72,55],[41,163],[-125,135]]
atrans = transformMat(DH[0,:])
combine = transformDH(DH[:,:])
q = (np.array([0,0,0,0,0,0,0]))
elbowPos, wristPos, wristRot = armPose(DH,q)
stepSize = 20

startTime = time.time()
rotation, elbow, wrist, angles = calcPointCloud(angleRangeRosen,stepSize,DH)
runTime = time.time() - startTime
print runTime