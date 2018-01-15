# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:29:25 2017

@author: Jordan

Parse Data from ART System, save to file
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import glob, os

#%%
#filePath = '../data/body.drf'
filePath = '../data/elbowFlex1.drf'
fileName = re.findall('.*/(.+).drf',filePath)[0]
dataLabelSearch = '([a-zA-Z0-9]+) .+' # RegEx Search string for the alphanum labels
dataLabels=['fr','ts','6dcal','6d','6di','6df2'] # Search string for alphanum labels
regexFloat = '[a-z0-9]+ ([0-9.]+)' # Regex for the float values
regexFloatInt = '-*[0-9]+\.*[0-9]*' # Float or int values, postive or negative
regexFloatInt2 = '-?[0-9]+\.?[0-9]*' # Float or int values, postive or negative
regexBodyObjects = '\[.+?\]\[.+?\]\[.+?\]' # Grab each body object
regexNumObjects = '6di ([0-9]) ' # Number of objects tracked
queryDevice = '(?:(-?[0-9]+) ?)' # Query for integer Omron / IMU Device data
query6d = '((\[(?:(?:-??[0-9]+\.??[0-9]*) ??){2}\])(\[(?:(?:-??[0-9]+\.??[0-9]*) ??){6}\])(\[(?:(?:-??[0-9]+\.??[0-9]*) ??){9}\]) ??)' # Grabs
query6di = '((\[(?:(?:-??[0-9]+\.??[0-9]*) ??){3}\])(\[(?:(?:-??[0-9]+\.??[0-9]*) ??){3}\])(\[(?:(?:-??[0-9]+\.??[0-9]*) ??){9}\]) ??)' # 6di data


# Data Arrays
#dataLabel = []
#frame = []
#time = []
#sixdcal = []
#NumObjects = -1
#positions = []

#%% Functions
def BodyPosition(query,string):
    # Search for position data and return as a list of float values
    BodyValues = re.findall(query,string) # Grab numbers
    BodyValues = [float(i) for i in BodyValues] # Convert to float
    position = BodyValues[3:6] # Grab position values
    return position

def BodyRotation(query,string):
    # Search for position data and return as a list of float values
    BodyValues = re.findall(query,string) # Grab numbers
    BodyValues = [float(i) for i in BodyValues] # Convert to float
    position = BodyValues[6:] # Grab position values
    return position
				
def parse(filePath): # Parse files
	file = open(filePath,'r') # Open file
	# Data Arrays
	dataLabel = []
	frame = []
	time = []
	sixdcal = []
	NumObjects = -1
	positions = []
	rotations = []
	deviceDatas = [] # Data from Omron / IMU Device
	for line in file: # Process one line at a time
		
		if re.findall(dataLabelSearch,line):
			dataType = re.findall(dataLabelSearch,line)[0] # This is the data type label
			if dataType == dataLabels[0]: # frame number
				aframe = re.findall(regexFloat,line)[0]
				frame.append(int(aframe))
			elif dataType == dataLabels[1]: # time value
				atime = re.findall(regexFloat,line)[0]
				time.append(float(atime))
			elif dataType == dataLabels[2]: # 6d calibration data (number of tracked bodies)
				asixdcal = re.findall(regexFloat,line)[0]
				sixdcal.append(int(asixdcal))	
			elif dataType == dataLabels[3]: # 6d tracker data
#				print '6d data instead of 6d inertial detected'
				pass
			elif dataType == dataLabels[4]: # 6d inertial tracker data
				if NumObjects == -1: # Check for the number of tracked objects if it isn't known yet
					NumObjects = int(re.findall(regexNumObjects,line)[0]) # Find number of objects that were being tracked				
					for i in range(NumObjects): # Pre-allocate a list for positions, rotations of each body
						positions.append([])
						rotations.append([])
						
				BodyObjects = re.findall(regexBodyObjects,line) # Grab each of the tracked bodies
				for i,body in enumerate(BodyObjects):
					position = BodyPosition(regexFloatInt,body) # Grab the position
					positions[i].append(position) # Append to appropriate list	
					rotation = BodyRotation(regexFloatInt,body) # Grab rotation data
					rotations[i].append(rotation) # Append to appropriate list	
			else: # This is likely a Omron IMU line
				if len(line) > 100: # This is probably a prototype data line if length is sufficient
					deviceData = re.findall(queryDevice,line)
					if len(deviceData) == 26:
						deviceData = [int(i) for i in deviceData]
						deviceDatas.append(deviceData)
					else:
						deviceDatas.append([0 for i in range(26)]) # Put an empty line in otherwise
				elif len(line) > 40:
					deviceDatas.append([0 for i in range(26)]) # Put an empty line in otherwise
	return frame,time,sixdcal, positions, rotations, deviceDatas

def savePosition(positions,time):
	lengths = []
	for body in positions:
		lengths.append(len(body))
	if lengths[1:] == lengths[:-1]:
		print 'All arrays are equal length'
		lengthsEqual = True
	else:
		print 'Arrays are not equal in length'
		minLength = min(lengths)
		newPositions = []
		for body in positions:
			newPositions.append(body[:minLength])
		positions = newPositions
		lengthsEqual = True
	
	if lengthsEqual:
		for i,body in enumerate(positions):
			if i == 0:
				positionArray = np.array(positions[0])
			else:
				positionArray = np.hstack((positionArray,body))
				
	if len(time) > len(positionArray):
		time=time[0:len(positionArray)]
#	elif len(time) < len(positionArray):
#		time = time +
	time = np.reshape(time,(len(time),1))
	positionArray = np.hstack((time,positionArray)) # Add in time data
	np.savetxt(fileName+'.csv',positionArray, delimiter=',')
	return positionArray

def savePositionJan12(positions,rotations,deviceDatas,time):
	data = []
	data.append(time)
	data.append(deviceDatas)
	[data.append(i) for i in positions]
	[data.append(i) for i in rotations]
	lengths = []
	for i in data:
		lengths.append(len(i))
	if lengths[1:] == lengths[:-1]:
		print 'All arrays are equal length'
		lengthsEqual = True
		newData = data
	else:
		print 'Arrays are not equal in length'
		minLength = min(lengths)
		newData = []
		for i in data:
			newData.append(i[:minLength])
		lengthsEqual = True
	
	if lengthsEqual:
		dataArray = np.array(newData[0]).reshape((len(newData[0]),1))# Put the time row in first
		for i in range(1,len(newData)):
#			print np.array(newData[i]).shape
			dataArray = np.hstack((dataArray,np.array(newData[i])))
				
	np.savetxt(fileName+'.csv',dataArray, delimiter=',')
	return dataArray

#%%
path = '../../ART IR Tracker Setup/data/jan12/'
files = glob.glob(path+'log*.txt')


for file in files:
	filePath = file
	fileName = os.path.basename(file)[:-4]
	frame,time,numBodies, positions, rotations, deviceDatas = parse(filePath)
	len(frame), len(time), len(positions[0]), len(rotations[0]), len(deviceDatas) # Check data length
	dataArray = savePositionJan12(positions,rotations,deviceDatas,time)
	np.savetxt(fileName+'.csv',dataArray, delimiter=',')


		
#%% Analysis on file

timePeriod = (np.array(time)[1:]-np.array(time)[:-1])	
freqAvg = 1/np.mean(timePeriod)
#plt.hist(timePeriod)
#plt.title('Period Histogram)
plt.hist(freqAvg)
plt.title('Frequency Histogram')

#%% Format as array
#lengths = []
#for body in positions:
#	lengths.append(len(body))
#if lengths[1:] == lengths[:-1]:
#	print 'All arrays are equal length'
#	lengthsEqual = True
#
#if lengthsEqual:
#	for i,body in enumerate(positions):
#		if i == 0:
#			positionArray = np.array(positions[0])
#		else:
#			positionArray = np.hstack((positionArray,body))
#np.savetxt(fileName+'.csv',positionArray)