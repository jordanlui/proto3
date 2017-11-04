# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 15:15:09 2017

@author: Jordan
Track position on webcam and read from prototype device
"""

# USAGE
# python script.py --video ball_tracking_example.mp4
# python script.py
# OpenCV3 required! There are a few lines below that will not work in OpenCV2
# This script is configured to track two points and return the coordinate values and a distance calculation

# Packages
from __future__ import division
from __future__ import print_function
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
from serial import *
import decodeBytes27
import myFunctions
import numpy as np


# WEBCAM SETUP
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# Parameters for operation
camera_address = 1 # 1 for the USB webcam, 0 for the onboard webcam
minRadius = 5 # Minimum radius of the circle to be shown on screen
greenLower = (41,94,60)			# Green tracker in Surrey 4040
greenUpper = (106,210,142)	
pts = deque(maxlen=args["buffer"])


if not args.get("video", False):
	camera = cv2.VideoCapture(camera_address)
else:
	camera = cv2.VideoCapture(args["video"])

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
size = (640,480)
video = cv2.VideoWriter('output.avi',fourcc, 30.0, size)
# Define and Open coordinates text file
text_file = open("OutputTest.txt", "w")
text_file.close()
# END OF WEBCAM SETUP

time.sleep(2.5)

# IR DEVICE SETUP
# Useful variables
mcuFreq = 50 # Microcontroller frequency, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds
# Run parameters
testLength = 10 #loop will run for testLength seconds
pathout = '../Data/nov3/'
filename = 'random_offtable2.csv'
filenameCoords = 'coords_' + filename
fullpathOut = pathout + filename
fullpathOutCoords = pathout + filenameCoords

coordData = [] # list of webcam coordinate acquisitions

#select what to do with data. 
storeOrDecode = 0 #0 = store in a datalist,  1 = decode instantly
packetsReceived = 0
serial = Serial("COM12", 115200, timeout=2) # Make serial connection
if serial:
	print('connected')
timeStart = time.time() # Record current time
timeout = timeStart + testLength # Represents end time of our acquisition
timeRemaining = testLength
deviceTime = []
datalist = [] # list of IR device acquisitions
dataCleaned = []
# END OF IR DEVICE SETUP

# MAIN LOOP
while True:
	
	# IR PROTO DEVICE DATA ACQUISITION
	data = serial.readline()
	deviceTime.append(time.time()) # Timestamp of the record time
	if len(data) > 0:
		if storeOrDecode == 0:
			datalist.append(data) # Store raw data in a datalist while running, process it later
		else:
			dataCleaned.append(decodeBytes27.decode(data))
		packetsReceived += 1
	#while loop times out in testLength seconds
	if time.time() > timeout:
		break
	time.sleep(mcuPeriod)
	
	timeElapsed = time.time() - timeStart
	timeRemaining = testLength - timeElapsed
	percentComplete = int(round(timeElapsed / testLength * 100))
	if percentComplete % 5 == 0:
		print('%i / 100'% percentComplete)
	
	# WEBCAM TRACKING
	(grabbed, frame) = camera.read()
	camTime = time.time() # Record the time that we sample from camera
#	text_file = open("OutputTest.txt", "a")
	if args.get("video") and not grabbed:
		break
	frame = cv2.flip(frame, 1)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2] # The [-2] grabs the list object
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # Sort by area
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 1:
		c = cnts[0]
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		c2 = cnts[1] 	# Grab the second largest contour
		((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
		M = cv2.moments(c2)
		center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		distance = np.sqrt( (x2-x)**2 + (y2-y)**2)
#		msgOut = "{X: %.3f, Y: %.3f, X2: %.3f, Y2: %.3f, Dist: %.3f} \n" % (x, y, x2, y2, distance)
		msgOut = "%.3f,%.3f,%.3f,%.3f,%.3f\n" % (x, y, x2, y2, distance) 
		coordData.append([camTime, x,y,x2,y2,distance])
#		print(msgOut)
#		text_file.write(msgOut)

		if radius > minRadius: # only proceed if the radius meets a minimum size	
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1) # draw the circle and centroid on the frame,
			
		if radius2 > minRadius:
			cv2.circle(frame, (int(x2), int(y2)), int(radius2),
				(0, 150, 150), 2)
			cv2.circle(frame, center, 5, (0, 0, 150), -1)
	pts.appendleft(center)

	
#	for i in xrange(1, len(pts)): # loop over the set of tracked points, display trail
#		if pts[i - 1] is None or pts[i] is None:
#			continue
#		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
#		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	video.write(frame)
	text_file.close()
#	time.sleep(0.07725)
	
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
video.release()
cv2.destroyAllWindows()
text_file.close()
serial.close()
# PROCESSING WEBCAM DATA

text_file = open("OutputTest.txt", "a")
for row in coordData:
	text_file.write(str(row))
text_file.close()
print("%i rows of data recorded"%len(coordData))
coordArray = np.array(coordData)
np.savetxt(fullpathOutCoords,coordArray,delimiter=','	)

# POST PROCESSING OF IR DATA

# Post-processing decoding if we store the data in memory during recording
indexWorkingRow = []
coordCleaned = []
if storeOrDecode == 0:
	for i in range (0,len(datalist)-1):
		row = datalist[i]
		dataOut = decodeBytes27.decode(row)
		if dataOut != 0:
			dataCleaned.append(dataOut)
			indexWorkingRow.append(i) # Write the index of this working row
#			coordCleaned.append(coordData[i]) # Save the coordinate from this time step
		elif dataOut == 0:	  # This error check helps to detect lines of bytes that got split into two lines.
			tryrow = row + datalist[i+1]
			dataOut = decodeBytes27.decode(tryrow)
			if dataOut != 0:	# This error check helps to detect lines of bytes that got split into two lines.
				dataCleaned.append(dataOut)
				indexWorkingRow.append(i) # Write the index of this working row
#				coordCleaned.append(coordData[i]) # Save the coordinate from this time step
				i = i + 1
		  
dataArray = []
# Flatten data into an array
for row, dtime in zip(dataCleaned,deviceTime): 
	rowA = np.array(row[0:12]) # Timing and distance data
	rowB = np.array(row[-1]) # Omron Array data
	rowC = dtime
	rowNew = np.hstack((rowA,rowB,rowC))
	rowNew = np.reshape(rowNew,(1,len(rowNew)))
	dataArray.append(rowNew)
		
dataArray = np.vstack((dataArray))
# Save processed data
np.savetxt(fullpathOut,dataArray, delimiter=',')

# Save raw data
rawpathOut = pathout + 'raw_' + filename[:-4] + '.txt'
f = open(rawpathOut, 'wb')
for row in datalist:
	f.write(row)
f.close()

# Check packet droppage and stats
packetStart = dataCleaned[0][0]
packetEnd = dataCleaned[-1][0]
packetSent = packetEnd - packetStart + 1
timeStart = dataCleaned[0][1]
timeEnd = dataCleaned[-1][1]
timeRun = timeEnd - timeStart + 1
freqExperimental = packetSent / timeRun * 1e3

#myFunctions.printStats(storeOrDecode,packetsReceived,datalist)
#freqExperimental = packetsReceived/testLength # Frequency of data received
#print 'frequency is' , freqExperimental, 'Hz'
print('frequency is %.2f Hz' %freqExperimental)
if packetSent == len(dataCleaned):
	print('packet drop unlikely')
print('sent %i packets and received %i packets'%(packetSent,len(dataCleaned)))

