# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:58:18 2017

@author: Jordan
"""

# Check packet integrity
import numpy as np

def countPackets(data):
	# Check number of packets for each time value, and see if they are similar. An indicator of packet loss
	# Sept 2017

	timeStart = data[0,0]
	timeEnd = data[-1,0]
	
	# Count the number of packets in each time step
	packetCount = []
	
	for i in range(int(timeStart), int(timeEnd)):
		m = [row for row in data if row[0]==i]
		packetCount.append(len(m))
		
	packetCount = packetCount[1:-1]# Ignore the start and end values
	fMin = np.min(packetCount)
	fMax = np.max(packetCount)
	fMean = np.mean(packetCount)
	fSTD = np.std(packetCount)
	print 'Frequency min, max, average, stdev is %.2f,%.2f,%.2f,%.2f'%(fMin,fMax,fMean,fSTD)
	return packetCount, fMin,fMax,fMean,fSTD

def packetContinuity(data):
	# Check packet continuity by taking a simple difference of all the elements
	packets = data[:,1]
	trydiff = packets[1:] - packets[:-1] # Ignore starting and end files
	diffMin = np.min(trydiff)
	diffMax = np.max(trydiff)
	etol = 1e-5
	if diffMin - 1 < etol and diffMax - 1 < etol:
		return True
	else:
		return False

def checkPackets(buffer):
	packetTime = []
	for line in buffer:
		packetTime.append(line["packet"])
	
	# Calculate deltas
	deltaTime = []
	for i in range(0,len(packetTime)-1):
		deltaTime.append(packetTime[i+1] - packetTime[i])
		
	# Math stats
	deltaMax = np.max(deltaTime) - 1
	deltaMin = np.min(deltaTime) - 1
	deltaMean = np.mean(deltaTime) - 1
	summaryText = 'Packet loss varies from %.2f to %.2f, with mean of %.2f'%(deltaMin,deltaMax,deltaMean)
	print(summaryText)
	return summaryText