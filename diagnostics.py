# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:58:18 2017

@author: Jordan
"""

# Check packet integrity
import numpy as np
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