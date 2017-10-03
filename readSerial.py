# Simple read from serial port

from __future__ import division
import serial
#import sys
import time
#import csv
import json
from saveCSV import cleanJSON


# Useful variables
mcuFreq = 16 # Microcontroller frequency, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds


# Make serial connection
serial = serial.Serial("COM12", 115200, timeout=0)
if serial:
	print('connected')
buffer=[]
i = 0
tmax = 60 # max time in seconds
imax = tmax * mcuFreq # Max packets required to meet time
while True:
	data = serial.readline()
	# print data
	if len(data) > 0:
		# print(data)
		buffer.append(data)
		i += 1
		if i%20 == 0: # Print out our packet count, intermittently
			print (i)
	time.sleep(mcuPeriod)
	if i > imax:
		break
	
# Save to file
path = '../Data/oct2/'
filename = 'stationary'
dataString = ''.join(buffer)
cleanJSON(dataString,path,filename)