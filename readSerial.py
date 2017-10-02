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
while True:
	data = serial.readline()
	# print data
	if len(data) > 0:
		# print(data)
		buffer.append(data)
	time.sleep(mcuPeriod)
	
	
# Save to file
path = '../Data/oct2/'
filename = 'stationary'
dataString = ''.join(buffer)
cleanJSON(dataString)