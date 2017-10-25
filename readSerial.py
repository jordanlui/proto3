# Simple read from serial port
# Contains loop timing to end acquisition and then save to a file
# Works with the Oct 2017 char transmission code from device, which contains time and packet data


from __future__ import division
import serial
#import sys
import time
#import csv
import json
from saveCSV import cleanJSON


# Useful variables and system parameters
mcuFreq = 15 # Microcontroller frequency, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds
recordingTime = 10 # Set desired recording time

# Make serial connection
serial = serial.Serial("COM12", 115200, timeout=0)
if serial:
	print('connected')
buffer=[]
i = 0
tmax = recordingTime # max time in seconds
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
path = '../Data/oct24/'
filename = 'moving'
dataString = ''.join(buffer) # Make a continuous data string out of the buffered pieces
cleanJSON(dataString,path,filename)