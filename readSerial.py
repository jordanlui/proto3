# Simple read from serial port

from __future__ import division
import serial
#import sys
import time
#import csv
import json


# Useful variables
mcuFreq = 50 # Microcontroller frequency, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds

# Make serial connection
serial = serial.Serial("COM12", 115200, timeout=0)
if serial:
	print('connected')

while True:
	data = serial.readline()
	# print data
	if len(data) > 0:
		print(data)
	time.sleep(mcuPeriod)