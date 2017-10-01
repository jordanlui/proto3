#!/usr/bin/python
# -*- coding: utf-8 -*-

# First basic build to read data from Arduino COM port
# We also try to write to csv now

# Initialize
from __future__ import division
import serial
#import sys
import time
#import csv
import json
from saveCSV import * # Custom script for saving CSV
from diagnostics import countPackets, packetContinuity

# Useful variables
mcuFreq = 50 # Microcontroller frequency, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds

# Make serial connection
serial = serial.Serial("COM12", 115200, timeout=0)
if serial:
	print('connected')

databuffer = []

# Run the loop until it crashes
while True:
# for i in range(0,10):connected	
	data = serial.readline()
	# if data:
		# print('Reading')
	# data = serial.readline().strip('\n\r')
 
	if len(data) > 0: # We only continue if we retrieve a line of data
		
		# Next we should determine if we receive a complete line of data.
		# Method one: Verify that string starts and ends with {}
		# Method two: verify a specific line length if we know what we're expecting
		print(data)

		if (data[0]=='{' or data[1]=='{') and data[-1] == '}':
	 		print('\n Complete')
#			print(data)
			
			try:
 				j = json.loads(data) # Putting this within a try loop is an easy way to reject the mashed data JSON strings that crash the loops
				

				if j:
					print(j)
					

					# Regular write method

					# csv_success = WriteToCSVAccGyro(j) # Write to a CSV file
					# print (csv_success)


					# Try Buffer write method
					databuffer.append(j) # Add to a buffer
					print(len(databuffer))
					if len(databuffer) > 100: # If buffer grows we will write to a file
						
						# csv_success = WriteToCSVAccGyro(databuffer) # Write to a CSV file for MPU9250-Uno
						csv_success = WriteToCSV(databuffer) # Saving for proto4 device data
						print (csv_success)
						databuffer = [] # Clear the buffer
				else:
					print 'JSON load problem'
			except:
				# This happens if the json file is formatted properly.
				# print(data)
				print('bad values')
				# break # Break is discouraged because it will crash the whole script
 			# print ('\n')
	 		# time.sleep(.5)
	 	
 	
	time.sleep(mcuPeriod) # Wait some time before reading again

