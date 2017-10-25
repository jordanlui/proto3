# Simple read from serial port
# Contains loop timing to end acquisition and then save to a file
# Serial read testing for IMU Speed testing
# Device: Genuino101 with MPU9250?

#import sys
from __future__ import division
from saveCSV import cleanJSON
# from StringIO import StringIO
import csv
import json
import re
import serial
import time

# Useful variables and system parameters
mcuFreq = 120 # Estimate of microcontroller frequency for loop timing, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds
recordingTime = 60 # Set desired recording time
packetCountFreq = 200 # Interval for system printing packet numbers

# Parameters to save to file
path = '../Data/oct16/'
filename = 'calibration_6axes'
fullFile = path + filename + '.csv'

# Make serial connection
serial = serial.Serial("COM6", 115200, timeout=0)
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
		if i%packetCountFreq == 0: # Print out our packet count, intermittently
			print (i)
	time.sleep(mcuPeriod)
	if i > imax:
		break
	

# Done recording

# Processing results
# Data format from Oct 2017 speed testing is 
#@87668,12526,0.21,0.28,-9.68,6.17,-7.23,-9.68#


completeLines = []

# Parse results with RegEx
datastring = ''.join(buffer)
query = '@([\s\S]+?),?#\\n' # RegEx string search for Oct 2017 speed testing
m = re.findall(query,datastring)

completeLines = list(m)

# OLD METHOD Parse results by simply comparing length and the start/end characters
#completeLines = []
#for line in buffer:
#	
#	if len(line) > 10: # Some trivially short string length to compare against
#		line = line.strip('\n\r') # Strip line return values
#		if line[0] == '@' and line[-1] == '#':
#			line = line[1:-1] # Remove the header/footer characters
#			completeLines.append(line)
reader = csv.reader(completeLines)
completeLines = list(reader) # Should load into a list	

# Look at loop times and data frequency
endTime = int(completeLines[-1][0])
startTime = int(completeLines[0][0])
duration = (endTime - startTime) /1e3
packetEnd = int(completeLines[-1][1])
packetStart = int(completeLines[0][1])
packetSent = packetEnd - packetStart + 1
frequency = (packetSent) / duration

print('Sent %i packets in %.2f sec. Freq %.2f Hz' % (packetSent, duration, frequency))
if packetSent == len(completeLines):
	print('Packet loss unlikely. Number of packets implies no drops')
else:
    print('Packet loss has possibly occured')
        

# Save to a file
with open(fullFile, 'wb') as csv_file:
	writer = csv.writer(csv_file, delimiter=",")
	for line in completeLines:
		writer.writerow(line)

#dataString = ''.join(buffer) # Make a continuous data string out of the buffered pieces
#cleanJSON(dataString,path,filename)