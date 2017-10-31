# Simple read from serial port

from __future__ import division
from __future__ import print_function
from serial import *
#import sys
import time
#import csv
#import json
import decodeBytes27
import myFunctions
import numpy as np


# Useful variables
mcuFreq = 100 # Microcontroller frequency, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds
#numBytes = 33 * 4 + 8

# Run parameters
testLength = 30 #loop will run for testLength seconds

pathout = '../Data/oct30/'
filename = 'labwalk.csv'
#filename = 'timertest.csv'
fullpathOut = pathout + filename

#select what to do with data. 0 = store in a datalist,  1 = decode instantly
storeOrDecode = 0

packetsReceived = 0

# Make serial connection
serial = Serial("COM12", 115200, timeout=None)
if serial:
    print('connected')
#    print('Packet Runtime DL1   DS1   DL2   DS2    AcX       AcY                  AcZ')

timeStart = time.time() # Record current time
timeout = timeStart + testLength # Represents end time of our acquisition
timeRemaining = testLength

datalist = []
dataCleaned = []



while True: # While loop for data recording
    data = serial.readline()
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

# Close the connection?
serial.close()
# Post-processing decoding if we store the data in memory during recording
if storeOrDecode == 0:
    for i in range (0,len(datalist)-1):
        row = datalist[i]
        dataOut = decodeBytes27.decode(row)
        if dataOut != 0:
            dataCleaned.append(dataOut)
        elif dataOut == 0:      # This error check helps to detect lines of bytes that got split into two lines.
            tryrow = row + datalist[i+1]
            dataOut = decodeBytes27.decode(tryrow)
            if dataOut != 0:    # This error check helps to detect lines of bytes that got split into two lines.
                dataCleaned.append(dataOut)
                i = i + 1
           


dataArray = []
# Flatten data into an array
for row in dataCleaned:
    rowA = np.array(row[0:12])
    rowB = np.array(row[-1])
    rowNew = np.hstack((rowA,rowB))
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


