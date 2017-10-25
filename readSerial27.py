# Simple read from serial port

from __future__ import division
from __future__ import print_function
import serial
#import sys
import time
import csv
#import json
import decodeBytes27
import myFunctions
import numpy as np


# Useful variables
mcuFreq = 25 # Microcontroller frequency, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds
#numBytes = 33 * 4 + 8
testLength = 25 #loop will run for testLength seconds

#select what to do with data 0 = store 1 = decode
storeOrDecode = 0
packetsReceived = 0

# File save parameters
path = 'data/'
fileName = 'walk.csv'
pathOut = path + fileName

# Make serial connection
serial = serial.Serial("COM12", 115200, timeout=None)
if serial:
    print('connected')
#    print('Packet Runtime DL1   DS1   DL2   DS2    AcX               AcY                  AcZ')

timeout = time.time() + testLength

datalist = []

# Recording
while True:
    data = serial.readline()
    if len(data) > 0:
        if storeOrDecode == 0:
            datalist.append(data)
        else:
            decodeBytes27.decode(data)
        packetsReceived += 1
    #while loop times out in testLength seconds
    if time.time() > timeout:
        break
    time.sleep(mcuPeriod)

# Stats on result
myFunctions.printStats(storeOrDecode,packetsReceived)

# Decode our data into a list
decodedData = []
for row in datalist:
    try:
        dataOut = decodeBytes27.decode(row, printOut=0)
        if dataOut != 0:
            print(dataOut)
            decodedData.append(dataOut)
    except 0:
        print('Error')
            
# Save to file
np.savetxt(pathOut, decodedData, delimiter=",", fmt='%s')

# Close out
serial.close()