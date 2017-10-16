# Simple read from serial port

from __future__ import division
from __future__ import print_function
import serial
#import sys
import time
#import csv
#import json
import decodeBytes27
import myFunctions



# Useful variables
mcuFreq = 50 # Microcontroller frequency, in Hz
mcuPeriod = 1 / mcuFreq # Python loop timing, in seconds
numBytes = 33 * 4 + 8
testLength = 5 #loop will run for testLength seconds

#select what to do with data 0 = store 1 = decode
storeOrDecode = 1 

# Make serial connection
serial = serial.Serial("COM10", 115200, timeout=None)
if serial:
    print('connected')
    print('Packet Runtime DL1   DS1   DL2   DS2    AcX               AcY                  AcZ')

timeout = time.time() + testLength

while True:
    data = serial.readline()
    if len(data) > 0:
        if storeOrDecode == 0:
            myFunctions.storeData(data.strip())
        else:
            decodeBytes27.decode(data.strip())
    
    #while loop times out in 5 seconds
    if time.time() > timeout:
        break
    time.sleep(mcuPeriod)

if storeOrDecode == 0:
    myFunctions.printStats(storeOrDecode)