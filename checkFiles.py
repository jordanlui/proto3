# Check integrity of the latest file we've saved


from diagnostics import countPackets, packetContinuity
import numpy as np

path = "../data/sept28/"

import glob
import os

# Glob the file list and find latest file
list_of_files = glob.glob(path + '*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)



data = np.genfromtxt(latest_file, delimiter=",", skip_header=1) # Load data

packetCount,fMin,fMax,fMean,fSTD = countPackets(data[:,0:2]) # Check if the number of packets is consistent in files
if fSTD < 1:
	print 'Frequency (%.2f Hz) seems roughly constant since STDEV is < 1 (%.2f)'%(fMean,fSTD)

# Check packet continuity
checkContinuous = packetContinuity(data)
if checkContinuous:
	print 'Packets appear continuous'
else:
	print 'Packet loss may have occured'
