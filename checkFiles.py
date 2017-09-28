# Check files


from diagnostics import *
import numpy as np

path = "data/"

import glob
import os

# Glob the file list and find latest file
list_of_files = glob.glob(path + '*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

print 'check average frequency value of latest file'

# Analyze this file
data = np.genfromtxt(latest_file, delimiter=",", skip_header=1)
timeStart = data[0,0]
timeEnd = data[-1,0]

# Count the number of packets in each time step
packetCount = []

for i in range(int(timeStart), int(timeEnd)):
	m = [row for row in data if row[0]==i]
	packetCount.append(len(m))
	
print m



# Check packet continuity
