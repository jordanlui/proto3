# Functions for saving to CSV

import time
from os.path import exists
import csv
import numpy as np

def filePath():
	path = "../data/"
	timestamp = time.strftime("%Y%m%d_")
	filename = str(path + timestamp+ "log.csv")
	return filename

def WriteProtoCSV(datalist,path,filename='log.csv'):
	# Write JSON data to CSV File
	global csv_success
	headerTimeIMU = ['time', 'packet', 'AcX', 'AcY', 'AcZ', 'gX', 'gY', 'gZ', 'DL1', 'DS1', 'DL2', 'DS2']
	header = np.hstack(( headerTimeIMU, np.repeat("O16",16), np.repeat("O8",8) ))
	filename = path + 'processed_' + filename # Prefix for a new filename
	f = csv.writer(open(filename,"w+"), lineterminator='\n')
	count = 0
	for r in datalist:
		if count == 0:
			f.writerow(header)
			count += 1
		rowTimeIMU = [ r['time'], r['packet'], r['AcX'], r['AcY'], r['AcZ'], r['GyX'], r['GyY'], r['GyZ'], r['DL1'], r['DS1'], r['DL2'], r['DS2']]
		omron16 = r['O16-1']
		omron8 = r['O8-1']
		rowPrimer = np.hstack((rowTimeIMU,omron16,omron8))
		f.writerow(rowPrimer)


def WriteToCSV(datalist):
	""" This function accepts data and writes to CSV file. 
	A better optimization on this CSV script for the future would be for the CSV write to recognize the JSON hierarchy and 
	write the CSV according to JSON structure.
	"""

	global csv_success
	# Define header
	header = ['time', 'accx', 'accy', 'accz', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']

	# Define our filename
#	ts = time.time()
	filename = filePath()

	# Handling to open our file if it exists or create new one
	if exists(filename):
		# try: 
		f = csv.writer(open(filename,"a"),lineterminator='\n')
			# break
		for row in datalist:
			f.writerow(row.values())
	else:
		f = csv.writer(open(filename,"a+"),lineterminator='\n')
		# Write our header line out if this is a new file
		count = 0
		for row in datalist:
			if count == 0:
				header = row.keys()
				f.writerow(header)
				count += 1
			f.writerow(row.values())
		
	try: # See if we have a multi element list of data to save
		len(datalist)
		if len(datalist) > 1:
			for data in datalist:
				f.writerow([ data['time'], data['acc'][0],data['acc'][1],data['acc'][2],data['gyro'][0],data['gyro'][1],data['gyro'][2],data['mag'][0],data['mag'][1],data['mag'][2] ])
				
	except: # If we don't, we save a single row

		data = datalist
		f.writerow([ data['time'], data['acc'][0],data['acc'][1],data['acc'][2],data['gyro'][0],data['gyro'][1],data['gyro'][2],data['mag'][0],data['mag'][1],data['mag'][2] ])
	
	
	csv_success = True
	return csv_success

def WriteToCSVAccGyro(datalist):
	""" This function accepts data and writes to CSV file. 
	Accelerometer and Gyroscope, but no magnetometer
	A better optimization on this CSV script for the future would be for the CSV write to recognize the JSON hierarchy and 
	write the CSV according to JSON structure.
	"""

	global csv_success
	# Define header
	header = ['time', 'packet', 'accx', 'accy', 'accz', 'gx', 'gy', 'gz']

	# Define our filename
	filename = filePath()

	# Handling to open our file if it exists or create new one
	if exists(filename):
		# try: 
		f = csv.writer(open(filename,"a"),lineterminator='\n')
			# break
		# except:
	else:
		f = csv.writer(open(filename,"a+"),lineterminator='\n')
		# Write our header line out if this is a new file
		f.writerow(header)
		
	try:
		len(datalist)
		if len(datalist) > 1:
			for data in datalist:
				f.writerow([ data['time'], data['packet'], data['acc'][0],data['acc'][1],data['acc'][2],data['gyro'][0],data['gyro'][1],data['gyro'][2] ])
	except:
		data = datalist
		f.writerow([ data['time'], data['packet'], data['acc'][0],data['acc'][1],data['acc'][2],data['gyro'][0],data['gyro'][1],data['gyro'][2] ])
	
	
	csv_success = True
	return csv_success

def WriteToCSV_wPacket(datalist):
	""" This function accepts data and writes to CSV file. 
	A better optimization on this CSV script for the future would be for the CSV write to recognize the JSON hierarchy and 
	write the CSV according to JSON structure.
	"""

	global csv_success
	# Define header
	header = ['time', 'packet', 'accx', 'accy', 'accz', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']

	# Define our filename
	filename = filePath()

	# Handling to open our file if it exists or create new one
	if exists(filename):
		# try: 
		f = csv.writer(open(filename,"a"),lineterminator='\n')
			# break
		# except:
	else:
		f = csv.writer(open(filename,"a+"),lineterminator='\n')
		# Write our header line out if this is a new file
		f.writerow(header)
		


	
	f.writerow([ datalist['time'], datalist['packet'], datalist['acc'][0],datalist['acc'][1],datalist['acc'][2],datalist['gyro'][0],datalist['gyro'][1],datalist['gyro'][2],datalist['mag'][0],datalist['mag'][1],datalist['mag'][2] ])
	
	
	csv_success = True
	return csv_success