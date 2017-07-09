# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:22:19 2016

Creates x matrix from CSV files.

Updating July 2017 
Note that motion tracking device as of June 2017 is labeled by the x-y coordinates, 
not the patient/class/trial info in the csv files.

July 9: Script will read each folder in a list and make a separate x matrix for each. Easier for a shuffle operation possibly


@author: Jordan
"""
# Read all of the CSV files in the directory and extract the min, max, and average value per trial on two axis and outputs

# Libraries
from __future__ import division
import numpy as np
import glob, os

# Import our custom functions
from xmatrix import xmatrix

#%% Parameters
# File paths for accessing data

path ='../Data/june23/1/'
# Declare a paths variable here if processing multiple folderss
path = ['../Data/june23/1/','../Data/june23/2/','../Data/june23/3/','../Data/june23/4/','../Data/june23/5/','../Data/june23/6/','../Data/june23/7/']
output_dir = '../Data/june23/analysis/'
output_file = 'analysis.csv'
output_path = os.path.join(output_dir,output_file)

# Initial variables
#nancount = 0

#%% Functions
def genfilelist(path):
    # generates array of all csv files in the directory
    filelist = []    
    if isinstance(path,(str,unicode))  == True: # Checks if the path is a single folder (string format), or a list
        # Then we hav a single path to generate files
        filelist = glob.glob(os.path.join(path,'*.csv'))
    else:
        # we should have a list or tuple of folder paths
        for path in paths:
            filelist.extend(glob.glob(os.path.join(path,'*.csv')))
    return filelist

def extractx(path):
    # All matching CSV files in a folder are read and generated into an x matrix
    filelist = genfilelist(path)#   Generate file list from one or more paths
    numfiles = len(filelist)    #   Number of files
    
    # Generate xmatrix with all files in filelist
    x,patient,gesture,trial,nancount  = xmatrix(filelist)  
    #%% Special Processing - May change with each hardware build
    # Note that June / July 2017 data outputs have xy coordinates at end of table. For consistency, want them at front.
    data_sensors = x[:,0:55] # x and y coordinates
    data_coord = x[:,55:] # rest of sensor data
    # Note a coordinate transform or remapping be required. Not necessary now.
    
    # Derive a distance value
    distance = np.sqrt( data_coord[:,0]**2 + data_coord[:,1]**2)
    distance = np.reshape(distance,(len(distance),1))
    newx = np.hstack((data_coord,distance,data_sensors)) # save to a new array
    x = newx
    
    #%% Combine arrays and save
    # Make the big x matrix
    xx = np.hstack((patient,gesture,trial,x))
    
    # Save to CSV File so we can use it later
    np.savetxt(path+"x.csv",x,delimiter=",")
    np.savetxt(path+"xx.csv",xx,delimiter=",")
    np.savetxt(path+"patient.csv",patient,delimiter=",")
    np.savetxt(path+"gesture.csv",gesture,delimiter=",")
    np.savetxt(path+"trial.csv",trial,delimiter=",")
#    return 0
#%% Main code
# Generate File list
    
# July 9. We take each of the 7 recordings from june23 and make an x matrix for each of them. Guess we won't normalize this data if we keep separate?
for apath in path:
    extractx(apath)



