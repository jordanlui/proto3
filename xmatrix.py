# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:19:25 2017
@author: Jordan
Creates an x-matrix for analysis

Updating July 8 2017.
We no longer need the patient 
"""

import numpy as np
import re

x = []
t = []
global nancount
nancount = 0

def loadfile(file):
    tempfile = []
    global nancount
    data = np.genfromtxt(file,delimiter=',')
            
    # Remove the header row
    data = data[1:,:]
    # Use RegEx to get the gesture number
    trial_info = re.findall('\[([0-9]{1,2})\]',file)
    trial_info = np.asarray(trial_info)
    trial_info = trial_info.astype(int)
#            patient = int(trial_info[0])
#    gesture = int(trial_info[1])
#            trial = int(trial_info[2])
    
    # Loop through the data and save to an array
    for row in data:
        # Use if loop to check for valid row data, ignoring ragged data
        if np.isnan(row).any() == True:
#                    print "nan found in file",file
            nancount = nancount + 1
        else:
            newrow = np.concatenate((trial_info,row),axis=0)                    
            tempfile.append(newrow)

def xmatrix(files):
    # Accepts a list of files, extracts each row, builds x matrix.
    # Updating the file to export a file of following format:
    # [patient class trial -data-]
    # Therefore files with MxN dimension will return a MxN+3 matrix

    # Update function as of July 8 2017 for following output, 60 column array
    # [patient, class, trial, x-coord, y-coord, DistIR, RPY, Omron Sensors]


    
    x = []
    patient = []
    gesture = []
    trial = []
    
    global nancount
    nancount = 0
    
    # Check if we have a single file
    if isinstance(files,str) == True:
        # Then just load the single file and output to x and t
        data = np.genfromtxt(files,delimiter=',')
        data = data[1:,:] # Only consider data from row 1 downwards (due to headers)
        trial_info = re.findall('\[([0-9]{1,2})\]',files)
        gesture = int(trial_info[1])
        for row in data:
            x.append(row)
            t.append(gesture)
            
    # Otherwise we have multiple files and need to iterate through them.
    else:
        
        for file in files:
            # Load the data from csv file  
            data = np.genfromtxt(file,delimiter=',')
            # Remove the header row
            data = data[1:,:]
            # Use RegEx to get the patient, class, trial data
            trial_info = re.findall('\[([0-9]{1,2})\]',file)
            trial_info = np.asarray(trial_info)
            trial_info = trial_info.astype(int)

            
            # Loop through the data and save to an array
            for row in data:
                # Use if loop to check for valid row data, ignoring ragged data
                if np.isnan(row).any() == True:
#                    print "nan found in file",file
                    nancount = nancount + 1
                else:
                    x.append(row)
                    patient.append(trial_info[0])
                    gesture.append(trial_info[1])
                    trial.append(trial_info[2])
    # Reformat as arrays
    x = np.asarray(x)
    numrows = len(patient)
    patient = np.reshape(patient,(numrows,1))
    gesture = np.reshape(gesture,(numrows,1))
    trial = np.reshape(trial,(numrows,1))
    
    return x, patient, gesture, trial, nancount