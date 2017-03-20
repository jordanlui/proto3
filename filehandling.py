# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 18:37:51 2017

@author: Jordan
Rename files and other things, as data prep for analysis
Note that file renaming for the Feb 24 data set is quite simple. I overthought the problem. Extract each of the numbers, shuffle, and rename the file.

"""


# Libraries
import glob,os, re, shutil

# Operating Variables
dir1 = '../Data/armruler_feb24' # The source directory
dir2 = '../Data/armruler_feb24/test1' # The output directory

# Grab a list of CSV files in the folder. Note this search should be modified if working with non CSV files
filelist = glob.glob(os.path.join(dir1,'*.csv'))

# Define each of our file operations here
def digitshift():
    """    
    # Performs digit shifting, putting the value from one character position into another character position
    # Current config will shift [a][b][c] to [d][c][a] (the [b] was junk data)
    Should accept 3 integer addresses and change to a new order
    """
    return 0

# Function to create a new file name required?
def mkfilename(a,b,c):
    """
    Generate new filename based on 3 inputs
    """    
    return '['+str(a)+']_['+str(b)+']_['+str(c)+'].csv'
# Loop through our files one at a time and perform whichever desired operation
for file in filelist:
    # extract the trial info from the files. The data in the []'s
    trial_info = re.findall('\[([0-9]{1,2})\]',file)
    a=trial_info[0]
    b=trial_info[1]
    c=trial_info[2]
#    print trial_info
    
    # Generate the new filename
    newname = mkfilename(1,c,a)
    print newname
    
    # Rename of our file
    oldfilepath = file
    newfilepath = os.path.join(dir2,newname)
#    os.rename(oldfilepath,newfilepath)
    shutil.copy(oldfilepath,newfilepath)
    

# Reconstruct the new file name based on our 
