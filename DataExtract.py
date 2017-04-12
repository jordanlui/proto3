# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:22:19 2016

@author: Jordan
"""
# Read all of the CSV files in the directory and extract the min, max, and average value per trial on two axis and outputs

import glob, os, csv
import numpy as np
import re

path ='../Data/dec6_1'
output_path = '../Analysis/'
output_file = 'dec6.csv'

filelist = glob.glob(os.path.join(path,'*.csv'))


with open(os.path.join(output_path,output_file),'wb') as outputfile:
    writer = csv.writer(outputfile)
#    Write our header row
    writer.writerow(['patient','gesture','trial','min1','max1','mean1','min2','max2','mean2'])
    for filename in filelist:
    #    Open the file
       print filename
       f = open(filename,'r')
       
       data = np.genfromtxt(filename,delimiter=',')
       while True:
           try:
               min1 = np.nanmin(data[:,0],axis=0)
               max1 = np.nanmax(data[:,0],axis=0)
               mean1 = np.nanmean(data[:,0],axis=0)
               min2 = np.nanmin(data[:,1],axis=0)
               max2 = np.nanmax(data[:,1],axis=0)
               mean2 = np.nanmean(data[:,1],axis=0)
               trial_info = re.findall('\[([0-9]{1,2})\]',string)
               patient = trial_info[0]
               gesture = trial_info[1]
               trial = trial_info[2]
               mylist = [patient,gesture,trial,min1,max1,mean1,min2,max2,mean2]
               writer.writerow(mylist)
               break
           except:
               break
          

   
