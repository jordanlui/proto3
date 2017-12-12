% Analysis on on Nov 3 results
% Proto device with coordinate data from webcam

close all;                     	% close all figures
clear;                         	% clear all variables
clc;          

%% Load Files

% Path load files
% Oct 2 data
sourceDir = '../Analysis/nov3/swing/';
% sourceDir = '../Analysis/nov3/forward/';
files = dir([sourceDir, '\XX*.csv']); % Grab the files in directory. Look for ones in XX1.csv format.
numfiles = length(files(not([files.isdir]))); 
pickFile = 2; % Pick the file to analyze
aFile = files(pickFile).name;
dataPath = strcat(sourceDir,aFile);
calibrationFile = 'calibration_nov3.mat';

%% Calibration Data

% gyrCal = [3.199669131	2.722107856	-3.369993885];
% AccMax = [10.28794097900390	9.93380355834960	9.83210849761962];
% AccMin = [-10.49731349945060	-10.70189952850340	-10.51525974273680];
% save('calibration_nov3', 'gyrCal', 'AccMax', 'AccMin')


%% Analysis of results
% Param Experimentation
filtHPF = 0.0003; % Actual Filter param in Hz % Default 0.001
filtLPF = 6; % Actual Filter param in Hz % Default 5
stationaryThreshold = 0.02; % Default 0.05
[pos,displacement,checkReturnCentre] = deadReckonGeneral(dataPath,filtLPF,filtHPF,stationaryThreshold,calibrationFile); % General Model

% Common Default Parameters
% filtHPF = 0.001;
% filtLPF = 5;
% stationaryThreshold = 0.01;