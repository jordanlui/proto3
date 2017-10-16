% Analysis on various proto recordings, sept24, oct2, etc.

close all;                     	% close all figures
clear;                         	% clear all variables
clc;          

%% Load Files

% Path load files
% Oct 2 data
sourceDir = '../Data/oct15/';
files = dir([sourceDir, '\*.csv']); % Grab the files in directory
numfiles = length(files(not([files.isdir]))); 
pickFile = 6; % Pick the file to analyze
aFile = files(pickFile).name;
dataPath = strcat(sourceDir,aFile);
%% Analysis of results

% General Madgwick approach
% Common Default Parameters
% filtHPF = 0.001;
% filtLPF = 5;
% stationaryThreshold = 0.01;

% Param Experimentation
filtHPF = 0.001; % Actual Filter param in Hz
filtLPF = 5; % Actual Filter param in Hz
stationaryThreshold = 0.019;
[pos,displacement,checkReturnCentre] = deadReckonGeneral(dataPath,filtLPF,filtHPF,stationaryThreshold); % General Model

%% Analysis of all files in a folder
% results = cell(numfiles,5);
% 
% for i = 1:numfiles
%     pickFile = i;
%     aFile = files(pickFile).name;
%     dataPath = strcat(sourceDir,aFile);
%     
%     [pos,displacement, maxDisplacement3Axis] = deadReckonGeneral(dataPath,filtLPF,filtHPF,stationaryThreshold); % General Model
%     % Store results
%     results{i,1} = aFile;
%     results{i,2} = displacement;
%     results{i,3} = filtHPF;
%     results{i,4} = filtLPF;
%     results{i,5} = stationaryThreshold;
% end
% outputName = strcat(sourceDir,'analysis/','results.csv');
% T = cell2table(results,'VariableNames',{'file','Displacement','HPF','LPF','Thresh'});
% writetable(T,outputName)
