% Analysis on various proto recordings, sept24, oct2, etc.



close all;                     	% close all figures
clear;                         	% clear all variables
clc;          

% single movement
% dataPath = 'data/20170924/move_cross.csv';
% mcuFreq = 43; % MCU Recording frequency, in Hz
% filtCutOff = 0.08;
% [linPosHP] = deadReckon(dataPath,mcuFreq,filtCutOff);
% displacement1 = sqrt(sum( (max(linPosHP) - min(linPosHP)).^2 ));
% checkReturnCentre = sqrt(sum( (linPosHP(end,:) - linPosHP(1,:)).^2 ));

% Analysis Sept 29 data

files = {'processed_fwd back 5x 30cm.csv', 'processed_move forward back.csv', 'processed_original proto data.csv', 'processed_proto modified.csv', 'processed_reach forward swing and back.csv', 'processed_sitting on desk near me.csv', 'processed_walk around lab.csv', 'processed_walk random.csv' };
sourceDir = '../Data/sept29/';
fileSelect = 6; % Choose your file here
aFile = char(files(fileSelect));
dataPath = strcat(sourceDir,aFile);

mcuFreq = 16; % MCU Recording frequency, in Hz

% Oct 2 data
sourceDir = '../Data/oct2/';
files = dir([sourceDir, '\*.csv']); % Grab the files in directory
numfiles = length(files(not([files.isdir]))); 
pickFile = 5; % Pick the file to analyze
aFile = files(pickFile).name;
dataPath = strcat(sourceDir,aFile);

%% Filter analysis on stationary device
% File 6 was recorded with proto on desk. It shouldn't be moving.
% filterRanges = [0.01:0.01:0.1]';
% movement = [];
% 
% for i = 1:length(filterRanges)
%     filtCutOff = filterRanges(i);
%     outputString = 'Analysis on %s \n';
%     fprintf(outputString,singleFile);
%     [linPosHP,displacement,checkReturnCentre] = deadReckon(dataPath,mcuFreq,filtCutOff);
%     movement(i,1) = displacement;
%     movement(i,2) = checkReturnCentre;
% end
% moveSummary = [filterRanges movement];
%%
% Frequency analysis
dataTemp = csvread(dataPath,1,0); % Skip the header
time = dataTemp(:,1);
packets = dataTemp(:,2);
acc = dataTemp(:,3:5); % Accelerometer data, g values
gyr = dataTemp(:,6:8); % Gyro data, degrees per second
M = length(packets);
imu = [acc gyr];

% FFT Plots of Accelerometer and Gyro
figure(1)

for i = 1:6
    subplot(2,3,i)
    Y = fft(imu(:,i));
    L = M;
    Fs = mcuFreq;
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = Fs*(0:(L/2))/L;
    plot(f,P1)
    plotTitle = sprintf('Single-Sided FFT of %s',aFile);
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
end

%% Regular Single analysis
% filtCutOff = 0.03;
% outputString = 'Analysis on %s \n';
% fprintf(outputString,aFile);
% % [linPosHP,displacement,checkReturnCentre] = deadReckon(dataPath,mcuFreq,filtCutOff); % Oscillation model
% 
% % Try a new general Madgwick approach
% filtHPF = 0.001;
% filtLPF = 5;
% stationaryThreshold = 0.05;
% [pos,displacement,checkReturnCentre] = deadReckonGeneral(dataPath,mcuFreq,filtLPF,filtHPF,stationaryThreshold); % General Model


