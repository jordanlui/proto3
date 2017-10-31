% Analysis on various proto recordings, sept24, oct2, etc.
% Comparing motion analysis oct 15 with some stationary data. Testing
% parameters to see if drift happens, and see if how filters perform

close all;                     	% close all figures
clear;                         	% clear all variables
clc;          

%% Movement data Analysis
% Path load files
sourceDir = '../Data/oct15/';
aFile = 'reach up forward swing return fast.csv';
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
[pos,displacement,checkReturnCentre,accgyr_orig,acc,gyr,vel, mcuFreq] = deadReckonGeneral(dataPath,filtLPF,filtHPF,stationaryThreshold); % General Model

%% Comparison with Stationary data
sourceDir = '../Data/oct14/';
aFile = 'stationary.csv';
dataPath = strcat(sourceDir,aFile);

[spos,sdisplacement,scheckReturnCentre,accgyr_orig_stationary,sacc,sgyr,svel, smcuFreq] = deadReckonGeneral(dataPath,filtLPF,filtHPF,stationaryThreshold); % General Model

%% Frequency Analysis between Motion and stationary (raw data)
L = length(acc);
Fs = mcuFreq;
f = Fs*(0:(L/2))/L;
plotTitles={'acc x moving raw', 'y raw', 'z raw','acc x stationary raw', 'y raw', 'z raw'};
yLims = [1 1 10];
figure()
for i = 1:3
    % Accelerometer fft (moving)
    subplot(2,3,i)
    hold on
    Y = fft(accgyr_orig(:,i));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    plot(f,P1)
    plotTitle = plotTitles{i};
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([-1 20])
%     ylim([0 0.1])

    % Accelerometer fft (stationary)
    subplot(2,3,i+3)
    hold on
    Y = fft(accgyr_orig_stationary(:,i));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    plot(f,P1)
    plotTitle = plotTitles{i+3};
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([-1 20])
%     ylim([0 0.1])
   
end

%% Gyro analysis
L = length(acc);
Fs = mcuFreq;
f = Fs*(0:(L/2))/L;
plotTitles={'gyro x moving raw', 'y raw', 'z raw','gyro x stationary raw', 'y raw', 'z raw'};
yLims = [1 1 10];
figure()
for i = 1:3
    % Gyro fft (moving)
    subplot(2,3,i)
    hold on
    Y = fft(accgyr_orig(:,i+3));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    plot(f,P1)
    plotTitle = plotTitles{i};
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([-1 20])
%     ylim([0 0.1])

    % Gyro fft (stationary)
    subplot(2,3,i+3)
    hold on
    Y = fft(accgyr_orig_stationary(:,i+3));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    plot(f,P1)
    plotTitle = plotTitles{i+3};
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([-1 20])
%     ylim([0 0.1])
   
end