% Analysis on Sept 24 data
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
path = '../Data/sept29/';
fileSelect = 6; % Choose your file here
singleFile = char(files(fileSelect));
dataPath = strcat(path,singleFile);

mcuFreq = 16; % MCU Recording frequency, in Hz



%% Filter analysis on stationary device
filterRanges = [0.01:0.025:1];
% for i = 1:length(filterRanges)
%     filtCutOff = filterRanges(i);
%     
%     
% end


%% Regular Single analysis
filtCutOff = 0.09;
outputString = 'Analysis on %s \n';
fprintf(outputString,singleFile);
[linPosHP] = deadReckon(dataPath,mcuFreq,filtCutOff);
displacement1 = sqrt(sum( (max(linPosHP) - min(linPosHP)).^2 ));
checkReturnCentre = sqrt(sum( (linPosHP(end,:) - linPosHP(1,:)).^2 ));

%% Plot the positions occupied
linPosHPSelect = linPosHP(round(0.25 * length(linPosHP)):end,:);
x = linPosHPSelect(:,1);
y = linPosHPSelect(:,2);
figure()
plot(x,y)
plotTitle = sprintf('xy position for "%s", filter cutoff %.2f', singleFile,filtCutOff);
title(plotTitle)
