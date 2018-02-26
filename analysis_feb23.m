% Analysis of Feb 1 data
% IR Band Project


clc
clear all
close all

set(0, 'DefaultLineLineWidth', 2);
tolDistance = 1000; 
jointLabel = {'wrist','forearm','arm','chest','shoulder1','shoulder2','elbow'};

accScale = 8192; % Conversion parameter for accelerometer to g value
gyrScale = 16.4; % Conversion parameter for gyroscope to deg/s
freq = 60;
load('BITcalibration20170125.mat')
path1 = '../Analysis/feb23/';
addpath(path1,'Functions');
files = dir(strcat(path1,'*.csv'));
% files = dir('*.csv');

% dataRaw = csvread(files(9).name); % Queen wave
% dataRaw = dataRaw(775:end,:);

% dataRaw = csvread(files(12).name); % Standing pronation
% dataRaw = dataRaw(794:end,:);

% dataRaw = csvread(files(13).name); % Elbow flex
% dataRaw = dataRaw(660:end,:);
% 
% dataRaw = csvread(files(14).name); % Reach to collar
% dataRaw = dataRaw(775:end,:);

dataRaw = csvread(files(15).name); % Walking
data = jointLoader(dataRaw,tolDistance,jointLabel);

% dataRaw = csvread(files(21).name); % Reaching sideways to the left
% dataRaw = dataRaw(540:2185,:);

% dataRaw = csvread(files(28).name); % Reach for box, unaffected
% dataRaw = dataRaw(660:2230,:);

% dataRaw = csvread(files(30).name); % Reach for box, spastic
% dataRaw = dataRaw(796:end,:);



figure(1)
hold on; plot(data.wrist); plot(data.forearm); plot(data.arm); plot(data.chest); hold off
legend('wrist','forearm','arm','chest'); title('coordinates vs. time')


%% Plots

figure(2)
plotallJoints3D(data.joints,'3d plot all joints',jointLabel)


%%
% close all
[row, col, quadrant] = omronAnalysis(data.omron(data.distCheck,:))


figure(10)
suptitle('Temp distribution by row, column, quadrant')
nbins = 10;
for i = 1:4
    subplot(4,3,3*(i-1)+1)
    histogram(row.mean{i},nbins)
    xlim([210 300])
    
    subplot(4,3,3*(i-1)+2)
    histogram(col.mean{i},nbins)
    xlim([210 300])
    
    subplot(4,3,3*(i-1)+3)
    histogram(quadrant.mean{i},nbins)
    xlim([210 300])
end
saveas(gcf,'histogramRowColQuad.png')

%% Temp changes across different axes of movement
figure(11)
tempDirections(data.omron(data.distCheck,:),row,col,quadrant)
saveas(gcf,'TempSpan.png')


%% Functions
function [row,col,quadrant] = omronAnalysis(omron)
    % Consider Omron variances by columns and by rows
    for i=1:4
        row.ind{i} = i:4:16;
        col.ind{i} = 4*(i-1)+1:4*(i-1)+4;
    end
    quadrant.ind{1} = [1 2 5 6];
    quadrant.ind{2} = [3 4 7 8];
    quadrant.ind{3} = [9 10 13 14];
    quadrant.ind{4} = [11 12 15 16];

    for i = 1:4
        row.mean{i} = mean(omron(:,row.ind{i}),2);
        col.mean{i} = mean(omron(:,col.ind{i}),2);
        quadrant.mean{i} = mean(omron(:,quadrant.ind{i}),2);

        row.std{i} = std(omron(:,row.ind{i}),[],2);
        col.std{i} = std(omron(:,col.ind{i}),[],2);
        quadrant.std{i} = std(omron(:,quadrant.ind{i}),[],2);
    end

end
function tempDirections(omron,row,col,quadrant)
    % Plot temperature differences in various axis directions
    
    suptitle('Temp span across Row, Column, Quadrant')
    hold on
    ylabel('Temp difference')
    xlabel('Time')
    plot(max([row.mean{1},row.mean{2},row.mean{3},row.mean{4}],[],2) - min([row.mean{1},row.mean{2},row.mean{3},row.mean{4}],[],2),'b:')
    plot(max([col.mean{1},col.mean{2},col.mean{3},col.mean{4}],[],2) - min([col.mean{1},col.mean{2},col.mean{3},col.mean{4}],[],2),'b--')
    plot(max([quadrant.mean{1},quadrant.mean{2},quadrant.mean{3},quadrant.mean{4}],[],2) - min([quadrant.mean{1},quadrant.mean{2},quadrant.mean{3},quadrant.mean{4}],[],2),'b.')
    yyaxis right
    plot(mean(omron,2),'r')
    ylabel('Mean temperature')
    hold off
    legend('row','col','quad','mean')
end