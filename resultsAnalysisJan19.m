% Results analysis jan 19 exercises

clc
clear all
close all
files = dir('../analysis/jan19/*.csv');

% Set some run parameters
plotSkipping = 30;
freq = 60; % Frequency for device and ART data from LabVIEW. Should be 60 Hz, but might not be perfect.
%% Load data and clip to proper start and end
afile = files(1).name;
data = csvread(strcat(files(1).folder,'/',afile));
data = data(2243:7125,:); % Clip data
[time, acc, gyr, quat, omron, arm, forearm, wrist] = parseData(data);
plotJoints(arm,forearm,wrist,afile)
distanceWristArm = sqrt(sum((wrist-arm).^2,2)); % Distance values
X = [omron acc gyr quat];
y = distanceWristArm;
data1 = [y X];

data1 = matNorm(data1);

afile = files(2).name;
data = csvread(strcat(files(1).folder,'/',afile));
data = data(1149:6344,:);
[time, acc, gyr, quat, omron, arm, forearm, wrist] = parseData(data);
plotJoints(arm,forearm,wrist,afile)
distanceWristArm = sqrt(sum((wrist-arm).^2,2)); % Distance values
X = [omron acc gyr quat];
y = distanceWristArm;
data2 = [y X];
data2 = matNorm(data2);

afile = files(3).name;
data = csvread(strcat(files(1).folder,'/',afile));
data = data(1121:6179,:);
[time, acc, gyr, quat, omron, arm, forearm, wrist] = parseData(data);
plotJoints(arm,forearm,wrist,afile)
distanceWristArm = sqrt(sum((wrist-arm).^2,2)); % Distance values
X = [omron acc gyr quat];
y = distanceWristArm;
data3 = [y X];
data3 = matNorm(data3);

% Combine our datasets
datas = {data1 data2 data3};
M = length(datas);

%% Dead Reckon with Madgwick Algorithm
% Data Prep (Always pass clean data to algorithm!)
% gyrCalStationary = [-2.172099087	1.585397653	2.456323338];
load('BITcalibration20170125.mat')

% Runtime parameters
accScale = 8192;
gyrScale = 16.4;

accIn = acc./accScale;
gyrIn = gyr - gyrCal;
gyrIn = gyrIn./gyrScale;
pos3 = deadReckonMadgwickOscillationFunc(accIn,gyrIn,freq,0.1);

figure()
subplot(1,2,1)
hold on
plot(pos3(:,1))
plot(pos3(:,2))
plot(pos3(:,3))
ylabel('Position (m)')
xlabel('Time (s)')
title('Dead Reckon Position')
hold off
subplot(1,2,2)
hold on
plot(wrist(:,1)/1e3)
plot(wrist(:,2)/1e3)
plot(wrist(:,3)/1e3)
ylabel('Position (m)')
xlabel('Time (s)')
title('Motion Tracker Wrist Position')
hold off
1
% Output data to file
% csvwrite('data1.csv',data)
% csvwrite('position1.csv',pos3)
%% Build Model
% After creating model in Regression Learner
% [trainedModel, validationRMSE] = trainRegressionModel(data1);
% ypred = trainedModel.predictFcn(data1(:,2:end));
% 
% figure()
% hold on
% plot(ypred)
% plot(data1(:,1))
% legend('predict','real')
% hold off
% title('Validation of model')

%% Check against other dataset


% ypred = trainedModel.predictFcn(data2(:,2:end));
% 
% figure()
% hold on
% plot(ypred)
% plot(data2(:,1))
% legend('predict','real')
% hold off
% title('Check against test data (unseen)')

%%


%% Loop Through data and apply fine tree regression learning
for i=1:M
    % Construct our data sets
    testData = datas{i};
    ytestmm = testData(:,1); % Test data in mm
    trainData = [];
    for j = 1:M
        if j ~= i
            trainData = [trainData; datas{j}];
        end
    end
    % Train model
    normStats = {mean(trainData) max(trainData) min(trainData)};
    trainData = (trainData - mean(trainData)) ./ (max(trainData) - min(trainData));
    testData = (testData - mean(trainData)) ./ (max(trainData) - min(trainData));
    [trainedModel, validationRMSE] = trainRegressionModel(data1);
    
    
    % Test Model
    ypred = trainedModel.predictFcn(testData(:,2:end));
    % Convert prediction back into mm values
    ypredmm = ypred * (max(trainData) - min(trainData)) + mean(trainData);
    
    % Test error calculation
    testError = ypredmm - ytestmm;
    
    figure()
    hold on
    plot(ypredmm)
    plot(ytestmm)
    legend('prediction','real')
    title('Result plot')
    hold off
    
    
    validationRMSE
    % MSE calculation
    mean(sum(testError.^2))
    
end

%% End of Script
%% Functions
function dataOut = matNorm(data)

    % Columnwise norm of matrix
    dataOut = (data - mean(data)) ./ (max(data) - min(data));
end

function plotJoints(arm,forearm,wrist,caption)
    % Plots ART joint data
    if nargin > 3
        pltTitle = caption;
    else
        pltTitle = '3D plot';
    end
    circSize = 1;
    figure()
    hold on
    scatter3(arm(:,1),arm(:,2),arm(:,3),circSize,'r')
    scatter3(forearm(:,1),forearm(:,2),forearm(:,3),circSize,'b')
    scatter3(wrist(:,1),wrist(:,2),wrist(:,3),circSize,'g')
    hold off
    xlabel('x')
    ylabel('y')
    zlabel('z')
    legend('arm','forearm','wrist')
    title(pltTitle)
end

function [time, acc, gyr, quat, omron, arm, forearm, wrist] = parseData(data)
    % Plots ART and proto IMU/omron data
    time = data(:,1);
    omron = data(:,2:17);
    acc = data(:,18:20);
    gyr = data(:,21:23);
    quat = data(:,24:27);
    arm = data(:,28:30);
    forearm = data(:,31:33);
    wrist = data(:,34:36);
end