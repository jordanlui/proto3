% Dead reckoning script adapated from Madgwick. Focused on high oscillation
% data filtering, so only HPF performed
% https://github.com/xioTechnologies/Oscillatory-Motion-Tracking-With-x-IMU


%% Housekeeping

function [linPosHP, displacement, checkReturnCentre] = deadReckon(dataPath,mcuFreq,filtCutOff)

addpath('Libraries/ximu_matlab_library');	% include x-IMU MATLAB library
addpath('Libraries/quaternion_library');
addpath('Libraries/MahonyAHRS');
addpath('Libraries');
close all;                     	% close all figures
% clear;                         	% clear all variables
% clc;                          	% clear the command terminal
 
%% Import data

% Import my own array

% files = {'processed_fwd back 5x 30cm.csv', 'processed_move forward back.csv', 'processed_original proto data.csv', 'processed_proto modified.csv', 'processed_reach forward swing and back.csv', 'processed_sitting on desk near me.csv', 'processed_walk around lab.csv', 'processed_walk random.csv' };
% path = '../Data/sept29/';
% fileSelect = 5; % Choose your file here
% fullpath = strcat(path,files(fileSelect));


dataTemp = csvread(char(dataPath),1,0); % Load the data

time = dataTemp(:,1);
packets = dataTemp(:,2);
acc = dataTemp(:,3:5);
gyr = dataTemp(:,6:8);
% Note some unit conversions are required
% mcuFreq = 26; % MCU Recording frequency, in Hz
samplePeriod = 1/mcuFreq; % Period is 1/frequency
% filtCutOff = 0.100;
cutoffFreq = (2*filtCutOff)/(1/samplePeriod);
% cutoffFreq = 9.8e-4;


%% Plot stuff
% Plot
figure('Name', 'Gyroscope');
hold on;
plot(gyr(:,1), 'r');
plot(gyr(:,2), 'g');
plot(gyr(:,3), 'b');
xlabel('sample');
ylabel('dps');
title('Gyroscope');
legend('X', 'Y', 'Z');

figure('Name', 'Accelerometer');
hold on;
plot(acc(:,1), 'r');
plot(acc(:,2), 'g');
plot(acc(:,3), 'b');
xlabel('sample');
ylabel('g');
title('Accelerometer');
legend('X', 'Y', 'Z');

%% Process data through AHRS algorithm (calcualte orientation)
% See: http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/

R = zeros(3,3,length(gyr));     % rotation matrix describing sensor relative to Earth

ahrs = MahonyAHRS('SamplePeriod', samplePeriod, 'Kp', 1);

for i = 1:length(gyr)
    ahrs.UpdateIMU(gyr(i,:) * (pi/180), acc(i,:));	% gyroscope units must be radians
    R(:,:,i) = quatern2rotMat(ahrs.Quaternion)';    % transpose because ahrs provides Earth relative to sensor
end

%% Calculate 'tilt-compensated' accelerometer

tcAcc = zeros(size(acc));  % accelerometer in Earth frame

for i = 1:length(acc)
    tcAcc(i,:) = R(:,:,i) * acc(i,:)';
end

% Plot
figure('Name', '''Tilt-Compensated'' accelerometer');
hold on;
plot(tcAcc(:,1), 'r');
plot(tcAcc(:,2), 'g');
plot(tcAcc(:,3), 'b');
xlabel('sample');
ylabel('g');
title('''Tilt-compensated'' accelerometer');
legend('X', 'Y', 'Z');

%% Calculate linear acceleration in Earth frame (subtracting gravity)

linAcc = tcAcc - [zeros(length(tcAcc), 1), zeros(length(tcAcc), 1), ones(length(tcAcc), 1)];
linAcc = linAcc * 9.81;     % convert from 'g' to m/s/s

% Plot
figure('Name', 'Linear Acceleration');
hold on;
plot(linAcc(:,1), 'r');
plot(linAcc(:,2), 'g');
plot(linAcc(:,3), 'b');
xlabel('sample');
ylabel('g');
title('Linear acceleration');
legend('X', 'Y', 'Z');

%% Calculate linear velocity (integrate acceleartion)

linVel = zeros(size(linAcc));

for i = 2:length(linAcc)
    linVel(i,:) = linVel(i-1,:) + linAcc(i,:) * samplePeriod;
end

% Plot
figure('Name', 'Linear Velocity');
hold on;
plot(linVel(:,1), 'r');
plot(linVel(:,2), 'g');
plot(linVel(:,3), 'b');
xlabel('sample');
ylabel('g');
title('Linear velocity');
legend('X', 'Y', 'Z');

%% High-pass filter linear velocity to remove drift

% order = 1;
% % filtCutOff = 0.1;
% [b, a] = butter(order, cutoffFreq, 'high');
% linVelHP = filtfilt(b, a, linVel);

%% Try filter from Madgwick Gait
% acc_mag = sqrt(accX.*accX + accY.*accY + accZ.*accZ);
acc_mag = sqrt(sum(acc.^2));

% HP filter accelerometer data
filtCutOff = 0.001;
filtHPF = (2*filtCutOff)/(1/samplePeriod);
% filtHPF = 7.8e-6;
[b, a] = butter(1, filtHPF, 'high');
acc_magFilt = filtfilt(b, a, acc_mag);

% Compute absolute value
acc_magFilt = abs(acc_magFilt);

% LP filter accelerometer data
filtCutOff = 5;
filtLPF = (2*filtCutOff)/(1/samplePeriod);
% filtLPF = 0.04;
[b, a] = butter(1, filtLPF, 'low');
acc_magFilt = filtfilt(b, a, acc_magFilt);

% Threshold detection
stationary = acc_magFilt < 0.05;

%% Plot
figure('Name', 'High-pass filtered Linear Velocity');
hold on;
plot(linVelHP(:,1), 'r');
plot(linVelHP(:,2), 'g');
plot(linVelHP(:,3), 'b');
xlabel('sample');
ylabel('g');
title('High-pass filtered linear velocity');
legend('X', 'Y', 'Z');

%% Calculate linear position (integrate velocity)

linPos = zeros(size(linVelHP));

for i = 2:length(linVelHP)
    linPos(i,:) = linPos(i-1,:) + linVelHP(i,:) * samplePeriod;
end

% Plot
figure('Name', 'Linear Position');
hold on;
plot(linPos(:,1), 'r');
plot(linPos(:,2), 'g');
plot(linPos(:,3), 'b');
xlabel('sample');
ylabel('g');
title('Linear position');
legend('X', 'Y', 'Z');

%% High-pass filter linear position to remove drift

order = 1;
% filtCutOff = 0.1;
[b, a] = butter(order, cutoffFreq, 'high');
linPosHP = filtfilt(b, a, linPos);

% Plot
figure('Name', 'High-pass filtered Linear Position');
hold on;
plot(linPosHP(:,1), 'r');
plot(linPosHP(:,2), 'g');
plot(linPosHP(:,3), 'b');
xlabel('sample');
ylabel('g');
title('High-pass filtered linear position');
legend('X', 'Y', 'Z');

%% Custom code from Jordan

displacement = sqrt(sum( (max(linPosHP) - min(linPosHP)).^2 )); % Check displacement from recording. Note this can be a naieve calculation
checkReturnCentre = sqrt(sum( (linPosHP(end,:) - linPosHP(1,:)).^2 ));

%% Play animation

SamplePlotFreq = 8;

SixDOFanimation(linPosHP, R, ...
                'SamplePlotFreq', SamplePlotFreq, 'Trail', 'Off', ...
                'Position', [9 39 1280 720], ...
                'AxisLength', 0.1, 'ShowArrowHead', false, ...
                'Xlabel', 'X (m)', 'Ylabel', 'Y (m)', 'Zlabel', 'Z (m)', 'ShowLegend', false, 'Title', 'Unfiltered',...
                'CreateAVI', true, 'AVIfileNameEnum', false, 'AVIfps', ((1/samplePeriod) / SamplePlotFreq));            

%% Plot the movement path (xy plane)
% linPosHPSelect = linPosHP(round(0.25 * length(linPosHP)):end,:); % Use
% this if you want to trim off any starting transient
linPosHPSelect = linPosHP;
x = linPosHPSelect(:,1);
y = linPosHPSelect(:,2);
figure()
plot(x,y)
plotTitle = sprintf('xy position for "%s", filter cutoff %.2f', dataPath,filtCutOff);
% xlabel = 'x position (m)';
% ylabel = 'y position (m)';
title(plotTitle)
            
            
%% End of script