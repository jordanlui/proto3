function [linPosHP] = deadReckonMadgwickOscillationFunc(acc,gyr,freq,filtFreq,plotBool)
% Function version of Madgwick dead-reckon algorithm. Accepts accelerometer
% and gyro data and returns filtered position data. Acc units should be XX
% and gyro values should be XX.

%% Housekeeping

addpath('Libraries/ximu_matlab_library');	% include x-IMU MATLAB library
addpath('Libraries/quaternion_library');	% include quatenrion library
addpath('Libraries/MahonyAHRS');	% include Mahony AHRS library
if nargin < 4
    % Set default filter frequency if we don't receive an input
    filtFreq = 0.1; 
    plotBool = 1;
end


%% Import data

samplePeriod = 1/freq;
if plotBool
    % Plot
    figure('NumberTitle', 'off', 'Name', 'Gyroscope');
    hold on;
    plot(gyr(:,1), 'r');
    plot(gyr(:,2), 'g');
    plot(gyr(:,3), 'b');
    xlabel('sample');
    ylabel('dps');
    title('Gyroscope');
    legend('X', 'Y', 'Z');

    figure('NumberTitle', 'off', 'Name', 'Accelerometer');
    hold on;
    plot(acc(:,1), 'r');
    plot(acc(:,2), 'g');
    plot(acc(:,3), 'b');
    xlabel('sample');
    ylabel('g');
    title('Accelerometer');
    legend('X', 'Y', 'Z');
end

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

if plotBool
    % Plot
    figure('NumberTitle', 'off', 'Name', '''Tilt-Compensated'' accelerometer');
    hold on;
    plot(tcAcc(:,1), 'r');
    plot(tcAcc(:,2), 'g');
    plot(tcAcc(:,3), 'b');
    xlabel('sample');
    ylabel('g');
    title('''Tilt-compensated'' accelerometer');
    legend('X', 'Y', 'Z');
end

%% Calculate linear acceleration in Earth frame (subtracting gravity)

linAcc = tcAcc - [zeros(length(tcAcc), 1), zeros(length(tcAcc), 1), ones(length(tcAcc), 1)];
linAcc = linAcc * 9.81;     % convert from 'g' to m/s/s

if plotBool
    % Plot
    figure('NumberTitle', 'off', 'Name', 'Linear Acceleration');
    hold on;
    plot(linAcc(:,1), 'r');
    plot(linAcc(:,2), 'g');
    plot(linAcc(:,3), 'b');
    xlabel('sample');
    ylabel('g');
    title('Linear acceleration');
    legend('X', 'Y', 'Z');
    
end

%% Calculate linear velocity (integrate acceleartion)

linVel = zeros(size(linAcc));

for i = 2:length(linAcc)
    linVel(i,:) = linVel(i-1,:) + linAcc(i,:) * samplePeriod;
end



%% High-pass filter linear velocity to remove drift

order = 1;
filtCutOff = filtFreq;
[b, a] = butter(order, (2*filtCutOff)/(1/samplePeriod), 'high');
linVelHP = filtfilt(b, a, linVel);

if plotBool
    % Plot Linear Velocity
    figure('NumberTitle', 'off', 'Name', 'Linear Velocity');
    hold on;
    plot(linVel(:,1), 'r');
    plot(linVel(:,2), 'g');
    plot(linVel(:,3), 'b');
    xlabel('sample');
    ylabel('g');
    title('Linear velocity');
    legend('X', 'Y', 'Z');
    
    % Plot
    figure('NumberTitle', 'off', 'Name', 'High-pass filtered Linear Velocity');
    hold on;
    plot(linVelHP(:,1), 'r');
    plot(linVelHP(:,2), 'g');
    plot(linVelHP(:,3), 'b');
    xlabel('sample');
    ylabel('g');
    title('High-pass filtered linear velocity');
    legend('X', 'Y', 'Z');
end



%% Calculate linear position (integrate velocity)

linPos = zeros(size(linVelHP));

for i = 2:length(linVelHP)
    linPos(i,:) = linPos(i-1,:) + linVelHP(i,:) * samplePeriod;
end



%% High-pass filter linear position to remove drift

order = 1;
filtCutOff = filtFreq;
[b, a] = butter(order, (2*filtCutOff)/(1/samplePeriod), 'high');
linPosHP = filtfilt(b, a, linPos);

if plotBool
    % Plot
    figure('NumberTitle', 'off', 'Name', 'Linear Position');
    hold on;
    plot(linPos(:,1), 'r');
    plot(linPos(:,2), 'g');
    plot(linPos(:,3), 'b');
    xlabel('sample');
    ylabel('g');
    title('Linear position');
    legend('X', 'Y', 'Z');
    
    % Plot
    figure('NumberTitle', 'off', 'Name', 'High-pass filtered Linear Position');
    hold on;
    plot(linPosHP(:,1), 'r');
    plot(linPosHP(:,2), 'g');
    plot(linPosHP(:,3), 'b');
    xlabel('sample');
    ylabel('g');
    title('High-pass filtered linear position');
    legend('X', 'Y', 'Z');

end


%% Play animation

SamplePlotFreq = 30;
if plotBool
    SixDOFanimation(linPosHP, R, ...
                    'SamplePlotFreq', SamplePlotFreq, 'Trail', 'DotsOnly', ...
                    'Position', [9 39 1280 720], ...
                    'AxisLength', 0.1, 'ShowArrowHead', false, ...
                    'Xlabel', 'X (m)', 'Ylabel', 'Y (m)', 'Zlabel', 'Z (m)', 'ShowLegend', false, 'Title', 'Unfiltered',...
                    'CreateAVI', false, 'AVIfileNameEnum', false, 'AVIfps', ((1/samplePeriod) / SamplePlotFreq));            
end
%% End of script
end