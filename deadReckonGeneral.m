% General Dead Reckon analysis and plotting
% Adapted from Madgwick gait analysis script
% More general: Contains HPF and LPF to filter both sides
% https://github.com/xioTechnologies/Gait-Tracking-With-x-IMU/tree/master/Gait%20Tracking%20With%20x-IMU

function [pos, displacement, checkReturnCentre] = deadReckonGeneral(dataPath,mcuFreq,filtLPF,filtHPF,stationaryThreshold)

addpath('Libraries/ximu_matlab_library');	% include x-IMU MATLAB library
addpath('Libraries/quaternion_library');
addpath('Libraries/MahonyAHRS');
addpath('Libraries');


% -------------------------------------------------------------------------
% Select dataset (comment in/out)

% filePath = 'Datasets/straightLine';
% startTime = 6;
% stopTime = 26;

% dataPath = '../data/sept29/processed_walk around lab.csv';
% dataPath = 'data/20170924/walk_forwardback.csv';
% dataPath = '../../../Projects/Dead Reckoning IMU/Libraries/Gait-Tracking-With-x-IMU-master/Gait Tracking With x-IMU/Datasets/straightLine_CalInertialAndMag.csv';


dataTemp = csvread(dataPath,1,0); % Skip the header
time = dataTemp(:,1);
packets = dataTemp(:,2);
acc = dataTemp(:,3:5); % Accelerometer data, g values
gyr = dataTemp(:,6:8); % Gyro data, degrees per second

% [accCal,gyrCal] = calibrateIMU(acc,gyr); % Note this still needs work
% gyrCal = [2.50245051837889,2.28717247879359,-4.21495994344957]; % Calibration data from Oct 2 data of device sitting on desk
% Calibration values from October 5. New code to ensure packet loss is
% essentially eliminated
gyrCal = [7.58, -2.60, 9.71];
AccMax = [9.97, 11.4, 12.23];
AccMin = [-11.47, -10.79, -10.24];

% Calibration related compensation
% Mean shift gyro values
for i = 1:3
    gyr(:,i) = gyr(:,i) - gyrCal(i);
end

% Normalize accelerometer values
% This will accept accelerometer values in m/s2 and normalize in g values
for i = 1:3
    col = acc(:,i);
    col = 2 * (col - AccMin(i)) / (AccMax(i) - AccMin(i)) - 1;
    acc(:,i) = col;
end


accX = acc(:,1);
accY = acc(:,2);
accZ = acc(:,3);
gyrX = gyr(:,1);
gyrY = gyr(:,2);
gyrZ = gyr(:,3);


% mcuFreq = 16; % MCU Recording frequency, in Hz
samplePeriod = 1 / (mcuFreq); % Period is 1/frequency
% cutoffFreq = (filtCutOff)/(1/samplePeriod);

% Calibrate data
% Recalc gyro


% -------------------------------------------------------------------------
% Manually frame data

% startTime = 0;
% stopTime = 10;

% indexSel = find(sign(time-startTime)+1, 1) : find(sign(time-stopTime)+1, 1);
% time = time(indexSel);
% gyrX = gyrX(indexSel, :);
% gyrY = gyrY(indexSel, :);
% gyrZ = gyrZ(indexSel, :);
% accX = accX(indexSel, :);
% accY = accY(indexSel, :);
% accZ = accZ(indexSel, :);

% -------------------------------------------------------------------------
% Detect stationary periods

% Compute accelerometer magnitude
acc_mag = sqrt(accX.*accX + accY.*accY + accZ.*accZ);

% Default filter values are 0.001 and 5, with threshold 0.05

% HP filter accelerometer data
% filtLPF = 0.001; % default value
filtHPF = (2*filtHPF)/(1/samplePeriod);
% filtHPF = 7.8e-6;
[b, a] = butter(1, filtHPF, 'high');
acc_magFilt = filtfilt(b, a, acc_mag);

% Compute absolute value
acc_magFilt = abs(acc_magFilt);

% LP filter accelerometer data
% filtLPF = 7.9; % default value
% filtLPF = 0.99;
filtLPF = (2*filtLPF)/(1/samplePeriod);
[b, a] = butter(1, filtLPF, 'low');
acc_magFilt = filtfilt(b, a, acc_magFilt);

% Threshold detection
% stationaryThreshold = 0.05;
stationary = acc_magFilt < stationaryThreshold;

% -------------------------------------------------------------------------
% Plot data raw sensor data and stationary periods

figure('Position', [9 39 900 600], 'NumberTitle', 'off', 'Name', 'Sensor Data');
ax(1) = subplot(2,1,1);
    hold on;
    plot(time, gyrX, 'r');
    plot(time, gyrY, 'g');
    plot(time, gyrZ, 'b');
    title('Gyroscope');
    xlabel('Time (s)');
    ylabel('Angular velocity (^\circ/s)');
    legend('X', 'Y', 'Z');
    hold off;
ax(2) = subplot(2,1,2);
    hold on;
    plot(time, accX, 'r');
    plot(time, accY, 'g');
    plot(time, accZ, 'b');
    plot(time, acc_magFilt, ':k');
    plot(time, stationary, 'k', 'LineWidth', 2);
    title('Accelerometer');
    xlabel('Time (s)');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z', 'Filtered', 'Stationary');
    hold off;
linkaxes(ax,'x');

% -------------------------------------------------------------------------
% Compute orientation

quat = zeros(length(time), 4);
AHRSalgorithm = AHRS('SamplePeriod', samplePeriod, 'Kp', 1, 'KpInit', 1);

% Initial convergence
initPeriod = 2;
indexSel = 1 : find(sign(time-(time(1)+initPeriod))+1, 1);
for i = 1:2000
    AHRSalgorithm.UpdateIMU([0 0 0], [mean(accX(indexSel)) mean(accY(indexSel)) mean(accZ(indexSel))]);
end

% For all data
for t = 1:length(time)
    if(stationary(t))
        AHRSalgorithm.Kp = 0.5;
    else
        AHRSalgorithm.Kp = 0;
    end
    AHRSalgorithm.UpdateIMU(deg2rad([gyrX(t) gyrY(t) gyrZ(t)]), [accX(t) accY(t) accZ(t)]);
    quat(t,:) = AHRSalgorithm.Quaternion;
end

% -------------------------------------------------------------------------
% Compute translational accelerations

% Rotate body accelerations to Earth frame
acc = quaternRotate([accX accY accZ], quaternConj(quat));

% % Remove gravity from measurements
% acc = acc - [zeros(length(time), 2) ones(length(time), 1)];     % unnecessary due to velocity integral drift compensation

% Convert acceleration measurements to m/s/s
acc = acc * 9.81;

% Plot translational accelerations
figure('Position', [9 39 900 300], 'NumberTitle', 'off', 'Name', 'Accelerations');
hold on;
plot(time, acc(:,1), 'r');
plot(time, acc(:,2), 'g');
plot(time, acc(:,3), 'b');
title('Acceleration');
xlabel('Time (s)');
ylabel('Acceleration (m/s/s)');
legend('X', 'Y', 'Z');
hold off;

% -------------------------------------------------------------------------
% Compute translational velocities

acc(:,3) = acc(:,3) - 9.81; 

% Integrate acceleration to yield velocity
vel = zeros(size(acc));
for t = 2:length(vel)
    vel(t,:) = vel(t-1,:) + acc(t,:) * samplePeriod;
    if(stationary(t) == 1)
        vel(t,:) = [0 0 0];     % force zero velocity when foot stationary
    end
end


% Compute integral drift during non-stationary periods
% Note that errors in code are happening here due to inconsistent numbers
% of of elements
velDrift = zeros(size(vel));
stationaryStart = find([0; diff(stationary)] == -1);
stationaryEnd = find([0; diff(stationary)] == 1);


% Introduce error handling for the mismatch
if length(stationaryEnd) == length(stationaryStart)
    % Do nothing, we can proceed
elseif length(stationaryEnd) > length(stationaryStart) % If End is longer
    if stationaryEnd(1) < stationaryStart(1)
        % If first start value is smaller, we have an issue. we have to
        % resize stationaryEnd
        stationaryEnd = stationaryEnd(2:end);
    else
        stationaryEnd = stationaryEnd(1:end-1);
    end
    
elseif length(stationaryEnd) < length(stationaryStart)
end

for i = 1:numel(stationaryEnd)
    driftRate = vel(stationaryEnd(i)-1, :) / (stationaryEnd(i) - stationaryStart(i));
    enum = 1:(stationaryEnd(i) - stationaryStart(i));
    drift = [enum'*driftRate(1) enum'*driftRate(2) enum'*driftRate(3)];
    velDrift(stationaryStart(i):stationaryEnd(i)-1, :) = drift;
end

% Remove integral drift
vel = vel - velDrift;

% Plot translational velocity
figure('Position', [9 39 900 300], 'NumberTitle', 'off', 'Name', 'Velocity');
hold on;
plot(time, vel(:,1), 'r');
plot(time, vel(:,2), 'g');
plot(time, vel(:,3), 'b');
title('Velocity');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('X', 'Y', 'Z');
hold off;

% -------------------------------------------------------------------------
% Compute translational position

% Integrate velocity to yield position
pos = zeros(size(vel));
for t = 2:length(pos)
    pos(t,:) = pos(t-1,:) + vel(t,:) * samplePeriod;    % integrate velocity to yield position
end

% Plot translational position
figure('Position', [9 39 900 600], 'NumberTitle', 'off', 'Name', 'Position');
hold on;
plot(time, pos(:,1), 'r');
plot(time, pos(:,2), 'g');
plot(time, pos(:,3), 'b');
title('Position');
xlabel('Time (s)');
ylabel('Position (m)');
legend('X', 'Y', 'Z');
hold off;

% -------------------------------------------------------------------------
%% Custom code from Jordan

displacement = sqrt(sum( (max(pos) - min(pos)).^2 )); % Check displacement from recording. Note this can be a naieve calculation
checkReturnCentre = sqrt(sum( (pos(end,:) - pos(1,:)).^2 ));


%% Plot 3D foot trajectory

% % Remove stationary periods from data to plot
% posPlot = pos(find(~stationary), :);
% quatPlot = quat(find(~stationary), :);
posPlot = pos;
quatPlot = quat;


% Extend final sample to delay end of animation
extraTime = 1;
onesVector = ones(extraTime*(1/samplePeriod), 1);
posPlot = [posPlot; [posPlot(end, 1)*onesVector, posPlot(end, 2)*onesVector, posPlot(end, 3)*onesVector]];
quatPlot = [quatPlot; [quatPlot(end, 1)*onesVector, quatPlot(end, 2)*onesVector, quatPlot(end, 3)*onesVector, quatPlot(end, 4)*onesVector]];

% Create 6 DOF animation
SamplePlotFreq = 15;
Spin = 120;
SixDOFanimation(posPlot, quatern2rotMat(quatPlot), ...
                'SamplePlotFreq', SamplePlotFreq, 'Trail', 'All', ...
                'Position', [9 39 1280 768], 'View', [(100:(Spin/(length(posPlot)-1)):(100+Spin))', 10*ones(length(posPlot), 1)], ...
                'AxisLength', 0.1, 'ShowArrowHead', false, ...
                'Xlabel', 'X (m)', 'Ylabel', 'Y (m)', 'Zlabel', 'Z (m)', 'ShowLegend', false, ...
                'CreateAVI', false, 'AVIfileNameEnum', false, 'AVIfps', ((1/samplePeriod) / SamplePlotFreq));
end