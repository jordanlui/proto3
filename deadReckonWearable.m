% General Dead Reckon analysis and plotting
% Adapted from Madgwick gait analysis script
% More general: Contains HPF and LPF to filter both sides
% https://github.com/xioTechnologies/Gait-Tracking-With-x-IMU/tree/master/Gait%20Tracking%20With%20x-IMU

function [pos, displacement, maxDisplacement3Axis,accgyr_orig,acc,gyr,vel,mcuFreq] = deadReckonWearable(dataPath,filtLPF,filtHPF,stationaryThreshold, calibrationFile)

addpath('Libraries/ximu_matlab_library');	% include x-IMU MATLAB library
addpath('Libraries/quaternion_library');
addpath('Libraries/MahonyAHRS');
addpath('Libraries');
% -------------------------------------------------------------------------
% Select dataset (comment in/out)
% filePath = 'Datasets/straightLine';
% startTime = 6;
% stopTime = 26;

if nargin < 5
    calibrationFile = 'calibration_flora_oct5.mat';
end

[filepath,name,ext] = fileparts(dataPath);
plotInfo = sprintf(' for "%s", filt with %.2f, %.4f, %.4f',name,filtLPF,filtHPF,stationaryThreshold);

% File parsing for Oct 27
% dataTemp = csvread(dataPath,1,0); % Skip the header
% time = dataTemp(:,1);
% packets = dataTemp(:,2);
% acc = dataTemp(:,3:5); % Accelerometer data, m/s2 values
% gyr = dataTemp(:,6:8); % Gyro data, degrees per second

% File parsing for Oct 27

accgyr_orig = [acc gyr]; % Original acc and gyro data, before calibration

% File parsing for Nov 3 data
dataTemp = csvread(dataPath,0,0); % No Header to skip for nov3
time = dataTemp(:,33); % Time in ms
packets = dataTemp(:,32); % packets
acc = dataTemp(:,5:7);
gyr = dataTemp(:,8:10);
accgyr_orig = [acc gyr];
% File parsing for Nov 3 data

% Calculate microcontroller frequency from the timesteps in data
mcuFreq = (packets(end) - packets(1) + 1 ) / (time(end) - time(1)) * 1e3; % Calculate frequency from the data
mcuFreq = floor(mcuFreq); % Integer frequency value
timeSteps = time(2:end) - time(1:end-1);
timeStep = mode(timeSteps); % Most common time step represents ideal timing period
mcuFreq = 1000/timeStep; % Controller Frequency in Hz 
samplePeriod = (1 / (mcuFreq)); % Period is 1/frequency

% Calibrate acceleomter and gyro values based on calibration file
gForce_bound = 1.0; % Upper and lower bound value that we will normalize to
load(calibrationFile);

% Calibration related compensation
% Mean shift gyro values
for i = 1:3
    gyr(:,i) = gyr(:,i) - gyrCal(i);
end

% Normalize accelerometer values
% This will accept accelerometer values in m/s2 and normalize to +/- 9.81
for i = 1:3
    col = acc(:,i);
    col = 2 * gForce_bound * (col - AccMin(i)) / (AccMax(i) - AccMin(i)) - gForce_bound;
    acc(:,i) = col;
end

%% Before and after analysis of acc and gyro data

% FFT Plots of Accelerometer and Gyro
M = length(dataTemp);
acc_both = [dataTemp(:,3:5) acc];
gyr_both = [dataTemp(:,6:8) gyr];
figure(1)
for i = 1:6 % Accelerometer data
    subplot(2,3,i)
    Y = fft(acc_both(:,i));
    L = M;
    Fs = mcuFreq;
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = Fs*(0:(L/2))/L;
    plot(f,P1)
    plotTitle = strcat('FFT acc', plotInfo);
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([0 20])
end

figure(2)
for i = 1:6 % Gyro data
    subplot(2,3,i)
    Y = fft(gyr_both(:,i));
    L = M;
    Fs = mcuFreq;
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = Fs*(0:(L/2))/L;
    plot(f,P1)
    plotTitle = strcat('FFT gyro', plotInfo);
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([0 20])
end


accX = acc(:,1);
accY = acc(:,2);
accZ = acc(:,3);
gyrX = gyr(:,1);
gyrY = gyr(:,2);
gyrZ = gyr(:,3);


% -------------------------------------------------------------------------
%% Manually frame data

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
%% Detect stationary periods


%% Filter Accelerometer mag value

% Compute accelerometer magnitude
acc_mag = sqrt(accX.*accX + accY.*accY + accZ.*accZ);
figure()
plot(acc_mag)
hold on

% Default filter values are 0.001 and 5, with threshold 0.05

% Method 1: Two step BPF 
% HP filter accelerometer data
wnHPF = (2*filtHPF)/(1/samplePeriod);
[b, a] = butter(1, wnHPF, 'high');
acc_magFilt = filtfilt(b, a, acc_mag);
plot(acc_magFilt)

% Compute absolute value
acc_magFilt = abs(acc_magFilt);

% LP filter accelerometer data
wnLPF = (2*filtLPF)/(1/samplePeriod);
[b, a] = butter(1, wnLPF, 'low');
acc_magFilt = filtfilt(b, a, acc_magFilt);
plot(acc_magFilt)

stationary = acc_magFilt < stationaryThreshold;
plot(stationary)
hold off
legend({'acc_mag','acc_mag HPF','acc_mag BPF'},'Interpreter', 'none')
title('Accelerometer values: Raw and Filtered')

% Method 2: IIR Filter 
% filtIIR = designfilt('bandpassiir','FilterOrder',20, ...
%          'HalfPowerFrequency1',filtHPF,'HalfPowerFrequency2',filtLPF, ...
%          'SampleRate',mcuFreq);
% filtFIR = designfilt('bandpassfir','FilterOrder',20, ...
%          'CutoffFrequency1',filtHPF,'CutoffFrequency2',filtLPF, ...
%          'SampleRate',mcuFreq); 
% fvtool(filtFIR)
% acc_magFilt = filter(filtFIR, acc_mag);
% % acc_magFilt = abs(acc_magFilt);
% stationary = acc_magFilt < stationaryThreshold;
% figure()
% hold on
% plot(acc_mag)
% plot(acc_magFilt)
% plot(stationary)
% hold off
% title('FIR/IIR filtering')
% legend('Original','FIR or IIR filter','stationary detection')

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


figure('Position', [9 39 900 600], 'NumberTitle', 'off', 'Name', 'Gyro Calibration');
ax(1) = subplot(1,2,2);
    hold on;
    plot(time, gyrX, 'r');
    plot(time, gyrY, 'g');
    plot(time, gyrZ, 'b');
    title('Gyroscope Calibrated');
    xlabel('Time (s)');
    ylabel('Angular velocity (^\circ/s)');
    legend('X', 'Y', 'Z');
    hold off;
ax(2) = subplot(1,2,1);
    hold on;
    plot(time, accgyr_orig(:,4), 'r');
    plot(time, accgyr_orig(:,5), 'g');
    plot(time, accgyr_orig(:,6), 'b');
    title('Gyroscope');
    xlabel('Time (s)');
    ylabel('Angular velocity (^\circ/s)');
    legend('X', 'Y', 'Z');
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
titleString = strcat('Position', plotInfo);
title(titleString);
xlabel('Time (s)');
ylabel('Position (m)');
legend('X', 'Y', 'Z');
hold off;

% FFT Analysis of position data
L = length(acc);
Fs = mcuFreq;
f = Fs*(0:(L/2))/L;
plotTitles={'acc x','acc y','acc z','vel x',' vel y','vel z',' pos x',' pos y',' pos z'};
figure()
for i = 1:3
    % Accelerometer fft
    subplot(3,3,i)
    hold on
    Y = fft(acc(:,i));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    plot(f,P1)
    plotTitle = plotTitles{i};
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([-1 20])
    ylim([0 2.5])
    
    % Velocity fft
    subplot(3,3,i+3)
    hold on
    Y = fft(vel(:,i));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    plot(f,P1)
    plotTitle = plotTitles{i+3};
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([-1 20])
    ylim([0 0.25])
    
    % Position fft
    subplot(3,3,i+6)
    hold on
    Y = fft(pos(:,i));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    plot(f,P1)
    plotTitle = plotTitles{i+6};
    title(plotTitle)
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    xlim([-1 20])
    ylim([0 0.3])
end

% -------------------------------------------------------------------------
%% Custom code from Jordan

maxDisplacement3Axis = sqrt(sum( (max(pos) - min(pos)).^2 )); % Check total 3 axis displacement from recording. Can be a naieve calculation
displacement = sqrt(sum( (pos(end,:) - pos(1,:)).^2 ));


%% Plot 3D trajectory

% % Remove stationary periods from data to plot
% posPlot = pos(find(~stationary), :);
% quatPlot = quat(find(~stationary), :);
posPlot = pos;
quatPlot = quat;


% Extend final sample to delay end of animation
extraTime = 1;
onesVector = ones(floor(extraTime*(1/samplePeriod)), 1);
posPlot = [posPlot; [posPlot(end, 1)*onesVector, posPlot(end, 2)*onesVector, posPlot(end, 3)*onesVector]];
quatPlot = [quatPlot; [quatPlot(end, 1)*onesVector, quatPlot(end, 2)*onesVector, quatPlot(end, 3)*onesVector, quatPlot(end, 4)*onesVector]];

% Create 6 DOF animation
SamplePlotFreq = 25;
Spin = 120;
SixDOFanimation(posPlot, quatern2rotMat(quatPlot), ...
                'SamplePlotFreq', SamplePlotFreq, 'Trail', 'All', ...
                'Position', [9 39 1280 768], 'View', [(100:(Spin/(length(posPlot)-1)):(100+Spin))', 10*ones(length(posPlot), 1)], ...
                'AxisLength', 0.1, 'ShowArrowHead', false, ...
                'Xlabel', 'X (m)', 'Ylabel', 'Y (m)', 'Zlabel', 'Z (m)', 'ShowLegend', false, ...
                'CreateAVI', false, 'AVIfileNameEnum', false, 'AVIfps', ((1/samplePeriod) / SamplePlotFreq));
end