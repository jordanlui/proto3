% Results analysis jan 19 exercises

clc
clear all
close all
files = dir('../analysis/jan19/*.csv');

% Set some run parameters
plotSkipping = 30;
freq = 60; % Frequency for device and ART data from LabVIEW. Should be 60 Hz, but might not be perfect.

% Runtime parameters
accScale = 8192; % Conversion parameter for accelerometer
gyrScale = 16.4; % Conversion parameter for gyroscope
load('BITcalibration20170125.mat')
%% Load data and clip to proper start and end
afile = files(1).name;
data = csvread(strcat(files(1).folder,'/',afile));
data = data(2243:7125,:); % Clip data
% Parse data and decide to reject some error tracked points
checkZero = data(:,28:36) == 0; % Detect zero values
checkZero = any(checkZero,2); % Compress into a vector. 1 indicates a zero is present
data = data(~checkZero,:);
[time, acc, gyr, quat, omron, arm, forearm, wrist] = parseData(data);
plotJoints(arm,forearm,wrist,afile)
distanceWristArm = sqrt(sum((wrist-arm).^2,2)); % Distance values
X = [omron acc gyr quat];
y1 = distanceWristArm;
data1 = [y1 X];
accIn = acc./accScale;
gyrIn = gyr - gyrCal;
gyrIn = gyrIn./gyrScale;
pos1 = deadReckonMadgwickOscillationFunc(accIn,gyrIn,freq,0.1,0);


afile = files(2).name;
data = csvread(strcat(files(1).folder,'/',afile));
data = data(1149:6344,:);
checkZero = data(:,28:36) == 0; % Detect zero values
checkZero = any(checkZero,2); % Compress into a vector. 1 indicates a zero is present
data = data(~checkZero,:);
[time, acc, gyr, quat, omron, arm, forearm, wrist] = parseData(data);
plotJoints(arm,forearm,wrist,afile)
distanceWristArm = sqrt(sum((wrist-arm).^2,2)); % Distance values
X = [omron acc gyr quat];
y2 = distanceWristArm;
data2 = [y2 X];
accIn = acc./accScale;
gyrIn = gyr - gyrCal;
gyrIn = gyrIn./gyrScale;
pos2 = deadReckonMadgwickOscillationFunc(accIn,gyrIn,freq,0.1,0);

% Trial 3 inspection
afile = files(3).name;
data = csvread(strcat(files(1).folder,'/',afile));
data = data(1121:6179,:);
checkZero = data(:,28:36) == 0; % Detect zero values
checkZero = any(checkZero,2); % Compress into a vector. 1 indicates a zero is present
data = data(~checkZero,:);
[time, acc, gyr, quat, omron, arm, forearm, wrist] = parseData(data);

plotJoints(arm,forearm,wrist,afile)
distanceWristArm = sqrt(sum((wrist-arm).^2,2)); % Distance values
X = [omron acc gyr quat];
y3 = distanceWristArm;
data3 = [y3 X];
accIn = acc./accScale;
gyrIn = gyr - gyrCal;
gyrIn = gyrIn./gyrScale;
pos3 = deadReckonMadgwickOscillationFunc(accIn,gyrIn,freq,0.1,0);

% Combine our datasets
datas = {data1 data2 data3};
positions = {pos1 pos2 pos3};
M = length(datas);


% Check if data sets overlap sufficiently
figure()
hold on
plot(y1)
plot(y2)
plot(y3)
hold off
title('Compare wrist-arm distances in multiple trials')
xlabel('time')
ylabel('distance (mm)')
legend('1','2','3')


%% Dead Reckon with Madgwick Algorithm
% Data Prep (Always pass clean data to algorithm!)
% gyrCalStationary = [-2.172099087	1.585397653	2.456323338];
load('BITcalibration20170125.mat')

accIn = acc./accScale;
gyrIn = gyr - gyrCal;
gyrIn = gyrIn./gyrScale;
pos3 = deadReckonMadgwickOscillationFunc(accIn,gyrIn,freq,0.1,0);

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
%% Build Model - Regression Tree
% After creating model in Regression Learner
% Train and test on a single recording to see results

% [trainedModel, validationRMSE] = trainRegressionModel(data1);
% x_train = data1(:,2:size(data1,2));
% ypred = trainedModel.predictFcn(x_train);
% y_test = data1(:,1);
% trainRMSE = sqrt( sum((ypred - y_test).^2) ./ length(ypred));
% sprintf('Train RMSE error is %.2f',trainRMSE)
% 
% figure()
% hold on
% plot(ypred)
% plot(data1(:,1))
% legend('predict','real')
% ylabel('Normalized Distance')
% xlabel('Time (s)')
% hold off
% title('Validation of model with training data')

%% Check against other dataset
% test = matNorm(data1);
y_test = data2(:,1);
% x_test = test(:,2:end);

ypred = trainedModel.predictFcn(data2(:,2:size(data2,2)));

figure()
hold on
plot(ypred)
plot(y_test)
legend('predict','real')
hold off
title('Check against test data (unseen)')
testRMSE = sqrt( sum((ypred - y_test).^2) ./ length(ypred));
testSpaceDistance = max(y_test) - min(y_test);
testErrorRelative = testRMSE / testSpaceDistance;
sprintf('Test RMSE error is %.2f',testRMSE)

%% Try with normalized data
setTrain = [data3 ;data2];
setTest = data1;

y_testmm = setTest(:,1); % Original value in mm
normStats = {mean(data1) max(data1) min(data1)};
setTrain = ((setTrain - normStats{1}) ./ (normStats{2} - normStats{3}));
setTest = ((setTest - normStats{1}) ./ (normStats{2} - normStats{3}));
x_train = setTrain(:,2:end);
y_train = setTrain(:,1);
x_test = setTest(:,2:end);
y_test = setTest(:,1);

% Regression Tree
[trainedModel, validationRMSE] = trainRegressionModel(setTrain);
y_pred = trainedModel.predictFcn(x_test);
y_predmm = y_pred .* (normStats{2}(1) - normStats{3}(1)) + normStats{1}(1); % Transform back into mm

testRMSE = sqrt( sum((y_pred - y_test).^2) ./ length(y_test)); % Normalized Error
testSpaceDistance = max(y_testmm) - min(y_testmm);
testErrorRelative = testRMSE / testSpaceDistance;
sprintf('Regression Tree. Training RMSE error is %.4f',validationRMSE)
sprintf('Regression Tree. Normalied Test RMSE error is %.4f',testRMSE)

errormm = y_predmm - y_testmm;
testRMSEmm = sqrt( sum((y_predmm - y_testmm).^2) ./ length(y_test)); % mm error
sprintf('Regression Tree. Test RMSE error is %.4f mm',testRMSEmm)


%% Support Vector Machine
Mdl = fitrsvm(x_train,y_train);
Mdl.ConvergenceInfo.Converged % Check model convergence
y_predSVR = Mdl.predict(x_test);

y_predSVRmm = y_predSVR .* (normStats{2}(1) - normStats{3}(1)) + normStats{1}(1);
errorSVR = y_predSVRmm - y_testmm;
testRMSEmm = sqrt( sum((errorSVR).^2) ./ length(y_test)); % mm error
sprintf('SVR. Test RMSE error is %.4f mm. Mean %.2f',testRMSEmm,mean(abs(errorSVR)))

figure()
hold on
plot(y_predSVRmm)
plot(y_testmm)
hold off
legend('Predict','Real')
title('SVR')
%% Loop Through data and apply fine tree regression learning
% for i=1:M
%     % Construct our data sets
%     testData = datas{i};
%     ytestmm = testData(:,1); % Test data in mm
%     trainData = [];
%     for j = 1:M
%         if j ~= i
%             trainData = [trainData; datas{j}];
%         end
%     end
%     % Train model
%     normStats = {mean(trainData) max(trainData) min(trainData)};
%     trainData = (trainData - mean(trainData)) ./ (max(trainData) - min(trainData));
%     testData = (testData - mean(trainData)) ./ (max(trainData) - min(trainData));
%     [trainedModel, validationRMSE] = trainRegressionModel(data1);
%     
%     
%     % Test Model
%     ypred = trainedModel.predictFcn(testData(:,2:end));
%     % Convert prediction back into mm values
%     ypredmm = ypred * (max(trainData) - min(trainData)) + mean(trainData);
%     
%     % Test error calculation
%     testError = ypredmm - ytestmm;
%     
%     figure()
%     hold on
%     plot(ypredmm)
%     plot(ytestmm)
%     legend('prediction','real')
%     title('Result plot')
%     hold off
%     
%     
%     validationRMSE
%     % MSE calculation
%     mean(sum(testError.^2))
%     
% end

%% End of Script
%% Functions
function distance = dist3D(A,B)
    % Distance between two matrices. Checks for zero values
    

end

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

function plotArmPosition(wrist,forearm,arm)
    figure()
    hold on
    plot(wrist(:,1))
    plot(wrist(:,2))
    plot(wrist(:,3))
    plot(forearm(:,1))
    plot(forearm(:,2))
    plot(forearm(:,3))
    plot(arm(:,1))
    plot(arm(:,2))
    plot(arm(:,3))
    plot(distanceWristArm,'--or')
    hold off
    title('Compare tracker positions')
    xlabel('time')
    ylabel('Position (mm)')
    legend('w1','w2','w3','f1','f2','f3','a1','a2','a3','wrist dist')
end