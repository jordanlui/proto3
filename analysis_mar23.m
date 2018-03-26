% Analysis March 23
% Isolated motions. IMU and ART Tracker

clc
clear all
close all

set(0, 'DefaultLineLineWidth', 2);
tolDistance = 1000; 
maxLengthArm = 800; % Max length in mm
freq = 60; % Data rate in Hz
T = 1/freq;

jointLabel = {'wrist','forearm','arm','chest','shoulder1','shoulder2','elbow'};
accScale = 8192; % Conversion parameter for accelerometer to g value
gyrScale = 16.4; % Conversion parameter for gyroscope to deg/s
freq = 60;
load('BITcalibration20170125.mat')
path1 = '../Analysis/mar23/';
addpath(path1,'Functions');
files = dir(strcat(path1,'*.csv'));
filtWindow = 20;


%% File load and configure
R7W = [1 0 0 ; 0 0 1; 0 -1 0]; % Version March 25, based on actual test results
RW7 = inv(R7W);
% Transform from shoulder {0} to Global frame
RG0 = [0 0 -1; 0 1 0; 1 0 0];
load('pointClouds/pointCloudKlopcar_step10.mat');

load('LuiArm2018.mat')
[predict] = armPose(dhparams,[0,0,90,0,0,0,0]');

% dataRaw = csvread(files(7).name); % Abd90 static
% dataRaw = csvread(files(12).name); % Arm down at hip
% dataRaw = csvread(files(19).name); % ShFl90
% dataRaw = csvread(files(25).name); % Abd90 static
dataRaw = csvread(files(26).name); % Abd 0-90 x2

ind = 215;
span = 275;

% data = jointLoader(dataRaw,maxLengthArm,jointLabel,16); % Parse all data to a struct
% data.quatClean = cleanQuaternionTol(data.quat,1000);

data = jointLoader(csvread(files(26).name),maxLengthArm,jointLabel,20); % ShAb 0-90 x2
data.quatClean = cleanQuaternionTol(data.quat,1000);

figure(1)
plotallJoints3D(data.joints)

figure(2)
hold on; 
plot(data.wrist.pos); 
ylabel('Distance (mm)'); xlabel('time')
yyaxis right
plot(data.gyr,':'); ylabel('Gyro, deg/s')
hold off
legend('wx','wy','wz','gyrX','gyrY','gyrZ'); title('coordinates vs. time')


%% Pose Analysis

k = 200;
% New params Mar 25
R7W = [1 0 0 ; 0 0 1; 0 -1 0]; % Version March 25, based on actual test results
RW7 = inv(R7W);
% Transform from shoulder {0} to Global frame
RG0 = [0 0 -1; 0 1 0; 1 0 0];

figure(3)
[result] =  poseAnalysis(data,k,predict,RG0,RW7);
rotm2eul(result.rotPredict) * 180/pi
rotm2eul(result.rotReal) * 180/pi
result.pReal
R07 = result.rotReal;

TG0 = rotm2tform(RG0);
TG0(1:3,4) = data.joints{6}.pos(k,:)';

%% Search for a match

load('pointClouds/pointCloudKlopcar_step20.mat');
figure(5)
subplot(1,2,1),plotArmPose(data,k), title('Real Arm position, orientation in global frame');

% Predict position in frame {0}
[guess] = lookupPose(reshape(R07,[1,9]),rotation,wristPos,elbowPos,0);

guess.wristPos
guess.elbowPos

% Transform to {G} frame
guess.wristG = TG0 * [guess.wristPos; 1];
guess.elbowG = TG0 * [guess.elbowPos; 1];
guess.wristG = guess.wristG(1:3);
guess.elbowG = guess.elbowG(1:3);

% Error comparison to actual data
wristEstimationError = sqrt(sum((guess.wristG - data.wrist.pos(k,:)').^2))
elbowEstimationError = sqrt(sum((guess.elbowG - data.elbow.pos(k,:)').^2))

% Error calculation compared to theoretical. WHY??
% wristEstimationError = sqrt(sum((guess.wristPos - predict.wrist.pos).^2))
% elbowEstimationError = sqrt(sum((guess.elbowPos - predict.elbow.pos).^2))

subplot(1,2,2), plotQuiver(guess.wristG,guess.rotation), plotQuiver(guess.elbowG), plotQuiver(data.shoulder2.pos(k,:)') , plotQuiver([0,0,0]), xlabel('x'),ylabel('y'), zlabel('z'), title('Result from lookup')

%% Search on whole series

real.wrist = data.wrist.pos(ind:ind+span,:);
real.elbow = data.elbow.pos(ind:ind+span,:);
guessed.wrist = zeros(size(real.wrist));
guessed.elbow = zeros(size(real.elbow));
error.wrist = zeros(length(real.wrist),1);
error.elbow = zeros(length(real.elbow),1);

for i = 1:(span)
    k = ind -1 + i; % Index for the real array
    TG0 = rotm2tform(RG0);
    TG0(1:3,4) = data.joints{6}.pos(k,:)';
    TGW = inv(rotm2tform(reshape(data.joints{1}.rot(k,:),[3,3])));
    % Combine to get the T07 Transform
    T07 = inv(TG0) * TGW * (rotm2tform(RW7));
    R07 = tform2rotm(T07);
    [guess] = lookupPose(reshape(R07,[1,9]),rotation,wristPos,elbowPos,0);

    % Transform to {G} frame
    guess.wristG = TG0 * [guess.wristPos; 1];
    guess.elbowG = TG0 * [guess.elbowPos; 1];
    guessed.wrist(i,:) = guess.wristG(1:3);
    guessed.elbow(i,:)= guess.elbowG(1:3);


    % Error comparison to actual data
    error.wrist(i) = sqrt(sum((guessed.wrist(i,:) - data.wrist.pos(k,:)).^2));
    error.elbow(i) = sqrt(sum((guessed.elbow(i,:) - data.elbow.pos(k,:)).^2));
end

%% Results comparison


figure(6),plotArmPrediction(guessed,real,error)


% subplot(2,2,1)
% hold on
% plot(guessed.wrist(:,1),'r:')
% plot(guessed.wrist(:,2),'m:')
% plot(guessed.wrist(:,3),'b:')
% plot(real.wrist(:,1),'r')
% plot(real.wrist(:,2),'m')
% plot(real.wrist(:,3),'b')
% hold off
% legend('x','y','z','x','y','z')
% title('Wrist real and predicted')
% 
% subplot(2,2,2)
% plot(guessed.elbow(:,1),'r:'), hold on
% plot(guessed.elbow(:,2),'m:')
% plot(guessed.elbow(:,3),'b:')
% plot(real.elbow(:,1),'r')
% plot(real.elbow(:,2),'m')
% plot(real.elbow(:,3),'b')
% hold off
% legend('x','y','z','x','y','z')
% title('Elbow real and predicted')
% 
% subplot(2,2,3)
% plot(error.wrist,'r.','LineStyle','none'), ylabel('Error (mm)')
% yyaxis right
% % hold on
% % scatter(1:length(guessed.wrist),guessWristdist,3,'m')
% % scatter(1:length(real.wrist),realWristDist,3,'b')
% % ylabel('Distance (mm)')
% % hold off
% title('Wrist Error')
% legend('error','Prediction','real')
% 
% subplot(2,2,4)
% plot(error.elbow,'r.','LineStyle','none'), ylabel('Error (mm)')
% yyaxis right
% % hold on
% % scatter(1:length(guessed.elbow),guessElbowdist,3,'m')
% % scatter(1:length(real.elbow),realElbowDist,3,'b')
% % hold off
% title('Elbow Error')
% legend('error','Prediction','real')