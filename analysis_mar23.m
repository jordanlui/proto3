% Analysis March 23
% Isolated motions. IMU and ART Tracker
% Test angle lookup method against known arm poses and predicted locations

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
RWP = [0 0 1; 0 -1 0; 1 0 0];
R7P = [1 0 0; 0 0 -1; 0 1 0];  % Determined graphically, but seems inconsistent
% R7P = R7W * RWP;
load('pointClouds/cloud_Lui_K20.mat');
load('LuiArm2018.mat')

% dataRaw = csvread(files(1).name); % Abd90 static
% dataRaw = csvread(files(7).name); % Abd90 static
% dataRaw = csvread(files(12).name); % Arm down at hip
% dataRaw = csvread(files(19).name); % ShFl90
% dataRaw = csvread(files(25).name); % Abd90 static
% dataRaw = csvread(files(26).name); % Abd 0-90 x2

% General Loader
% data = jointLoader(dataRaw,maxLengthArm,jointLabel,0); % Parse all data to a struct

% data = jointLoader(csvread(files(15).name),maxLengthArm,jointLabel,0); % ElFl0-90
% ind = 70; span = 350;

% data = jointLoader(csvread(files(17).name),maxLengthArm,jointLabel,40); % ElFl0-90
% ind = 90; span = 300;

data = jointLoader(csvread(files(26).name),maxLengthArm,jointLabel,20,1); % ShAb 0-90 x2
ind = 205; span = 270;
% 
% data = jointLoader(csvread(files(30).name),maxLengthArm,jointLabel,40); % ShAb 0-90 x2
% ind = 166; span = 500;

% data = jointLoader(csvread(files(31).name),maxLengthArm,jointLabel,40); % ShAb 0-90 x2
% ind = 210; span = 600;

% data = jointLoader(csvread(files(32).name),maxLengthArm,jointLabel,40); % ShAb 0-90 x2
% ind = 150; span = 700;


data.quatClean = cleanQuaternionTol(data.quat,1000);
data.eul = quat2eul(data.quatClean);
data.rot = quat2rotm(data.quatClean);

close all
figure(1)
plotallJoints3D(data.joints)

figure(2)
plot(data.wrist.pos); 
ylabel('Distance (mm)'); xlabel('time')
yyaxis right
plot(data.gyr,':'); ylabel('Gyro, deg/s')
legend('wx','wy','wz','gyrX','gyrY','gyrZ'); title('coordinates vs. time')

k = 50;
figure(3)
plotArmPose(data,k)
hold on, plotQuiver(data.joints{1}.pos(k,:)+0,data.rot(:,:,k) * inv(R7P)), hold off
% figure(4), plotArmPose(data,k)

% Figuring out wrist proto orientation relative to Wrist ART, Mar 26, 2018
RGW = reshape(data.joints{1}.rot(k,:), [3 3]) % Rotation from global to wrist
RUP = data.rot(:,:,k) % The wrist IMU orientation wrt to unknown frame

%% Pose Analysis with comparison to theoretical

% New params Mar 25
R7W = [1 0 0 ; 0 0 1; 0 -1 0]; % Version March 25, based on actual test results
RW7 = inv(R7W);
% Transform from shoulder {0} to Global frame
% Assumption that this is static - but we are over-simplifying
RG0 = [0 0 -1; 0 1 0; 1 0 0];

% Neutral pose at begining
k = 205;
[predict] = armPose(dhparams,[0,0,0,0,0,0,0]');

figure(3)
[result] =  poseAnalysis(data,k,predict,RG0,RW7); % Compare to predicted
rotm2eul(result.rotPredict) * 180/pi
rotm2eul(result.rotReal) * 180/pi
result.pReal
R07 = result.rotReal;

% Make transform matrix from shoulder {0} to global
TG0 = rotm2tform(RG0);
TG0(1:3,4) = data.joints{6}.pos(k,:)';

% Search for a match


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
    TGW = (rotm2tform(reshape(data.joints{1}.rot(k,:),[3,3])));
    useWristPos = data.joints{1}.pos(k,:)'; % Believe this is needed. Not sure.
    % Combine to get the T07 Transform
    T07 = inv(TG0) * TGW * (rotm2tform(RW7));
    R07 = tform2rotm(T07);
    [guess] = lookupPose(reshape(R07,[1,9]),rotation,wristPos,elbowPos,0);

    % Transform to {G} frame
    guess.wristG = TG0 * [guess.wristPos; 1];
    guess.elbowG = TG0 * [guess.elbowPos; 1];
    guessed.wrist(i,:) = guess.wristG(1:3);
    guessed.elbow(i,:)= guess.elbowG(1:3);


%     % Error comparison to actual data
%     error.wrist(i) = sqrt(sum((guessed.wrist(i,:) - data.wrist.pos(k,:)).^2));
%     error.elbow(i) = sqrt(sum((guessed.elbow(i,:) - data.elbow.pos(k,:)).^2));
end
error.wrist = sqrt(sum((guessed.wrist - real.wrist).^2,2));
error.elbow = sqrt(sum((guessed.elbow - real.elbow).^2,2));

%% Dig deep on a single time point, Apr 4

k = 215;
% Make transform matrices
TG0 = rotm2tform(RG0);
TG0(1:3,4) = data.joints{6}.pos(k,:)';
TGW = (rotm2tform(reshape(data.joints{1}.rot(k,:),[3,3]))); % Why is this inverted?
TGW(1:3,4) = data.joints{1}.pos(k,:)'; % Believe this is needed. Not sure.
% Combine to get the T07 Transform
T07 = inv(TG0) * TGW * (rotm2tform(RW7));
R07 = tform2rotm(T07);
[guess] = lookupPose(reshape(R07,[1,9]),rotation,wristPos,elbowPos,0);

guess.wristPos
guessG.wristPos = TG0 * [guess.wristPos ; 1];
guessG.wristPos = guessG.wristPos(1:3);
guessG.elbowPos = TG0 * [guess.elbowPos ; 1];
guessG.elbowPos = guessG.elbowPos(1:3);

error.wrist = sqrt(sum((guessG.wristPos - data.wrist.pos(k,:)').^2))
error.elbow = sqrt(sum((guessG.elbowPos- data.elbow.pos(k,:)').^2))
%% Results comparison
% error.wristOrig = error.wrist;
% error.wrist = error.wrist - error.elbow;

mean(error.wrist)
mean(error.elbow)

% error.y = guessed.wrist(:,2) - real.wrist(:,2);
% figure(),hist(abs(error.y))


figure(6),plotArmPrediction(guessed,real,error)
figure(7),hold on, hist([error.wrist, error.elbow]), legend('wrist','elbow'), xlabel('mm error')