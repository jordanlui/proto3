% Point cloud generator script. Calls the calcPointCloud() function
clc
clear all
close all
addpath('Functions');



%% Lui built system
% Arm lengths in mm
% load('measurementsfeb23.mat')
% % % 
% Ll = floor(Ll);
% Lu = floor(Lu);
% dhparams = [0 0 0 90;
%     90 0 0 90;
%     90 0 0 -90;
%     -90 Lu 0 0;
%     90 Ll 0 0;
%     90 0 0 -90;
%     90 0 0 90];
% save('LuiArm2018.mat','dhparams','Lu','Ll')

load('LuiArm2018.mat')

dh.orig = dhparams;


%% Point cloud generation
angleRange = [-60,180;-40,120;-30,120;0,150;0,180]; % Angle range form Shen 2016
% angleRange = [-10,62;-14,134;-72,55;41,163;-125,135]; % ROM in Rosen 2005 ADL
stepSize = 10;
printoutStep = 2e4;
[rotation,elbowPos,wristPos,angles] = calcPointCloud(angleRange,stepSize);

% Store the lookup matrix
elbowMat = cell2mat(elbowPos);
wristMat = cell2mat(wristPos);

save('pointCloudShen_step10.mat','rotation','elbowMat','wristMat','angles')

%% Plot the point cloud
% clc
% clear all
% load('pointCloudShen.mat')
% close all

figure()
title('Potential elbow and wrist locations')
hold on
stepPlot = 2;
wristMat = wristMat(1:stepPlot:end,:);
elbowMat = elbowMat(1:stepPlot:end,:);
sizePt = 1;
scatter3(wristMat(:,1),wristMat(:,2),wristMat(:,3),sizePt,'r')
scatter3(elbowMat(:,1),elbowMat(:,2),elbowMat(:,3),sizePt*4,'b')
scatter3(0,0,0,100,'k')
xlabel('x')
ylabel('y')
zlabel('z')
hold off

% Number checking
elbowDistances = sqrt(sum(elbowMat.^2,2));
wristDistances = sqrt(sum((wristMat - elbowMat).^2,2));


