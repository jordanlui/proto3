% Point cloud generator script. Calls the calcPointCloud() function
clc
clear all
close all
addpath('Functions');



%% Lui built system
% Load arm length and DH Parameters
load('LuiArm2018.mat') 
% Update DH params 
Ll = 247;
Lu = 320;
dhparams(4,2) = Lu;
dhparams(5,2) = Ll;
save('LuiArm2018.mat') 
dh.orig = dhparams;

%% Point cloud generation
angleRangeShen = [-60,180;-40,120;-30,120;0,150;0,180]; % Angle range form Shen 2016. [ShFlx, ShAbd, ShRot, ElbFlx, ElbPro]
angleRangeRosen = [-10,62;-14,134;-72,55;41,163;-125,135]; % ROM in Rosen 2005 ADL [ShFlx, ShAbd, ShRot, ElbFlx, ElbPro]
angleRangeKlopcar = [-60,170; -10,170; -60,90 ; -90,60 ; 0,180]; % [ShFlx, ShAbd, ShRot, ElbFlx, ElbPro]
stepSize = 15;
printoutStep = 2e4;

[rotation,elbowPos,wristPos,angles] = calcPointCloud(angleRangeShen,stepSize);
% [rotation,elbowPos,wristPos,angles] = calcPointCloudKlopcar(angleRangeKlopcar,stepSize);
disp('Final length')
length(rotation)

save('pointClouds\cloud_Lui_S15.mat','rotation','elbowPos','wristPos','angles')
% save('pointClouds\cloud_Lui_K10.mat','rotation','elbowPos','wristPos','angles')


%% Plot the point cloud
% clc
% clear all
% load('pointCloudKlopcar_step20.mat')
% close all

figure()
title('Potential elbow and wrist locations')
hold on
stepPlot = 5;
% wristPos = wristPos(1:stepPlot:end,:);
% elbowPos = elbowPos(1:stepPlot:end,:);
sizePt = 1;
plot3(wristPos(1:stepPlot:end,1),wristPos(1:stepPlot:end,2),wristPos(1:stepPlot:end,3),'r.')
plot3(elbowPos(1:stepPlot:end,1),elbowPos(1:stepPlot:end,2),elbowPos(1:stepPlot:end,3),'b.')
% scatter3(wristPos(1:stepPlot:end,1),wristPos(1:stepPlot:end,2),wristPos(1:stepPlot:end,3),sizePt,'r')
% scatter3(elbowPos(1:stepPlot:end,1),elbowPos(1:stepPlot:end,2),elbowPos(1:stepPlot:end,3),sizePt*4,'b')
scatter3(0,0,0,100,'k')
xlabel('x')
ylabel('y')
zlabel('z')
hold off

%% Plot as a surface

% x = wristPos(1:stepPlot:end,1); 
% y = wristPos(1:stepPlot:end,2);
% z = wristPos(1:stepPlot:end,3);
% interpStep = 5;
% [xq,yq] = meshgrid(-600:interpStep:600, -600:interpStep:600);
% zq = griddata(x,y,z,xq,yq);
% figure()
% hold on
% plot3(x,y,z,'o')
