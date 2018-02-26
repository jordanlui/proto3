% Analysis of Feb 1 data
% IR Band Project


clc
clear all
close all

set(0, 'DefaultLineLineWidth', 2);
tolDistance = 1000; 
jointLabel = {'wrist','forearm','arm','chest','shoulder1','shoulder2','elbow'};

accScale = 8192; % Conversion parameter for accelerometer to g value
gyrScale = 16.4; % Conversion parameter for gyroscope to deg/s
freq = 60;
load('BITcalibration20170125.mat')
path1 = '../Analysis/feb23/';
addpath(path1,'Functions');
files = dir(strcat(path1,'*.csv'));
% files = dir('*.csv');

% dataRaw = csvread(files(9).name); % Queen wave
% dataRaw = dataRaw(775:end,:);

% dataRaw = csvread(files(12).name); % Standing pronation
% dataRaw = dataRaw(794:end,:);

% dataRaw = csvread(files(13).name); % Elbow flex
% dataRaw = dataRaw(660:end,:);
% 
% dataRaw = csvread(files(14).name); % Reach to collar
% dataRaw = dataRaw(775:end,:);

dataRaw = csvread(files(15).name); % Walking
data = jointLoader(dataRaw,tolDistance);

% dataRaw = csvread(files(21).name); % Reaching sideways to the left
% dataRaw = dataRaw(540:2185,:);

% dataRaw = csvread(files(28).name); % Reach for box, unaffected
% dataRaw = dataRaw(660:2230,:);

% dataRaw = csvread(files(30).name); % Reach for box, spastic
% dataRaw = dataRaw(796:end,:);



figure(1)
hold on; plot(data.wrist); plot(data.forearm); plot(data.arm); plot(data.chest); hold off
legend('wrist','forearm','arm','chest'); title('coordinates vs. time')

%% Derivations


%% Plots

figure(2)
plotallJoints3D(data.joints,'3d plot all joints',jointLabel)

% figure(3)
% hold on
% plot(distanceWristArm)
% plot(distWristChest)
% hold off
% legend('Wrist arm','wrist chest')
% 
% figure(4)
% plot(data.elbowAngle)
% title('Elbow Angle')
% 
% figure(5)
% hold on
% plot(distWristChest)
% plot(mean(omron,2))
% ylabel('distance, angle')
% yyaxis right
% plot(data.elbowAngle)
% ylabel('Elbow angle')
% xlabel('time')
% hold off
% title('elbow angle vs arm distance')
% legend('Distance wrist chest','omron','elbow angle')

% Consider Elbow angle as a function of other variables
% figure(6)
% suptitle('Elbow Angle analysis')
% subplot(2,2,1)
% scatter(distWristChest(distCheck,:),data.elbowAngle(distCheck,:))
% title('Arm distance vs Elbow angle')
% xlabel('Reach distance, mm')
% ylabel('Elbow angle, degrees')
% 
% subplot(2,2,2)
% scatter(mean(data.omron(distCheck,:),2),data.elbowAngle(distCheck,:))
% title('Elbow angle vs omron mean')
% xlabel('Mean temp'); ylabel('Elbow angle');
% 
% subplot(2,2,3)
% scatter(max(data.omron(distCheck,:),[],2),data.elbowAngle(distCheck,:))
% title('Elbow angle vs omron max')
% xlabel('Max temp'); ylabel('Elbow angle');
% 
% subplot(2,2,4)
% scatter(max(data.omron(distCheck,:),[],2)./mean(data.omron(distCheck,:),2),data.elbowAngle(distCheck,:))
% title('Elbow angle vs normalized omron max over mean')
% xlabel('Normed Max Temp'); ylabel('Elbow angle');

%%
% close all
% Consider Omron variances by columns and by rows
for i=1:4
    row.ind{i} = i:4:16;
    col.ind{i} = 4*(i-1)+1:4*(i-1)+4;
end
quadrant.ind{1} = [1 2 5 6];
quadrant.ind{2} = [3 4 7 8];
quadrant.ind{3} = [9 10 13 14];
quadrant.ind{4} = [11 12 15 16];

for i = 1:4
    row.mean{i} = mean(data.omron(distCheck,row.ind{i}),2);
    col.mean{i} = mean(data.omron(distCheck,col.ind{i}),2);
    quadrant.mean{i} = mean(data.omron(distCheck,quadrant.ind{i}),2);
    
    row.std{i} = std(data.omron(distCheck,row.ind{i}),[],2);
    col.std{i} = std(data.omron(distCheck,col.ind{i}),[],2);
    quadrant.std{i} = std(data.omron(distCheck,quadrant.ind{i}),[],2);
end

% figure(7)
% suptitle('By Row')
% subplot(1,3,1); title('Mean'); hold on; xlabel('time'); ylabel('temperature (1/10 C)')
% for i = 1:4;  plot(row.mean{i}); end; hold off; legend('1','2','3','4')
% subplot(1,3,2); title('Standard Deviation'); hold on; xlabel('time'); ylabel('temperature (1/10 C)')
% for i = 1:4; plot(row.std{i}); end; hold off; legend('1','2','3','4')
% subplot(1,3,3)
% for i = 1:4; hold on; scatter(distWristChest(distCheck,:),row.mean{i}); end; hold off; title('Averages vs Reach'); xlabel('Reach (mm)'); ylabel('Temp');legend('1','2','3','4');
% saveas(gcf,'omronRowwise.png')
% 
% figure(8)
% suptitle('By Column')
% subplot(1,3,1); title('Mean'); hold on; xlabel('time'); ylabel('temperature (1/10 C)')
% for i = 1:4;  plot(col.mean{i}); end; hold off; legend('1','2','3','4')
% subplot(1,3,2); title('Standard Deviation'); hold on; xlabel('time'); ylabel('temperature (1/10 C)')
% for i = 1:4; plot(col.std{i}); end; hold off; legend('1','2','3','4')
% subplot(1,3,3)
% for i = 1:4; hold on; scatter(distWristChest(distCheck,:),col.mean{i}); end; hold off; title('Averages vs Reach'); xlabel('Reach (mm)'); ylabel('Temp');legend('1','2','3','4');
% saveas(gcf,'omronColwise.png')
% 
% figure(9)
% suptitle('By Quadrant')
% subplot(1,3,1); title('Mean'); hold on; xlabel('time'); ylabel('temperature (1/10 C)')
% for i = 1:4;  plot(quad.mean{i}); end; hold off; legend('1','2','3','4')
% subplot(1,3,2); title('Standard Deviation'); hold on; xlabel('time'); ylabel('temperature (1/10 C)')
% for i = 1:4; plot(quad.std{i}); end; hold off; legend('1','2','3','4')
% subplot(1,3,3)
% for i = 1:4; hold on; scatter(distWristChest(distCheck,:),quad.mean{i}); end; hold off; title('Averages vs Reach'); xlabel('Reach (mm)'); ylabel('Temp');legend('1','2','3','4');
% saveas(gcf,'omronQuadwise.png')

figure(10)
suptitle('Temp distribution by row, column, quadrant')
nbins = 10;
for i = 1:4
    subplot(4,3,3*(i-1)+1)
    histogram(row.mean{i},nbins)
    xlim([210 300])
    
    subplot(4,3,3*(i-1)+2)
    histogram(col.mean{i},nbins)
    xlim([210 300])
    
    subplot(4,3,3*(i-1)+3)
    histogram(quadrant.mean{i},nbins)
    xlim([210 300])
end
saveas(gcf,'histogramRowColQuad.png')

%% Temp changes across different axes of movement
figure(11)
tempDirections(data.omron(distCheck,:),row,col,quadrant)
saveas(gcf,'TempSpan.png')


%% Functions
function data = jointLoader(dataRaw,tolDistance)
    % Load raw data and format into a struct
    % Correction of empty data packets required in Feb23 recordings
    for i=2:length(dataRaw)
        if sum(dataRaw(i,2:27)) == 0
            dataRaw(i,:) = dataRaw(i-1,:);
        end
    end

    [time, acc, gyr, quat, omron, wrist, forearm, arm, chest, shoulder1, shoulder2, elbow] = parseData(dataRaw);
    joints{1} = wrist; joints{2} = forearm; joints{3} = arm; joints{4} = chest; joints{5} = shoulder1; joints{6} = shoulder2;  joints{7} = elbow;
    

    % Load into struct
    data.raw = dataRaw;
    data.time = time;
    data.acc = acc;
    data.gyr = gyr;
    data.omron = omron;
    data.quat = quat;
    data.eul = quat2eul(data.quat);
    data.wrist = wrist;
    data.forearm = forearm;
    data.arm = arm;
    data.chest = chest;
    data.elbow = elbow;
    data.shoulder1 = shoulder1;
    data.shoulder2 = shoulder2;
    data.jointLabel = jointLabel;
    data.joints = joints;
    
    distanceWristArm = sqrt(sum((wrist-arm).^2,2));
    distWristChest = sqrt(sum((wrist-chest).^2,2));
    data.distWristChest = distWristChest;
    % Elbow angle is the angle between wrist-elbow vector and elbow-arm vector
    vWristElbow = wrist - elbow;
    vElbowArm = arm - elbow;
    data.elbowAngle = vectorAngle(vWristElbow,vElbowArm);

    % Look for illegal distance values so we can ignore them
    data.distCheck = find(distWristChest < tolDistance);
end


function tempDirections(omron,row,col,quadrant)
suptitle('Temp span across Row, Column, Quadrant')
hold on
ylabel('Temp difference')
xlabel('Time')
plot(max([row.mean{1},row.mean{2},row.mean{3},row.mean{4}],[],2) - min([row.mean{1},row.mean{2},row.mean{3},row.mean{4}],[],2),'b:')
plot(max([col.mean{1},col.mean{2},col.mean{3},col.mean{4}],[],2) - min([col.mean{1},col.mean{2},col.mean{3},col.mean{4}],[],2),'b--')
plot(max([quadrant.mean{1},quadrant.mean{2},quadrant.mean{3},quadrant.mean{4}],[],2) - min([quadrant.mean{1},quadrant.mean{2},quadrant.mean{3},quadrant.mean{4}],[],2),'b.')
yyaxis right
plot(mean(omron,2),'r')
ylabel('Mean temperature')
hold off
legend('row','col','quad','mean')
end