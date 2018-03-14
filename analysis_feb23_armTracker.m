% Analysis of Feb 23 data
% IR Band Project

% Using Shen 2016 Point Cloud method to determine arm pose
% Mar 4: System shows overall trend but there is a transform missing on
% wrist and elbow moving too much


clc
clear all
close all

set(0, 'DefaultLineLineWidth', 2);
tolDistance = 1000; 
tolRotation = 1.5; % Tolerance on rotation search similarity
jointLabel = {'wrist','forearm','arm','chest','shoulder1','shoulder2','elbow'};
freq = 60;
T = 1/freq;
% Trealwrist = [1 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 1]; % Arbitrary transform for wrist mounted tracker
T7W = 0;
% T7W = [0 0 1; 1 0 0; 0 1 0]; % Attempt 2, Jordan attempt
% T7W = [1 0 0; 0 0 -1; 0 1 0]; % From Ahmed March 5th
T7W = [1 0 0; 0 0 1;0 -1 0]; % Inverse of Ahmed matrix
% T7W = [0 -1 0; 1 0 0; 0 0 1]; % Arbitrary transform that works, Mar 11. Transform Wrist ART to W7 frame
% T7W = [0 0 1; 1 0 0; 0 -1 0]; % By inspection, it seems we should make flip sign on x,y. Doesn't work

% T7P = [1 0 0; 0 0 -1; 0 1 0]; % Transform from wrist proto IMU to frame 7

accScale = 8192; % Conversion parameter for accelerometer to g value
gyrScale = 16.4; % Conversion parameter for gyroscope to deg/s
freq = 60;
load('BITcalibration20170125.mat')
path1 = '../Analysis/feb23/';
addpath(path1,'Functions');
files = dir(strcat(path1,'*.csv'));

dataRaw = csvread(files(13).name); % Elbow flex 180-90, Shoulder flexed 90
% dataRaw = csvread(files(10).name); % Downwards neutral pronation
% dataRaw = dataRaw(660:end,:); % Clipping the macro level data
% dataRaw = dataRaw(ind:ind+120,:);% Grabbing a flexion action
% dataRaw = csvread(files(14).name); % Reaching to collar action
% dataRaw = csvread(files(16).name); % Elbow 90, rotating shoulder 180 up and down
% dataRaw = csvread(files(21).name); % Reach left

% Clip Data
% Movement?
ind = 1710;
span = 200;

% Stationary, hand at side
% ind = 590;
% span = 150;


% Note the device values appear to need shifting by 44 steps for Feb 23
data = jointLoader(dataRaw,tolDistance,jointLabel,44);
% Scale gyro values into proper numbers
data.gyr = (data.gyr-gyrCal) / gyrScale;


% figure(1)
% hold on; 
% plot(data.wrist.pos(ind:ind+span,:)); 
% ylabel('Distance (mm)'); xlabel('time')
% yyaxis right
% plot(data.gyr(ind:ind+span,:),':'); ylabel('Gyro, deg/s')
% hold off
% legend('wx','wy','wz','gyrX','gyrY','gyrZ'); title('coordinates vs. time')

figure(2)
hold on; 
plot(data.wrist.pos); 
ylabel('Distance (mm)'); xlabel('time')
yyaxis right
plot(data.gyr,':'); ylabel('Gyro, deg/s')
hold off
legend('wx','wy','wz','gyrX','gyrY','gyrZ'); title('coordinates vs. time')

%% Plot Quaternion data, Compare ART and Proto
% % Convert Quat for ART
% ARTQUAT = rotm2quat(reshape(data.wrist.rot(:,:),[3,3,length(data.wrist.rot)]));
% ARTEUL = rotm2eul(reshape(data.wrist.rot(:,:),[3,3,length(data.wrist.rot)]));
% 
% % Fix BIT Quaternion zeros
% findZeros = [1;1;3;3];
% % while length(findZeros>2)
% findZeros = find(sum(data.quat(ind:ind+span,:),2)==0);
% data.quat(findZeros+ind-1,:) = data.quat(findZeros-2+ind,:);
% 
% % Plot
% figure(3)
% % ind = 2000;
% % span = 500;
% subplot(2,2,1)
% plot(data.quat(ind:ind+span,:));
% title('BIT Quaternion')
% subplot(2,2,2)
% plot(ARTQUAT(ind:ind+span,:));
% title('ART Quaternion')
% 
% subplot(2,2,3)
% plot(data.eul(ind:ind+span,:));
% title('BIT Device Euler'), legend('x','y','z')
% subplot(2,2,4)
% plot(ARTEUL(ind:ind+span,:));
% title('ART System Euler'),  legend('x','y','z')

%% Determine arm length based on ART MoCap system readings throughout trial
t1 = 1000;
t2 = 3070;
Ll = sqrt(sum((data.wrist.pos(t1:t2,:) - data.elbow.pos(t1:t2,:)).^2,2));
Ll = mean(Ll);
Lu = sqrt(sum((data.elbow.pos(t1:t2,:) - data.shoulder2.pos(t1:t2,:)).^2,2));
Lu = mean(Lu);

% figure(3)
% plotallJoints3D(data.joints,'3d plot all joints',jointLabel)

% save('measurementsfeb23.mat','Ll','Lu')

%% Arm pose and wrist calibration
% Get arm pose and transform info while in neutral pose. Feb 23 File 13
% k_neutral = 630; % Neutral, with hand at side
% disp('neutral pose')
% for i = [1,6,7]
%     disp(data.jointLabel(i))
%     rotm2eul(reshape(data.joints{i}.rot(k_neutral,:),[3 3]))  * 360/2/pi
% end
% data.eul(k_neutral,:) * 360/2/pi
% 
% % Pose while shoulder flexed 90, palm facing down towards ground
% k_ShFlex90 = 845;
% disp('Shoulder flex 90')
% for i = [1,6,7]
%     disp(data.jointLabel(i))
%     rotm2eul(reshape(data.joints{i}.rot(k_ShFlex90,:),[3 3]))  * 360/2/pi
% end
% data.eul(k_ShFlex90,:) * 360/2/pi
% 
% wristNeutral = reshape(data.joints{1}.rot(k_neutral,:),[3 3]);
% wristShRaise = reshape(data.joints{1}.rot(k_ShFlex90,:),[3 3]);
% rotm2eul(wristNeutral) * 360/2/pi
% rotm2eul(wristShRaise) * 360/2/pi
% 
% T_neutral = rotm2tform(wristNeutral);
% T_neutral(1:3,end) = data.wrist.pos(k_neutral,:)'
% 
% T_ShRaise = rotm2tform(wristShRaise);
% T_ShRaise(1:3,end) = data.wrist.pos(k_ShFlex90,:)'
%% Arm Tracker logic
% load('pointCloudShen_step20.mat')
load('pointCloudRosen_step20.mat')

% Put work space in terms of shoulder centered space

data.wrist.pos = data.wrist.pos - data.shoulder2.pos;
data.elbow.pos = data.elbow.pos - data.shoulder2.pos;

% Careful - Transpose required
data.wrist.rot = (reshape(data.wrist.rot',[3,3,length(data.wrist.rot)]));
data.elbow.rot = reshape(data.elbow.rot',[3,3,length(data.elbow.rot)]);
data.shoulder.rot = reshape(data.shoulder2.rot',[3,3,length(data.shoulder2.rot)]);


% Transform the ART System coordinate for wrist into the kinematic model
% for the wrist frame
if any(T7W)
    for i = 1:size(data.wrist.rot,3)
        data.wrist.rot(:,:,i) = T7W * data.wrist.rot(:,:,i);
    end
end
% Transform the wrist proto data into frame 7
% for i = 1:length(data.quat)
%     if all(data.quat(i,:) == 0)
%         % Do nothing - just leave 0 values in the quaternion
%     else
%         data.quat(i,:) = rotm2quat(T7P *  quat2rotm(data.quat(i,:)));
%     end
% end

data.wrist.eul = rotm2eul(data.wrist.rot);
data.elbow.eul = rotm2eul(data.elbow.rot);
data.shoulder.eul = rotm2eul(data.shoulder.rot);
% Plot angular changes

% Compare orientation changes of 3 limbs and compare to proto device
figure(4)
ax1 = subplot(4,1,1);
plot(data.wrist.eul(:,:)); 
title('wrist, ART'), legend('x','y','z')
ax2 = subplot(4,1,2);
plot(data.eul(:,:));  
title('wrist, Proto'),legend('x','y','z')
ax3 = subplot(4,1,3);
plot(data.elbow.eul); 
title('elbow, ART'),legend('x','y','z')
ax4 = subplot(4,1,4);
plot(data.shoulder.eul)
title('shoulder, ART'),legend('x','y','z')
suptitle('Euler Angle changes')
linkaxes([ax1,ax2,ax3,ax4],'x')




%% Use rotation matrix look up
ind = 350;
indend = 3070;
% Create a search object s
for i=1:(indend-ind)
    % Get wrist, shoulder orientation
    s.rotW{i} = reshape(data.wrist.rot(:,:,i+ind),[1,9]);
    s.rotS{i} = data.joints{6}.rot(i+ind,:);
    
    % Get joint coordinates, in Shoulder Coordinate space
    s.shoulder{i} = data.joints{6}.pos(i+ind,:);
    % Wrist, shoulder into Torso Coordinate space
    s.wrist{i} = data.joints{1}.pos(i+ind,:) - s.shoulder{i};
    s.elbow{i} = data.joints{7}.pos(i+ind,:) - s.shoulder{i};
    

    % Searching for a rotation match - compare wrist rotation against database
    rotDiff = sum(abs(rotation - s.rotW{i}),2);
    lowestDiff = min(rotDiff);
    % Look for the closest matching wrist orientation
    % Add a weighted or average approach here
    guess.ind = find(rotDiff <= tolRotation);  
    if length(guess.ind) > 1
        guess.elbowPos{i} = mean(elbowMat(guess.ind,:))';
        guess.wristPos{i} = mean(wristMat(guess.ind,:))';
    else
        guess.wristPos{i} = wristMat(guess.ind,:)';
        guess.elbowPos{i} = elbowMat(guess.ind,:)'; 
    end
    % Single optimum approach
%     guess.ind = find(rotDiff == lowestDiff,1);
%     guess.wristPos{i} = wristMat(guess.ind,:)';
%     guess.elbowPos{i} = elbowMat(guess.ind,:)';
    
    
    guess.rot{i} = rotation(guess.ind,:);
    guess.erWrist{i} = sqrt(sum((guess.wristPos{i} - s.wrist{i}').^2));
    guess.erElbow{i} = sqrt(sum((guess.elbowPos{i} - s.elbow{i}').^2));
    
    
end

%% Plot out result
guessedWrist = (cell2mat(guess.wristPos))';

% Window smoothing was tried here, but ultimately distorts the output. HMM!

guessedElbow = (cell2mat(guess.elbowPos))';
errorWrist = (abs(cell2mat(guess.erWrist)))';
errorElbow = (abs(cell2mat(guess.erElbow)))';

realWrist = (cell2mat(s.wrist'));
realElbow = cell2mat(s.elbow');
realShoulder = cell2mat(s.shoulder');


realWristDist = sqrt(sum(realWrist.^2,2));
realElbowDist = sqrt(sum(realElbow.^2,2));
guessWristdist = sqrt(sum((guessedWrist).^2,2));
guessElbowdist = sqrt(sum(guessedElbow.^2,2));
sprintf('Mean error wrist: %.2f, elbow: %.2f',mean(errorWrist),mean(errorElbow))

figure(6)
subplot(2,2,1)
hold on
plot(guessedWrist(:,1),'r:')
plot(guessedWrist(:,2),'m:')
plot(guessedWrist(:,3),'b:')
plot(realWrist(:,1),'r')
plot(realWrist(:,2),'m')
plot(realWrist(:,3),'b')
hold off
legend('x','y','z','x','y','z')
title('Wrist real and predicted')

subplot(2,2,2)
plot(guessedElbow(:,1),'r:'), hold on
plot(guessedElbow(:,2),'m:')
plot(guessedElbow(:,3),'b:')
plot(realElbow(:,1),'r')
plot(realElbow(:,2),'m')
plot(realElbow(:,3),'b')
hold off
legend('x','y','z','x','y','z')
title('Elbow real and predicted')

subplot(2,2,3)
plot(errorWrist,'r.','LineStyle','none'), ylabel('Error (mm)')
yyaxis right
hold on
scatter(1:length(guessWristdist),guessWristdist,3,'m')
scatter(1:length(realWristDist),realWristDist,3,'b')
ylabel('Distance (mm)')
hold off
title('Wrist Error')
legend('error','Prediction','real')

subplot(2,2,4)
plot(errorElbow,'r.','LineStyle','none'), ylabel('Error (mm)')
yyaxis right
hold on
scatter(1:length(guessElbowdist),guessElbowdist,3,'m')
scatter(1:length(realElbowDist),realElbowDist,3,'b')
hold off
title('Elbow Error')
legend('error','Prediction','real')

figure(7)
subplot(4,2,1)
hold on
plot(guessedWrist(:,1),'r:')
plot(realWrist(:,1),'r'), hold off, title('Wrist Position, x')

subplot(4,2,3)
hold on
plot(guessedWrist(:,2),'m:')
plot(realWrist(:,2),'m'), hold off, title('y')

subplot(4,2,5)
hold on
plot(guessedWrist(:,3),'b:')
plot(realWrist(:,3),'b'), hold off, title('z')

subplot(4,2,2)
hold on
plot(guessedElbow(:,1),'r:')
plot(realElbow(:,1),'r'), hold off, title('Elbow Position, x')

subplot(4,2,4)
hold on
plot(guessedElbow(:,2),'m:')
plot(realElbow(:,2),'m'), hold off, title('y')

subplot(4,2,6)
hold on
plot(guessedElbow(:,3),'b:')
plot(realElbow(:,3),'b'), hold off, title('z')

subplot(4,2,7)
hold on
plot(errorWrist), title('Wrist Error')
plot(errorWrist - errorElbow), legend('Wrist overal error','Wrist Error w/o elbow')
subplot(4,2,8)
plot(errorElbow), title('Elbow Error')

%% Functions

function compareTrials(data1,data2)
    
    [row1, col1, quadrant1] = omronAnalysis(data1.omron(data1.distCheck,:));
    figure(4)
    subplot(2,2,1); title('Temp Span, Nominal')
    tempDirections(data1.omron(data1.distCheck,:),row1,col1,quadrant1)
    subplot(2,2,3); title('Temp vs. Distance')
    tempDistance(data1.distWristChest(data1.distCheck),data1.omron(data1.distCheck,:),row1,col1,quadrant1)

    
    [row2, col2, quadrant2] = omronAnalysis(data2.omron(data2.distCheck,:));
    subplot(2,2,2); title('Temp Span, Affected')
    tempDirections(data2.omron(data2.distCheck,:),row2,col2,quadrant2)
    subplot(2,2,4); title('Temp vs. Distance')
    tempDistance(data2.distWristChest(data2.distCheck),data2.omron(data2.distCheck,:),row2,col2,quadrant2)

end

function [row,col,quadrant] = omronAnalysis(omron)
    % Consider Omron variances by columns and by rows
    % Return stats by column, row, quadrant
    for i=1:4
        row.ind{i} = i:4:16;
        col.ind{i} = 4*(i-1)+1:4*(i-1)+4;
    end
    quadrant.ind{1} = [1 2 5 6];
    quadrant.ind{2} = [3 4 7 8];
    quadrant.ind{3} = [9 10 13 14];
    quadrant.ind{4} = [11 12 15 16];

    for i = 1:4
        row.mean{i} = mean(omron(:,row.ind{i}),2);
        col.mean{i} = mean(omron(:,col.ind{i}),2);
        quadrant.mean{i} = mean(omron(:,quadrant.ind{i}),2);

        row.std{i} = std(omron(:,row.ind{i}),[],2);
        col.std{i} = std(omron(:,col.ind{i}),[],2);
        quadrant.std{i} = std(omron(:,quadrant.ind{i}),[],2);
    end

end

function tempDistance(distance,omron,row,col,quadrant)
    % Plot temperature differences in against distance
    row.diff = max([row.mean{1},row.mean{2},row.mean{3},row.mean{4}],[],2) - min([row.mean{1},row.mean{2},row.mean{3},row.mean{4}],[],2);
    col.diff = max([col.mean{1},col.mean{2},col.mean{3},col.mean{4}],[],2) - min([col.mean{1},col.mean{2},col.mean{3},col.mean{4}],[],2);
    quadrant.diff = max([quadrant.mean{1},quadrant.mean{2},quadrant.mean{3},quadrant.mean{4}],[],2) - min([quadrant.mean{1},quadrant.mean{2},quadrant.mean{3},quadrant.mean{4}],[],2);
    suptitle('Relating Omron to distance')
    hold on
    ylabel('Temp difference')
    xlabel('distance')
    scatter(distance,row.diff)
    scatter(distance,col.diff)
    scatter(distance,quadrant.diff)
%     yyaxis right
%     scatter(distance,mean(omron,2))
%     ylabel('Mean temperature')
    hold off
    legend('row','col','quad','mean')
end

function tempDirections(omron,row,col,quadrant)
    % Plot temperature differences in various axis directions
    
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