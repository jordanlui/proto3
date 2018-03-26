function data = jointLoader(dataRaw,tolDistance,jointLabel,timeShift)
    % Load raw data and format into a struct
    % Some secondary features like arm distances and angles are calculated
    % 2018 Feb Format didn't include magnetometer parsing
    % 2018 March includes Magnetometer
    if nargin < 4
        tolDistance = 1000;
        jointLabel = {'wrist','forearm','arm','chest','shoulder1','shoulder2','elbow'};
        timeShift = 0;
    end
    [M,N] = size(dataRaw);
    % Correct for BIT ART timing mismatch. Positive value means BIT leads,
    % ART lags. Push BIT values later to match
    if timeShift ~=0
        if timeShift > 0
            newData = zeros(M,26);
            newData(1+timeShift:M,:) = dataRaw(1:M-timeShift,2:27);
            dataRaw(1:M,2:27) = newData;
        end
    end
    
    % Correction of empty data packets required in Feb23 recordings
    for i=2:length(dataRaw)
        if sum(dataRaw(i,2:27)) == 0
            dataRaw(i,:) = dataRaw(i-1,:);
        end
    end

    data.raw = dataRaw;
    data.jointLabel = jointLabel;
    
    ind = 1;
    time = dataRaw(:,ind);
    ind = ind + 1;
    data.omron = dataRaw(:,ind:ind+15);
    ind = ind + 16;
    data.acc = dataRaw(:,ind:ind+2);
    ind = ind + 3;
    data.gyr = dataRaw(:,ind:ind+2);
    ind = ind + 3;
    data.mag = dataRaw(:,ind:ind+2);
    ind = ind + 3;
    data.quat = dataRaw(:,ind:ind+3);
    ind = ind + 4;
    data.eul = quat2eul(data.quat);
    
    data.wrist.pos = dataRaw(:,ind:ind+2); ind = ind + 3;
    data.forearm.pos = dataRaw(:,ind:ind+2); ind = ind + 3;
    data.arm.pos = dataRaw(:,ind:ind+2); ind = ind + 3;
    data.chest.pos = dataRaw(:,ind:ind+2); ind = ind + 3;
    data.shoulder1.pos = dataRaw(:,ind:ind+2); ind = ind + 3;
    data.shoulder2.pos = dataRaw(:,ind:ind+2); ind = ind + 3;
    data.elbow.pos = dataRaw(:,ind:ind+2); ind = ind + 3;
    
    data.wrist.rot = dataRaw(:,ind:ind+8); ind = ind + 9;
    data.forearm.rot = dataRaw(:,ind:ind+8); ind = ind + 9;
    data.arm.rot = dataRaw(:,ind:ind+8); ind = ind + 9;
    data.chest.rot = dataRaw(:,ind:ind+8); ind = ind + 9;
    data.shoulder1.rot = dataRaw(:,ind:ind+8); ind = ind + 9;
    data.shoulder2.rot = dataRaw(:,ind:ind+8); ind = ind + 9;
    data.elbow.rot = dataRaw(:,ind:ind+8);
    
    
    data.joints = {data.wrist, data.forearm, data.arm, data.chest, data.shoulder1, data.shoulder2, data.elbow};
%     joints{1} = wrist; joints{2} = forearm; joints{3} = arm; joints{4} = chest; joints{5} = shoulder1; joints{6} = shoulder2;  joints{7} = elbow;
    
    % Load into struct
%     data.raw = dataRaw;
%     data.time = time;
%     data.acc = acc;
%     data.gyr = gyr;
%     data.omron = omron;
%     data.quat = quat;
%     data.eul = quat2eul(data.quat);
%     data.wrist = wrist;
%     data.forearm = forearm;
%     data.arm = arm;
%     data.chest = chest;
%     data.elbow = elbow;
%     data.shoulder1 = shoulder1;
%     data.shoulder2 = shoulder2;
%     
%     data.joints = joints;
    
    
    data.distWristChest = sqrt(sum((data.wrist.pos-data.chest.pos).^2,2));
    
    % Elbow angle is the angle between wrist-elbow vector and elbow-arm vector
    vWristElbow = data.wrist.pos - data.elbow.pos;
    vElbowArm = data.arm.pos - data.elbow.pos;
    data.elbowAngle = vectorAngle(vWristElbow,vElbowArm);

    % Look for illegal distance values so we can ignore them
    data.distCheck = find(data.distWristChest < tolDistance);
end