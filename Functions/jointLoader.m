function data = jointLoader(dataRaw,tolDistance,jointLabel,timeShift)
    % Load raw data and format into a struct
    % Some secondary features like arm distances and angles are calculated
    if nargin < 2
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

    % Parse out variables
    time = dataRaw(:,1);
    omron = dataRaw(:,2:17);
    acc = dataRaw(:,18:20);
    gyr = dataRaw(:,21:23);
    quat = dataRaw(:,24:27);
    
    wrist.pos = dataRaw(:,28:30);
    forearm.pos = dataRaw(:,31:33);
    arm.pos = dataRaw(:,34:36);
    chest.pos = dataRaw(:,37:39);
    shoulder1.pos = dataRaw(:,40:42);
    shoulder2.pos = dataRaw(:,43:45);
    elbow.pos = dataRaw(:,46:48);
    
    ind = 49; 
    wrist.rot = dataRaw(:,ind:ind+8);
    ind = 58; 
    forearm.rot = dataRaw(:,ind:ind+8);
    ind = 67; 
    arm.rot = dataRaw(:,ind:ind+8);
    ind = 76; 
    chest.rot = dataRaw(:,ind:ind+8);
    ind = 85; 
    shoulder1.rot = dataRaw(:,ind:ind+8);
    ind = 94; 
    shoulder2.rot = dataRaw(:,ind:ind+8);
    ind = 103; 
    elbow.rot = dataRaw(:,ind:ind+8);
    
    
    
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
    
    distanceWristArm = sqrt(sum((wrist.pos-arm.pos).^2,2));
    distWristChest = sqrt(sum((wrist.pos-chest.pos).^2,2));
    data.distWristChest = distWristChest;
    % Elbow angle is the angle between wrist-elbow vector and elbow-arm vector
    vWristElbow = wrist.pos - elbow.pos;
    vElbowArm = arm.pos - elbow.pos;
    data.elbowAngle = vectorAngle(vWristElbow,vElbowArm);

    % Look for illegal distance values so we can ignore them
    data.distCheck = find(distWristChest < tolDistance);
end