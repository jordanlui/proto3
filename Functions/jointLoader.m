function data = jointLoader(dataRaw,tolDistance,jointLabel,timeShift,freqShift)
    % Load raw data and format into a struct
    % Some secondary features like arm distances and angles are calculated
    % 2018 Feb Format didn't include magnetometer parsing
    % 2018 March includes Magnetometer
    if nargin < 5
        tolDistance = 1000;
        jointLabel = {'wrist','forearm','arm','chest','shoulder1','shoulder2','elbow'};
        timeShift = 0;
        freqShift = 0;
    end
    [M,N] = size(dataRaw);
    data.raw = dataRaw;
    data.jointLabel = jointLabel;
    N_inertial = 30; % Number of columns from inertial tracker
    
    % Correct for the different recording frequencies of the two sources
    % ART at 60 Hz, Proto at ~56.6 Hz
    resampleFactor = 56.6/60;
    resampleFactor = 58/60; % Trying new resample factor Apr 4 2018.
    if freqShift
        lenNew = floor(resampleFactor * M);
        ART.raw = dataRaw(:,31:end);
        ART.resample = interp1(1:M,ART.raw,linspace(1,M,lenNew));
        
        % Clip the proto data to match
        proto.raw = dataRaw(:,1:30);
        proto.resample = proto.raw(1:lenNew,:);
        
        % Write the new data to object, continue parsing
        dataRaw = [proto.resample, ART.resample];
        [M,N] = size(dataRaw);
    end
    
    % Correct for BIT ART timing mismatch. Positive value means BIT leads,
    % ART lags. Push BIT values later to match
    if timeShift ~=0
        if timeShift > 0
            newData = zeros(M,29);
            newData(1+timeShift:M,:) = dataRaw(1:M-timeShift,2:30);
            dataRaw(1:M,2:30) = newData;
        end
    end
    
    % Correction of empty data packets required in Feb23 recordings
    for i=2:size(dataRaw,1)
        if sum(dataRaw(i,2:27)) == 0
            dataRaw(i,:) = dataRaw(i-1,:);
        end
    end

    
    % Start parsing data. 
    % Parse inertial Device data
    ind = 1;
    data.time = dataRaw(:,ind);
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
    
    % Parse ART data. Check number of columns first.
    if N == 114

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
        data.distWristChest = sqrt(sum((data.wrist.pos-data.chest.pos).^2,2));

        % Elbow angle is the angle between wrist-elbow vector and elbow-arm vector
        vWristElbow = data.wrist.pos - data.elbow.pos;
        vElbowArm = data.arm.pos - data.elbow.pos;
        data.elbowAngle = vectorAngle(vWristElbow,vElbowArm);

        % Look for illegal distance values so we can ignore them
        data.distCheck = find(data.distWristChest < tolDistance);
        data.quatClean = cleanQuaternionTol(data.quat,1000);
        
    else
        % If we don't 114 columns then we don't have 7 joints in tracker
        numJoints = (N-N_inertial)/12; % Determine number of joints we have
        
        % Loop through and assign position and rotation for each joint
        for i = 1:numJoints
            indPos = N_inertial+1+3*(i-1);
            indRot = N_inertial+3*numJoints+1+9*(i-1);
            data.joints{i}.pos = dataRaw(:,indPos:indPos+2);
            
            data.joints{i}.rot = dataRaw(:,indRot:indRot+8);
        end
        
    end
    
end