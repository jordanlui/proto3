function [time, acc, gyr, quat, omron, wrist , forearm, arm, back, shoulder1, shoulder2, elbow] = parseData(data)
    % Plots ART and proto IMU/omron data
    time = data(:,1);
    omron = data(:,2:17);
    acc = data(:,18:20);
    gyr = data(:,21:23);
    quat = data(:,24:27);
    
    wrist = data(:,28:30);
    forearm = data(:,31:33);
    arm = data(:,34:36);
    back = data(:,37:39);
    shoulder1 = data(:,40:42);
    shoulder2 = data(:,43:45);
    elbow = data(:,46:48);
end