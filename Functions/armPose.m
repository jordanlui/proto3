function [elbow, wrist] = armPose(dh,q)
    % Model arm pose based on DH parameter table. 
    % Returns coordinates of elbow and wrist
    % Kinematic model for Lui 2018, left arm
    % Angles in q refer to Sh Extension, Sh Rotation, Sh Abduction, El
    % Extension, Wrist Deviation, Wr Flexion, Wr Pronation
    
    
    dh(:,4) = dh(:,4) - q; % Modified angles
    
    elbow.T = transformDH(dh(1:4,:));
    wrist.T = transformDH(dh(1:7,:));    
    elbow.pos = elbow.T(1:3,4);
    wrist.pos = wrist.T(1:3,4);
    elbow.R = elbow.T(1:3,1:3);
    wrist.R = wrist.T(1:3,1:3);
end