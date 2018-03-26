function [pose] = armPose(dh,q)
    % Model arm pose based on DH parameter table. 
    % Returns coordinates of elbow and wrist
    % Kinematic model for Lui 2018, left arm
    % Angles in q refer to Sh Extension, Sh Rotation, Sh Abduction, El
    % Extension, Wrist Deviation, Wr Flexion, Wr Pronation
    if nargin < 2
        dhparams = load('LuiArm2018.mat');
        dh = dhparams;
    end
    % Reverse value of angles for ShExt, ShRot, and ElEx from left arm
    q = q.* [-1 -1 1 -1 1 1 1]';
    dh(:,4) = dh(:,4) + q; % Modified angles
    
    elbow.T = transformDH(dh(1:4,:));
    elbow.pos = elbow.T(1:3,4);
    elbow.R = elbow.T(1:3,1:3);
    
    wrist.T = transformDH(dh(1:7,:));    
    wrist.pos = wrist.T(1:3,4);
    wrist.R = wrist.T(1:3,1:3);
    
    shoulder.T = transformDH(dh(1:3,:));
    shoulder.pos = shoulder.T(1:3,4);
    shoulder.R = shoulder.T(1:3,1:3);
    
    pose.elbow = elbow;
    pose.wrist = wrist;
    pose.shoulder = shoulder;
    
end