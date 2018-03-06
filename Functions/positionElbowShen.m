function P_elbow = positionElbowShen(theta1,theta2,Lu)
    % Calculate elbow posiiton based on Shen 2016 paper
    % Note the DH Parameter setup for this calculation are not known
    % theta1 is flexion, theta2 is abduction
    if nargin == 3
        P_elbow = Lu * [cosd(theta2) * sind(theta1); sind(theta2); -cosd(theta1) * cosd(theta2)];
    else
        disp('Not enough input parameters')
        P_elbow = 0;
    end
end