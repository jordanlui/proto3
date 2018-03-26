function plotallJoints3D(joints,caption,jointLabel)
    % Plot any joints in 3D
    if nargin > 1
        pltTitle = caption;
        labels = jointLabel;
    else
        pltTitle = '3D plot';
        labels = {'wrist','forearm','arm','chest','shoulder1','shoulder2','elbow'};
    end
    
    numJoints = length(joints);
    circSize = 1;
%     figure()
    hold on
    for i = 1:numJoints
        scatter3(joints{i}.pos(:,1),joints{i}.pos(:,2),joints{i}.pos(:,3),circSize)
    end
    hold off
    xlabel('x')
    ylabel('y')
    zlabel('z')
    legend(labels)
    title(pltTitle)
end