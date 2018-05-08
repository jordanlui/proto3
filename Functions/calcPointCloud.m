function [rotation,elbowPos,wristPos,angles] = calcPointCloud(ROM,stepSize)
% Create point cloud of possible elbow and arm locations
% ROM input angles are for [ShFlx, ShAbd, ShRot, ElbFlx, ElbPro]
% Input should be a 5x2 matrix

load('LuiArm2018.mat')
if nargin <2
    stepSize = 20;
end
printoutStep = 2e4;

disp('Starting permuations. Expected number iterations:')
numIter = ceil(prod((ROM(:,2) - ROM(:,1))./stepSize));
angles = zeros(numIter,5);
elbowPos = zeros(numIter,3);
wristPos = zeros(numIter,3);
rotation = zeros(numIter,9);

cnt = 1;
tic

for i=ROM(1,1):stepSize:ROM(1,2)
    for j = ROM(2,1):stepSize:ROM(2,2)
        for k = ROM(3,1):stepSize:ROM(3,2)
            for l = ROM(4,1):stepSize:ROM(4,2)
                for m = ROM(5,1):stepSize:ROM(5,2)
                    [pose] = armPose(dhparams,[i,k,j,l,0,0,m]');
                    angles(cnt,:) = [i j k l m];
                    rotation(cnt,:) = reshape(pose.wrist.R,[1,9]);
                    elbowPos(cnt,:) = pose.elbow.pos;
                    wristPos(cnt,:) = pose.wrist.pos;
                    cnt = cnt + 1;
                    if mod(cnt,printoutStep)==0
                        cnt
                    end
                end
            end
        end
    end
end

toc

disp(cnt)


end