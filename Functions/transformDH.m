function tform = transformDH(input)
    % Accepts DH params representing a link and returns the transform
    % matrix. Inputs expect in degrees
    
    if size(input,1) == 1
    
        alpha = input(1); a = input(2); d = input(3); theta=input(4);
        tform = [cosd(theta), -sind(theta), 0, a;
            sind(theta)*cosd(alpha), cosd(theta)*cosd(alpha), -sind(alpha), -sind(alpha)*d;
            sind(theta)*sind(alpha), cosd(theta)*sind(alpha), cosd(alpha), cosd(alpha)*d;
            0,0,0,1];
    else
        for i=1:size(input,1)
            alpha = input(i,1); a = input(i,2); d = input(i,3); theta=input(i,4);
            transforms{i} = [cosd(theta), -sind(theta), 0, a;
            sind(theta)*cosd(alpha), cosd(theta)*cosd(alpha), -sind(alpha), -sind(alpha)*d;
            sind(theta)*sind(alpha), cosd(theta)*sind(alpha), cosd(alpha), cosd(alpha)*d;
            0,0,0,1];
        end
        tform = transforms{1};
        for i = 2:size(input,1)
            tform = tform * transforms{i};
        end
    end
end