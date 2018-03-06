function angle = vectorAngle(u,v)
    % Angle between two vectors
    CosTheta = dot(u',v')' ./ (sqrt(sum(u.^2,2)).*sqrt(sum(v.^2,2)));
    angle = acosd(CosTheta);
end