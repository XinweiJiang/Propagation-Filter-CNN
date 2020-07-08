function [fimage] = spatial_feature( img,w,sigma)
%SPATIAL_FEATURE Summary of this function goes here
%   Detailed explanation goes here
% w = r;
% sigma = 40;
[m, n, o] = size(img);
fimage = zeros(m, n, o);

for z = 1:o
    if mod(z,round(o/10))==0
            fprintf('*...');
    end
    for i=1:m
        for j=1:n
            [result] = weight(img(:, :, z), w, i, j, sigma) ;
            fimage(i, j, z) = result;
        end    
    end    
end
