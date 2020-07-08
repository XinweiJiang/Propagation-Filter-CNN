function [ R ] = average_fusion( img,n )
    %PCA_FUSION Summary of this function goes here
    %   Detailed explanation goes here
    [no_lines, no_rows, no_bands] = size(img);
    img=reshape(img,[no_lines*no_rows no_bands]);
    [mappedX, mapping] = compute_mapping(img, 'PCA', n);
    img=mappedX;
    R=reshape(img,[no_lines no_rows n]);
end
