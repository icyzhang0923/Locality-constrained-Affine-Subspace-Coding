function [im, scale] = readImage(imagePath, varargin)
% READIMAGE   Read and standardize image
%    [IM, SCALE] = READIMAGE(IMAGEPATH) reads the specified image file,
%    converts the result to SINGLE class, and rescales the image
%    to have a maximum height of 480 pixels, returing the corresponding
%    scaling factor SCALE.
%
%    READIMAGE(IM) where IM is already an image applies only the
%    standardization to it.

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isempty(varargin)
    minSize = 512;
else
    minSize = varargin{1};
end

I = imread(imagePath);
if size(I,3) < 3
     I = repmat(I, [1 1 3]);
end
[height,width,~] = size(I);
if min(height,width) ~= minSize   
    I = imresize(I, minSize/min(height,width), 'bicubic'); 
end
im = single(I) ;
