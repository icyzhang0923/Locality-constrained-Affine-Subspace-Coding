function encoder = trainEncoder(images, scales, maxSize, varargin)
% TRAINENCODER   Train image encoder: BoVW, VLAD, FV
%   ENCODER = TRAINENCOER(IMAGES) trains a BoVW encoder from the
%   specified list of images IMAGES.
%
%   TRAINENCODER(..., 'OPT', VAL, ...) accepts the following options:
%
%   Type:: 'bovw'
%     Bag of visual words ('bovw'), VLAD ('vlad') or Fisher Vector
%     ('fv').
%
%   numPcaDimension:: +inf
%     Use PCA to reduce the descriptor dimensionality to this
%     dimension. Use +inf to deactivate PCA.
%
%   Whitening:: false
%     Set to true to divide the principal components by the
%     corresponding standard deviation s_i.
%
%   WhiteningRegul:: 0
%     When using whitening, divide by s_max * WhiteningRegul + s_i
%     instead of s_i alone.
%
%   Renormalize:: false
%     If true, descriptors are L2 normalized after PCA or
%     whitening.
%
%
%   Subdivisions:: []
%     A list of spatial subdivisions. Each column is a rectangle
%     [XMIN YMIN XMAX YMAX]. The spatial subdivisions are
%
%   Layouts:: {'1x1'}
%     A list of strings representing regular spatial subdivisions
%     in the format MxN, where M is the number of vertical
%     subdivisions and N the number of horizontal ones. For
%     example {'1x1', 2x2'} uses 5 partitions: the whole image and
%     four quadrants. The subdivisions are appended to the ones
%     specified by the SUBDIVISIONS option.
%
%   ReadImageFn:: @readImage
%     The function used to load an image.
%
%   ExtractorFn:: @getDenseSIFT
%     The function used to extract the feature frames and
%     descriptors from an image.

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.type = 'bovw' ;
opts.numWords = [] ;
opts.seed = 1 ;
opts.numPcaDimensions = +inf ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.numSamplesPerWord = [] ;
opts.renormalize = false ;
opts.layouts = {'1x1'} ;
opts.geometricExtension = 'none' ;
opts.subdivisions = zeros(5,0) ;
opts.readImageFn = @readImage ;
opts.extractorFn = @getDenseSIFT ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

pgrid = zeros(1,numel(opts.layouts));
for i = 1:numel(opts.layouts)
  t = sscanf(opts.layouts{i},'%dx%d') ;
  pgrid(i) = t(1) * t(2);
end
weights = (1./pgrid);
weights = weights/sum(weights);

for i = 1:numel(opts.layouts)
  t = sscanf(opts.layouts{i},'%dx%d') ;
  m = t(1) ;
  n = t(2) ;
  [x,y] = meshgrid(...
    linspace(0,1,n+1), ...
    linspace(0,1,m+1)) ;
  x1 = x(1:end-1,1:end-1) ;
  y1 = y(1:end-1,1:end-1) ;
  x2 = x(2:end,2:end) ;
  y2 = y(2:end,2:end) ;
  w = repmat(weights(i), 1, m*n);
  opts.subdivisions = cat(2, opts.subdivisions, ...
    [x1(:)' ;
     y1(:)' ;
     x2(:)' ;
     y2(:)' ;
     w(:)'] ) ;
end

if isempty(opts.numWords)
    switch opts.type
      case {'fv'}
        opts.numWords = 64 ;
        opts.numPcaDimensions = 50 ;
      otherwise
        assert(false) ;
    end
end

if isempty(opts.numSamplesPerWord)
    switch opts.type
      case {'lasc','fv'}
        opts.numSamplesPerWord = 5000 ;
      otherwise
        assert(false) ;
    end
    if opts.lite
      opts.numSamplesPerWord = 10 ;
    end
end

disp(opts) ;
encoder.scales = scales ;
encoder.maxSize = maxSize ;
encoder.type = opts.type ;
encoder.subdivisions = opts.subdivisions ;
encoder.readImageFn = opts.readImageFn ;
encoder.extractorFn = opts.extractorFn ;
encoder.numWords = opts.numWords ;
encoder.renormalize = opts.renormalize ;
encoder.geometricExtension = opts.geometricExtension ;

%% Step 0: obtain sample image descriptors
numImages = numel(images) ;
numDescrsPerImage = ceil(opts.numWords * opts.numSamplesPerWord / numImages) ;
fprintf('%s: obtaining sample image descrptors...\n', mfilename) ;
for i = 1:numImages
  features = CalculateFeat(images{i}, encoder, encoder.maxSize);
  randn('state',0) ;
  rand('state',0) ;
  sel = vl_colsubset(1:size(features,2), single(numDescrsPerImage)) ;
  descrs{i} = features(:,sel) ;
end
descrs = cat(2, descrs{:}) ;


%% Step 1 (optional): learn PCA projection for fv and vlad
switch encoder.type
  case 'fv'
    if opts.numPcaDimensions < min(size(descrs,1),inf) || opts.whitening
      fprintf('%s: learning PCA rotation/projection\n', mfilename) ;
      encoder.projectionCenter = mean(descrs,2) ;
      x = bsxfun(@minus, descrs, encoder.projectionCenter) ;
      X = x*x' / size(x,2) ;
      [V,D] = eig(X) ;
      d = diag(D) ;
      [d,perm] = sort(d,'descend') ;
      d = d + opts.whiteningRegul * max(d) ;
      m = min(opts.numPcaDimensions, size(descrs,1)) ;
      V = V(:,perm) ;
      if opts.whitening
        encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
      else
        encoder.projection = V(:,1:m)' ;
      end
      clear X V D d ;
    else
      encoder.projection = 1 ;
      encoder.projectionCenter = 0 ;
    end
    descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;
    if encoder.renormalize
      descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
    end
end

%% Step 2 (optional): geometrically augment the features
% descrs = extendDescriptorsWithGeometry(opts.geometricExtension, frames, descrs) ;

%% Step 3: learn a VQ or GMM vocabulary
switch encoder.type
  case 'lasc'
    % k-means by sp_kmeans %    
    fprintf('%s: running k-means...\n', mfilename) ; 
    options = foptions;
    options(1) = 1;     % 1: display 0: no display
    options(2) = 1;
    options(3) = 1e-5;  % precision
    options(5) = 1;     % initialization
    options(14) = 10;   % maximum iterations
    centers = zeros(opts.numWords, size(descrs,1));    
    [encoder.words,index] = sp_kmeans(centers, descrs', options);
    encoder.words = encoder.words';  
    encoder.pca = Subspace_PCA(encoder.words, opts.numPcaDimensions, encoder.numWords, descrs, index) ;  
  case 'fv'
    vl_twister('state', opts.seed) ;
      v = var(descrs')' ;
      [encoder.means, encoder.covariances, encoder.priors] = ...
          vl_gmm(descrs, opts.numWords, 'verbose', ...
                 'Initialization', 'kmeans', ...
                 'CovarianceBound', double(max(v)*0.0001), ...
                 'NumRepetitions', 1) ;
   
end

