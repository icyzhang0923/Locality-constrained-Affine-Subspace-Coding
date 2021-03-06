function  encodeImage(encoder, im, kernel, varargin)
% ENCODEIMAGE   Apply an encoder to an image
%   DESCRS = ENCODEIMAGE(ENCODER, IM) applies the ENCODER
%   to image IM, returning a corresponding code vector PSI.
%
%   IM can be an image, the path to an image, or a cell array of
%   the same, to operate on multiple images.
%
%   ENCODEIMAGE(ENCODER, IM, CACHE) utilizes the specified CACHE
%   directory to store encodings for the given images. The cache
%   is used only if the images are specified as file names.
%
%   See also: TRAINENCODER().

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.cacheDir = [] ;
opts.cacheChunkSize = 512 ;
opts = vl_argparse(opts,varargin) ;

if ~iscell(im), im = {im} ; end

% break the computation into cached chunks
startTime = tic ;
numChunks = ceil(numel(im) / opts.cacheChunkSize) ;

for c = 1:numChunks
  n  = min(opts.cacheChunkSize, numel(im) - (c-1)*opts.cacheChunkSize) ;
  chunkPath = fullfile(opts.cacheDir, sprintf('chunk-%03d.mat',c)) ;
  if ~isempty(opts.cacheDir) && exist(chunkPath)
     fprintf('%s: skipping %s\n', mfilename, chunkPath);
  else
    range = (c-1)*opts.cacheChunkSize + (1:n) ;%
    fprintf('%s: processing a chunk of %d images (%3d of %3d, %5.1fs to go)\n',mfilename, numel(range), c, numChunks, toc(startTime) / (c - 1) * (numChunks - c + 1)) ;
    data = processChunk(encoder, im(range));
    % apply kernel maps
    switch kernel
      case 'linear', data = single(data);
      case 'hell', data = single(data); data = sign(data) .* sqrt(abs(data)) ;
      case 'chi2', data = single(data); data = vl_homkermap(data,1,'kchi2') ;
      otherwise, assert(false) ;
    end
        data = bsxfun(@times, data, 1./(sqrt(sum(data.^2))+eps)) ;                                                                                                                                          
    if ~isempty(opts.cacheDir)
      save(chunkPath, 'data', '-v7.3') ;
    end
  end
  clear data ;
end

% --------------------------------------------------------------------
function psi = processChunk(encoder, im)
% --------------------------------------------------------------------
psi = cell(1,numel(im)) ;
for i = 1:numel(im)
    psi{i} = encodeOne(encoder, im{i}) ;
end
psi = cat(2, psi{:}) ;

% --------------------------------------------------------------------
function z = encodeOne(encoder, Im)
% --------------------------------------------------------------------
descrs =  CalculateFeat(Im, encoder, 384);

switch encoder.type
    case 'lasc'
     z = lasc_encode(descrs, encoder); 

    case 'fv'
      descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;
      if encoder.renormalize
         descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
      end
      z = vl_fisher(descrs, encoder.means, encoder.covariances, encoder.priors, 'Improved') ;
        
z = z(:);
end





