function descrs = getDenseCNNres(im, net, varargin)

opts.scales = logspace(log10(1), log10(.25), 5) ;
opts.contrastthreshold = 0 ;
opts.step = 3 ;
opts.rootSift = true ;
opts.normalizeSift = true ;
opts.binSize = 8 ;
opts.geometry = [4 4 8] ;
opts.sigma = 0 ;
opts = vl_argparse(opts, varargin) ;

% gpuDevice(1);
[x1, x2, ~] = size(im);
min_scale = 225.0 / min([x1, x2]);
descrs = cell(1,numel(opts.scales));
for si = 1:numel(opts.scales)
  
  im_ = imresize(im, max(opts.scales(si),min_scale)) ;
  [height,width,~] = size(im_);
    if max(height,width) > 1200   
        im_ = imresize(im_, 1200/max(height,width), 'bicubic'); 
    end
  mean_im = imresize(net.meta.normalization.averageImage, [size(im_,1), size(im_,2)]);
  im_ = im_ - mean_im;
  im_ = gpuArray(im_) ;
  res = vl_simplenn(net, im_) ;
  clear im_
  feats = gather(res(32).x);  
  clear res
  descrs{si} = feats;
  
end



