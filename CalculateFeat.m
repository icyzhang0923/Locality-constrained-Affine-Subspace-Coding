function features = CalculateFeat(imPath, encoder, maxSize)

norm = 'matrix'; % 'l2', 'none', 'root', 'matrix'

%%% load features in cell type %%%
feat_dir = ['features' imPath(5:end-4) '.mat'];
if exist(feat_dir, 'file')
    load(feat_dir, 'features');
else
    im = encoder.readImageFn(imPath, maxSize) ;
    features = encoder.extractorFn(im) ;  
    if ~exist(fileparts(feat_dir),'dir')
        mkdir(fileparts(feat_dir));
    end
    save(feat_dir, 'features') ;
end

for si = 1:numel(features)
    feats = features{si};
    feats = reshape(feats, size(feats,1)*size(feats,2), size(feats,3))';   
    switch norm
    case 'none'
        features{si} = feats ;
    case 'l2'
        features{si} = bsxfun(@times, feats, 1./(sqrt(sum(feats.^2))+eps)) ;
    case 'root'
        features{si} = bsxfun(@times, feats, 1./(sqrt(sum(feats.^2))+eps)) ;
        features{si} = features{si}.^0.5 ;
    case 'matrix' 
        [~, s, ~] = svd(feats);  
        features{si} = feats./max(diag(s));
    end
end
features = cat(2, features{:});


