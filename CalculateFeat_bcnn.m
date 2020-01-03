function features = CalculateFeat_bcnn(imPath, encoder, maxSize)

%% load features in cell type %%
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

