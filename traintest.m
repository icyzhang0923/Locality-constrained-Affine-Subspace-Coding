% --------------------------------------------------------------------
%              Parameter settings
% --------------------------------------------------------------------
clear
run(fullfile(fileparts(which(mfilename)), 'VLFeat', 'toolbox', 'vl_setup.m')) ;
run(fullfile(fileparts(which(mfilename)), 'matconvnet', 'matlab', 'vl_setupnn.m')) ;
gpuDevice(1);
net = load('imagenet-vgg-verydeep-16.mat') ; % load the pre-trained models
net.layers = net.layers(1:31); % the output of the last conv layer
net = vl_simplenn_move(net, 'gpu') ;
             
addpath(fullfile(pwd, 'evaluation'));
addpath(fullfile(pwd, 'setup'));  
addpath(fullfile(pwd, 'descrs'));
addpath(fullfile(pwd, 'encode'));


opts.dataset = 'voc07' ;
% opts.dataset = 'caltech256' ;
% opts.dataset = 'scene67' ;
% opts.dataset = 'sun397' ;
% opts.dataset = 'Holidays' ;
% opts.dataset = 'UKBench' ;
% opts.dataset = 'Oxford5k' ;

opts.prefix = 'lasc' ;  % lasc, fv, bovw, vlad, llc, sc, bcnn
opts.encoderParams = {...
    'type', opts.prefix, ... 
    'numWords',128,...
    'layouts', {'1x1', '2x2'}, ...
    'geometricExtension', 'none', ...
    'numPcaDimensions',256, ...
    'renormalize', true, ... 
    'extractorFn', @(x) getDenseCNNres(x, net, 'step', 4,'scales', [2/3, 1, 4/3])};
opts.scales = 3 ;
opts.maxImsize = 384 ;
opts.seed = 1 ;
opts.C = 10 ;
opts.lite = false ;
opts.kernel = 'linear' ;
opts.dataDir = 'data' ;
opts.datasetDir = fullfile(opts.dataDir, opts.dataset) ;
opts.resultDir = fullfile(opts.dataDir, [opts.dataset '-' opts.prefix]) ;
opts.imdbPath = fullfile(opts.resultDir, 'imdb.mat') ;
opts.encoderPath = fullfile(opts.resultDir, 'encoder.mat') ;
opts.cacheDir = fullfile(opts.resultDir, 'cache') ;
opts.pcaDir = fullfile(opts.resultDir, 'pca') ;
opts.pca = false ;
opts.pcaDim = 512 ;
opts.whitening = 0 ;
vl_xmkdir(opts.cacheDir) ;
disp('options:' ); disp(opts) ;

% --------------------------------------------------------------------
%                                                   Get image database
% --------------------------------------------------------------------
 switch opts.dataset
   case 'voc07', imdb = setupVoc(opts.datasetDir, 'lite', opts.lite, 'edition', '2007') ;
   case 'caltech256', imdb = setupCaltech256(opts.datasetDir, 'lite', opts.lite) ;
   case 'scene67', imdb = setupScene67(opts.datasetDir, 'lite', opts.lite) ;
   case 'sun397', imdb = setupSUN397(opts.datasetDir, 'lite', opts.lite, ...
                                             'variant', 'sun397', 'seed', opts.seed) ;
   case 'Holidays', imdb = setupHolidays(opts.datasetDir, 'seed', opts.seed) ;
   case 'UKBench', imdb = setupUKB(opts.datasetDir, 'seed', opts.seed) ;
   case 'Oxford5k', imdb = setupOxford5k(opts.datasetDir, 'seed', opts.seed) ;
   otherwise, error('Unknown dataset type.') ;
 end
 save(opts.imdbPath, '-struct', 'imdb') ;
% --------------------------------------------------------------------
%                                      Train encoder and encode images
% --------------------------------------------------------------------
if exist(opts.encoderPath, 'file')
  encoder = load(opts.encoderPath) ;
else
    switch opts.dataset
        case {'voc07','caltech256','scene67','sun397'}, numTrain = sum(imdb.images.set <= 2) ;
        case {'Holidays','UKBench','Oxford5k'},numTrain = sum(imdb.images.set == 3) ;
    end
  if opts.lite, numTrain = 10 ; end
  
  switch opts.dataset
        case {'voc07','caltech256','scene67','sun397'}, train_idx = vl_colsubset(find(imdb.images.set <= 2), numTrain, 'uniform')  ;
        case {'Holidays','UKBench','Oxford5k'},train_idx = vl_colsubset(find(imdb.images.set == 3), numTrain, 'uniform') ;
  end
  if ~strcmp(opts.prefix, 'bcnn')
      encoder = trainEncoder(fullfile(imdb.imageDir,imdb.images.name(train_idx)), ...
          opts.scales, opts.maxImsize, opts.encoderParams{:}, 'lite', opts.lite) ;
      save(opts.encoderPath, '-struct', 'encoder') ;
  end
end

switch opts.prefix
    case {'fv','bovw','vlad','lasc','llc','sc'}, opts.kernel = 'hell';
    case 'bcnn', opts.kernel = 'linear' ;
end

if strcmp(opts.prefix, 'bcnn')
    encoder.type = 'bcnn' ;
    encoder.readImageFn = @readImage ;
    encoder.extractorFn =  @(x) getDenseCNNres(x, net, 'step', 4,'scales', [2/3, 1, 4/3]) ;
    opts.pca = true ;
end
encodeImage(encoder, fullfile(imdb.imageDir, imdb.images.name),opts.kernel,'cacheDir', opts.cacheDir) ;         

dataDir = opts.cacheDir;


% --------------------------------------------------------------------
%                                        Perform PCA on image features
% --------------------------------------------------------------------
if opts.pca
dataDir = opts.pcaDir;
if strcmp(opts.prefix, 'bcnn')
    Dim = 512 * 512 ;
elseif strcmp(opts.prefix,'sc') ||strcmp(opts.prefix,'llc')
    Dim = opts.encoderParams{1, 4};
else
    Dim = encoder.numWords * opts.encoderParams{1, 10} * 2 ;
end

pcaDim = opts.pcaDim ;
whitening = opts.whitening ;
DimReduce(imdb, opts, Dim, pcaDim, whitening)
end

% --------------------------------------------------------------------
%                                            Train and evaluate models
% --------------------------------------------------------------------
if isfield(imdb.images, 'class')
  classRange = unique(imdb.images.class) ;
else
  classRange = 1:numel(imdb.classes.imageIds) ;
end

% train and test %
fprintf('traintest: testing...\n') ;
switch opts.dataset
    case { 'voc07','scene67','sun397','Holidays','UKBench', 'Oxford5k'}, iter_num = 1;
    case { 'caltech256'}, iter_num = 3;
end
accuracies = zeros(iter_num, 1);
maps = zeros(iter_num,1) ;
recalls = zeros(iter_num,1) ;
for iter = 1:iter_num
     switch opts.dataset
         case 'voc07', imdb = setupVoc(opts.datasetDir, 'lite', opts.lite, 'edition', '2007') ;                                   
         case 'caltech256', imdb = setupCaltech256(opts.datasetDir, 'lite', opts.lite) ;
         case 'scene67', imdb = setupScene67(opts.datasetDir, 'lite', opts.lite) ;
         case 'sun397', imdb = setupSUN397(opts.datasetDir, 'lite', opts.lite,'variant', 'sun397', 'seed', opts.seed) ;
         case 'Holidays', imdb = setupHolidays(opts.datasetDir, 'seed', opts.seed) ;
         case 'UKBench', imdb = setupUKB(opts.datasetDir, 'seed', opts.seed) ;
         case 'Oxford5k', imdb = setupOxford5k(opts.datasetDir, 'seed', opts.seed) ;
         otherwise, error('Unknown dataset type.') ;
     end
    switch opts.dataset
    case {'voc07','caltech256','scene67','sun397'}
        train_idx = find(imdb.images.set <= 2) ;
        test_idx = find(imdb.images.set == 3) ;
    end
    
    % evaluate %
    switch opts.dataset
        case {'caltech256','scene67','sun397'}
            accuracies(iter) = svm_vlfeat(imdb, opts, dataDir, classRange, train_idx, test_idx); 
        case 'voc07'
            maps(iter) = svm_vlfeat(imdb, opts, dataDir, classRange, train_idx, test_idx);
        case 'Holidays'
            maps(iter) = evaluateHolidays(imdb, dataDir) ;
        case  'UKBench'
            recalls(iter) = evaluateUKB(imdb, dataDir) ;           
        case 'Oxford5k' 
            maps(iter) = evaluateOxford5k(imdb, dataDir) ;
    end 
    switch opts.dataset
        case {'caltech256','scene67','sun397'}
            fprintf('traintest: Iteration %2d  accuracy: %5.4f\n', iter,  accuracies(iter));
        case 'voc07'
            fprintf('traintest: Iteration %2d  mAP: %5.4f\n', iter,  accuracies(iter));
    end
end
switch opts.dataset
    case {'caltech256','scene67','sun397'}
        fprintf('Average accuracy over %d times: mean+std %5.4f+%3.4f\n', iter_num, mean(accuracies), std(accuracies));  
    case {'voc07','Holidays', 'Oxford5k'}
        fprintf('Mean Average Precision over %d times: mean+std %5.4f+%3.4f\n', iter_num, mean(maps), std(maps));
    case 'UKBench'
        fprintf('Average top-4 recall of each object over %d times: mean+std %5.4f+%3.4f\n', iter_num, mean(recalls), std(recalls));
end

