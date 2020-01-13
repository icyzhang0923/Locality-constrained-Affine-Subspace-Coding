% --------------------------------------------------------------------
%              Parameter settings
% --------------------------------------------------------------------
clear
S = RandStream('mt19937ar','seed',1024);
RandStream.setGlobalStream(S);

run(fullfile(fileparts(which(mfilename)), 'VLFeat', 'toolbox', 'vl_setup.m')) ;
run(fullfile(fileparts(which(mfilename)), 'matconvnet', 'matlab', 'vl_setupnn.m')) ;
gpuDevice(1);
net = load('imagenet-vgg-verydeep-16.mat') ; % load the pre-trained models
net.layers = net.layers(1:31); % the output of pool5 layer
net = vl_simplenn_move(net, 'gpu') ;
             
addpath(fullfile(pwd, 'evaluation'));
addpath(fullfile(pwd, 'setup'));  
addpath(fullfile(pwd, 'descrs'));
addpath(fullfile(pwd, 'encode'));
addpath(fullfile(pwd, 'build'));


opts.dataset = 'voc07' ;
% opts.dataset = 'caltech256' ;
% opts.dataset = 'scene67' ;
% opts.dataset = 'sun397' ;

opts.prefix = 'lasc' ;  % lasc, fv
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
opts.kernel = 'linear' ; % 'hell' for retrieval task
opts.dataDir = 'data' ; 
opts.datasetDir = fullfile(opts.dataDir, opts.dataset) ;
opts.resultDir = fullfile(opts.dataDir, [opts.dataset '-' opts.prefix]) ;
opts.imdbPath = fullfile(opts.resultDir, 'imdb.mat') ;
opts.encoderPath = fullfile(opts.resultDir, 'encoder.mat') ;
opts.cacheDir = fullfile(opts.resultDir, 'cache') ;
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
   otherwise, error('Unknown dataset type.') ;
 end
 save(opts.imdbPath, '-struct', 'imdb') ;
% --------------------------------------------------------------------
%                                      Train encoder and encode images
% --------------------------------------------------------------------
if exist(opts.encoderPath, 'file')
  encoder = load(opts.encoderPath) ;
else
    numTrain = sum(imdb.images.set <= 2) ;
    train_idx = vl_colsubset(find(imdb.images.set <= 2), numTrain, 'uniform') ;
    if opts.lite, numTrain = 10 ; end
      encoder = trainEncoder(fullfile(imdb.imageDir,imdb.images.name(train_idx)), ...
          opts.scales, opts.maxImsize, opts.encoderParams{:}, 'lite', opts.lite) ;
      save(opts.encoderPath, '-struct', 'encoder') ;
end

encodeImage(encoder, fullfile(imdb.imageDir, imdb.images.name),opts.kernel,'cacheDir', opts.cacheDir) ;         
dataDir = opts.cacheDir;

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
    case { 'voc07','scene67','sun397'}, iter_num = 1;
    case { 'caltech256'}, iter_num = 3;
end
train_idx = find(imdb.images.set <= 2) ;
test_idx = find(imdb.images.set == 3) ;
accuracies = zeros(iter_num, 1);
maps = zeros(iter_num,1) ;
for iter = 1:iter_num
     switch opts.dataset
         case 'voc07', imdb = setupVoc(opts.datasetDir, 'lite', opts.lite, 'edition', '2007') ;                                   
         case 'caltech256', imdb = setupCaltech256(opts.datasetDir, 'lite', opts.lite) ;
         case 'scene67', imdb = setupScene67(opts.datasetDir, 'lite', opts.lite) ;
         case 'sun397', imdb = setupSUN397(opts.datasetDir, 'lite', opts.lite,'variant', 'sun397', 'seed', opts.seed) ;
         otherwise, error('Unknown dataset type.') ;
     end    
    % evaluate %
    switch opts.dataset
        case {'caltech256','scene67','sun397'}
            accuracies(iter) = svm_vlfeat(imdb, opts, dataDir, classRange, train_idx, test_idx); 
        case 'voc07'
            maps(iter) = svm_vlfeat(imdb, opts, dataDir, classRange, train_idx, test_idx);
    end 
    switch opts.dataset
        case {'caltech256','scene67','sun397'}
            fprintf('traintest: Iteration %2d  accuracy: %5.4f\n', iter,  accuracies(iter));
        case 'voc07'
            fprintf('traintest: Iteration %2d  mAP: %5.4f\n', iter,  maps(iter));
    end
end
if strcmp(opts.dataset, 'voc07')
  fprintf('Average accuracy over %d times: mean+std %5.4f+%3.4f\n', iter_num, mean(maps), std(maps)); 
else
  fprintf('Average accuracy over %d times: mean+std %5.4f+%3.4f\n', iter_num, mean(accuracies), std(accuracies)); 
end
