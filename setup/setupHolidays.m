function imdb = setupHolidays(datasetDir, varargin)

opts.lite = false ;
opts.seed = 1 ;
opts.numTrain = 0 ;
opts.numTest = 2000 ;
opts.autoDownload = true ;
opts = vl_argparse(opts, varargin) ;

imdb = setupGeneric(fullfile(datasetDir), ...
  'numTrain', opts.numTrain, 'numVal', 0, 'numTest', opts.numTest,  ...
  'expectedNumClasses', 1, ...
  'seed', opts.seed, 'lite', opts.lite) ;

names = cat(1,imdb.images.name{:});
names = names(:,5:end);
class = double(names(:,2:4)-48);
class = class(:,1)*100+class(:,2)*10+class(:,3);
for i = 0:499
    idx = find(class==i);
    imdb.images.class(idx) = i+1;
    imdb.images.set(idx(1)) = 1;
end
