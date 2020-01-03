function imdb = setupSUN397(datasetDir, varargin)

opts.numTrain = 50 ;
opts.numTest = 50 ;

imdb.meta.sets = {'train', 'val', 'test'} ;
names = dir(datasetDir) ;
names = {names([names.isdir]).name} ;
names = setdiff(names, {'.', '..'}) ;
imdb.meta.classes = names ;

load('sun_split10.mat');
split = split{1};
names = {} ;
classes = {} ; 
for c = 1:numel(imdb.meta.classes)
  split{c}.ClassName(split{c}.ClassName == '/') = '_';
  class = ['sun397' split{c}.ClassName];
  names_train = strcat([class filesep], split{c}.Training(1:opts.numTrain)) ;
  sets_train = ones(1,numel(names_train));
  names_test = strcat([class filesep], split{c}.Testing(1:opts.numTest)) ;
  sets_test = 3*ones(1,numel(names_test));
  
  imdb.meta.classes{c} = class;
  names{c} = [names_train  names_test];
  sets{c} = [sets_train  sets_test];
  classes{c} = repmat(c, 1, numel(names{c})) ;
end

imdb.images.name = cat(2,names{:}) ;
imdb.images.class = cat(2,classes{:}) ;
imdb.images.set = cat(2,sets{:}) ;
imdb.imageDir = datasetDir ;


