function imdb = setupUKB(datasetDir, varargin)

fnames = dir(fullfile(datasetDir, '*.jpg'));
imdb.images.name = {fnames.name} ;
imdb.images.id = 1:numel(imdb.images.name);

num_class = numel(imdb.images.name)/4;
class = 1:num_class;
imdb.images.class = repmat(class, 4, 1);
imdb.images.class = imdb.images.class(:)';

imdb.images.set = 3 * ones(1, numel(imdb.images.name));
idx = (class -1) * 4 + 1; 
imdb.images.set(idx) = 1;

imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = class ;
imdb.imageDir = datasetDir;