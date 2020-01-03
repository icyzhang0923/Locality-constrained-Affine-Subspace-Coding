function imdb = setupOxford5k(datasetDir, varargin)

fnames = dir(fullfile(datasetDir, '*.jpg'));
imdb.images.name = {fnames.name} ;
imdb.images.id = 1:numel(imdb.images.name);
imdb.images.class = zeros(1, numel(imdb.images.name));  % undefined for now

gtDir = fullfile('setup', 'gt_files');
landmarks = {'all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket',...
             'hertford', 'keble', 'magdalen', 'pitt_rivers','radcliffe_camera'};
query = cell(1, numel(landmarks) * 5);
class = cell(1, numel(landmarks) * 5);
for i = 1:numel(landmarks)
    for j = 1:5
        class{5*(i-1)+j} = sprintf('%s_%d', landmarks{i}, j);
        name = textread(fullfile(gtDir, sprintf('%s_%d_query.txt', landmarks{i}, j)), '%s');
        query{5*(i-1)+j} = [name{1}(6:end), '.jpg'];
    end
end

imdb.images.set = 3 * ones(1, numel(imdb.images.name));
for i = 1:numel(query)
    ind = strcmp(imdb.images.name, query{i});
    imdb.images.set(ind) = 1;
end

imdb.meta.sets = {'query', 'null', 'retrieval'} ;
imdb.meta.classes = class ;
imdb.meta.landmarks = landmarks;
imdb.imageDir = datasetDir;

