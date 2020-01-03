function mAP = evaluateUKB(imdb, dataDir)

ChunkSize = 512 ;
numChunks = ceil(numel(imdb.images.name) / ChunkSize) ;
all_data = cell(1, numChunks);
for c = 1:numChunks
  chunkPath = fullfile(dataDir, sprintf('chunk-%03d.mat',c)) ;
  fprintf('%s: loading descriptors from %s\n', mfilename, chunkPath) ;
  load(chunkPath, 'data') ;
  all_data{c} = data';
  clear data ;
end
all_data = cat(1,all_data{:});
dist = sp_dist2(all_data, all_data);

returned = zeros(numel(imdb.images.name), 1);
for i = 1:numel(imdb.images.name)
    [~, index] = sort(dist(i,:),'ascend');
    index = index(1:4);
    returned(i) = sum(imdb.images.class(index) == imdb.images.class(i));
end

mAP = mean(returned);
fprintf('%s: mAP is %5.4f\n', mfilename, mAP) ;

