function mAP= evaluateHolidays(imdb, dataDir)

ChunkSize = 512 ;
numChunks = ceil(numel(imdb.images.name) / ChunkSize) ;
query_data = cell(1,sum(imdb.images.set <= 2)) ; 
test_data = cell(1,sum(imdb.images.set == 3));
for c = 1:numChunks
  chunkPath = fullfile(dataDir, sprintf('chunk-%03d.mat',c)) ;
  fprintf('%s: loading descriptors from %s\n', mfilename, chunkPath) ;
  load(chunkPath, 'data') ;
  lo = ChunkSize * (c-1) + 1;
  hi = min(ChunkSize * c,numel(imdb.images.name));
  chunk_split = imdb.images.set(lo:hi);
  query_data{c} = data(:,chunk_split <= 2);
  test_data{c} = data(:,chunk_split == 3);  
  clear data ;
end
query_data = cat(2,query_data{:})' ;
test_data = cat(2,test_data{:})' ;  
query_idx = find(imdb.images.set <= 2) ;
test_idx = find(imdb.images.set == 3 ); 
test_class = imdb.images.class(test_idx);
dist = sp_dist2(query_data, test_data);
AP = zeros(length(query_idx), 1);

for i = 1:length(query_idx)
     idx = find(test_class == i);
    % compute ap %
    labels = -ones(1, numel(test_idx));
    labels(idx) = 1;    
    [~,~,info] = vl_pr(labels, 1./dist(i,:)) ;
    AP(i) = info.ap ;

end
mAP = mean(AP);
fprintf('\n%s: mAP is %5.4f\n', mfilename, mAP) ;

