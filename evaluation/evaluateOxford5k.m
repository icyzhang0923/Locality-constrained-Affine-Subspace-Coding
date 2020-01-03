function mAP=evaluateOxford5k(imdb, dataDir)

ChunkSize = 512 ;
numChunks = ceil(numel(imdb.images.name) / ChunkSize) ;
data_all = cell(1, numChunks);
for c = 1:numChunks
  chunkPath = fullfile(dataDir, sprintf('chunk-%03d.mat',c)) ;
  fprintf('%s: loading descriptors from %s\n', mfilename, chunkPath) ;
  load(chunkPath, 'data') ;
  data_all{c} = data;
  clear data ;
end
data_all = cat(2,data_all{:})' ;

gtDir = fullfile('setup', 'gt_files');
query_idx = zeros(1, numel(imdb.meta.classes));
for i = 1:numel(imdb.meta.classes)
    name = textread(fullfile(gtDir, [imdb.meta.classes{i}, '_query.txt']), '%s');
    query_idx(i) = find(strcmp(imdb.images.name, [name{1}(6:end) '.jpg']));
end

AP = zeros(length(query_idx), 1);
for i = 1:numel(imdb.meta.classes)
    
    good_idx = findIDX(imdb.images.name, imdb.meta.classes{i}, 'good');
    ok_idx   = findIDX(imdb.images.name, imdb.meta.classes{i}, 'ok');
    junk_idx = findIDX(imdb.images.name, imdb.meta.classes{i}, 'junk');

    % compute ap %
    query = data_all(query_idx(i), :);
    dist = sp_dist2(query, data_all);
    [~, idx] = sort(dist, 'ascend');
    
    
    AP(i) = compute_ap([good_idx; ok_idx], junk_idx, idx);     
end

mAP = zeros(numel(imdb.meta.landmarks), 1);
for i = 1:numel(imdb.meta.landmarks)
    mAP(i) = mean(AP( 5*(i-1)+1 : 5*i));
    fprintf('%s: %s mAP is %5.4f\n', mfilename, imdb.meta.landmarks{i}, mAP(i)) ;
end

mAP = mean(mAP);
fprintf('\n%s: mAP is %5.4f\n', mfilename, mAP) ;
end

%% compute ap %%
function ap = compute_ap(pos_set, junk_set, ranked_list)
old_recall = 0; old_precision = 1; ap = 0;
right = 0; j = 0;
x = 0;
y = 1;
for i = 1:numel(ranked_list)
    if  any(junk_set == ranked_list(i))
        continue;
    end
    if any(pos_set == ranked_list(i)) 
        right = right + 1;
    end
    recall = right/numel(pos_set); x = [x, recall];
    precision = right/(j+1); y = [y, precision];
    ap = ap + (recall - old_recall)*((old_precision + precision)/2);
    old_recall = recall;
    old_precision = precision;
    j = j + 1;
end
end

%% find sample idx %%
function type_idx = findIDX(names, query, type_name)
    gtDir = fullfile('setup', 'gt_files');
    type = textread(fullfile(gtDir, sprintf('%s_%s.txt', query, type_name)), '%s');
    type_idx = zeros(numel(type), 1);
    for j = 1:numel(type)
        this_name = sprintf('%s.jpg', type{j});
        type_idx(j) = find(strcmp(names, this_name));
    end
end




