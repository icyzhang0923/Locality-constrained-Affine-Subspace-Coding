function mAP = evaluateOxford5k(imdb, dataDir)

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
% showSamples(imdb.images.name, query_idx);

AP = zeros(length(query_idx), 1);
for i = 1:numel(imdb.meta.classes)
    
    good_idx = findIDX(imdb.images.name, imdb.meta.classes{i}, 'good');
    ok_idx   = findIDX(imdb.images.name, imdb.meta.classes{i}, 'ok');
    junk_idx = findIDX(imdb.images.name, imdb.meta.classes{i}, 'junk');
%     showSamples(imdb.images.name, good_idx);

    % compute ap %
    query = data_all(query_idx(i), :);
    dist = sp_dist2(query, data_all);
    [~, idx] = sort(dist, 'ascend');
    
%     figure; 
%     set (gcf,'Position',[400,500,800,400])
%     im = imread(fullfile('data', 'Oxford5k',imdb.images.name{query_idx(i)}));
%     subplot(1,2,1);
%     imshow(im);
%     name = imdb.images.name{query_idx(i)};
%     name(name=='_') = ' ';
%     title(sprintf('%s(%d)',name, length([good_idx; ok_idx])));
    
    AP(i) = compute_ap([good_idx; ok_idx], junk_idx, idx); 
%     saveas(gcf, fullfile('results', name), 'jpg');
    
    % save results %
%     labels = -ones(1, numel(imdb.images.name));
%     labels([query_idx(i);junk_idx]) = 0;
%     labels([good_idx;ok_idx]) = 1;
%     showResults(imdb.images.name, labels, idx(1:11));

    % my compute ap %
%     labels = -ones(1, numel(imdb.images.name));
%     labels([query_idx(i);junk_idx]) = 0;
%     labels([good_idx;ok_idx]) = 1;
% 
%     query = data_all(query_idx(i), :);
%     retrieval = data_all(labels~=0, :);
%     labels = labels(labels ~= 0);
%     dist = sp_dist2(query, retrieval);
%     
%     [~,~,info] = vl_pr(labels, 1./dist) ;
%     AP(i) = info.ap ;
    
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
% subplot(1,2,2);
% plot(x, y, 'b.-');
% grid on;
% title(sprintf('ap = %.4f', ap));
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

%% show samples %%
function showSamples(names, idx)
    figure;
    num = min(length(idx), 15);
    SizeX = 150;
    SizeY = 150;
    line = 3;
    col = ceil(num/line);

    all_ims = zeros(line*SizeX+(line)*4, col*SizeY+(col-1)*4, 3); 
    all_ims = uint8(all_ims);
    all_ims(:,:,:) = 255;
    for i =1:num
        I = imread(fullfile('data', 'Oxford5k', names{idx(i)}));
        I_s = imresize(I, [SizeX SizeY] ,'bicubic');

        j = ceil(i/line);
        ids = (j-1)*SizeY+(j-1)*4+1;
        ide = (j-1)*SizeY+(j-1)*4+SizeY;

        z = mod(i,line)-1;
        if z == -1
           z = line-1;
        end
        all_ims(z*SizeX+z*4+1:z*SizeX+z*4+SizeX,ids:ide,:) = I_s;
    end
    imshow(all_ims);
end

%% show results %%
function showResults(names, labels, idx)
%     figure;
    num = min(length(idx), 11);
    SizeX = 150;
    SizeY = 150;
    line = 1;
    col = ceil(num/line);

    all_ims = zeros(line*SizeX+(line)*4, col*SizeY+(col-1)*4, 3); 
    all_ims = uint8(all_ims);
    all_ims(:,:,:) = 255;
    for i =1:num
        I = imread(fullfile('data', 'Oxford5k', names{idx(i)}));
        I_s = imresize(I, [SizeX SizeY] ,'bicubic');
        
        color_box_w = zeros(4,150,3);
        color_box_h = zeros(150,4,3);
        if i == 1
        elseif labels(idx(i)) == 1
            color_box_w(:,:,2) = 255;
            color_box_h(:,:,2) = 255;
            I_s(1:4, :, :) = color_box_w;
            I_s(end-3:end, :, :) = color_box_w;
            I_s(:, 1:4, :) = color_box_h;
            I_s(:, end-3:end, :) = color_box_h;           
        elseif labels(idx(i)) == -1
            color_box_w(:,:,1) = 255;
            color_box_h(:,:,1) = 255;
            I_s(1:4, :, :) = color_box_w;
            I_s(end-3:end, :, :) = color_box_w;
            I_s(:, 1:4, :) = color_box_h;
            I_s(:, end-3:end, :) = color_box_h;                 
        else
            color_box_w(:,:,3) = 255;
            color_box_h(:,:,3) = 255;
            I_s(1:4, :, :) = color_box_w;
            I_s(end-3:end, :, :) = color_box_w;
            I_s(:, 1:4, :) = color_box_h;
            I_s(:, end-3:end, :) = color_box_h;    
        end
        
        j = ceil(i/line);
        ids = (j-1)*SizeY+(j-1)*4+1;
        ide = (j-1)*SizeY+(j-1)*4+SizeY;

        z = mod(i,line)-1;
        if z == -1
           z = line-1;
        end
        all_ims(z*SizeX+z*4+1:z*SizeX+z*4+SizeX,ids:ide,:) = I_s;
    end
%     imshow(all_ims);
    imwrite(all_ims, fullfile('results',names{idx(1)}), 'jpg');
end



