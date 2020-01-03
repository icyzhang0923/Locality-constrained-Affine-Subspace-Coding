function DimReduce(imdb, opts, dim, dim_pca, whitening)

% load training data
opts.cacheChunkSize = 512 ;
numChunks = ceil(numel(imdb.images.name) / opts.cacheChunkSize) ;
train_data = cell(1,sum(imdb.images.set <= 2)) ; 
num = ceil(1e4/numChunks);
for c = 1:numChunks
  chunkPath = fullfile(opts.cacheDir, sprintf('chunk-%03d.mat',c)) ;
  load(chunkPath, 'data') ;
  lo = opts.cacheChunkSize * (c-1) + 1;
  hi = min(opts.cacheChunkSize * c,numel(imdb.images.name));
  chunk_split = imdb.images.set(lo:hi);
  data = data(:,chunk_split <= 2);
  sel = vl_colsubset(1:size(data,2), min(size(data,2),single(num))) ;
  train_data{c} = data(:, sel);
  clear data ;
end
train_data = cat(2,train_data{:}) ;

num_block = size(train_data,1)/dim;
pcaData.mu = cell(num_block, 1);
pcaData.proj = cell(num_block, 1);
dim_pca = min(dim_pca, size(train_data,2)-1);
for b = 1:num_block
    tic;
    lo = (b-1)*dim +1;
    hi = b*dim;
    this_block = train_data(lo:hi, :);
    pcaData.mu{b} = mean(this_block,2);
    this_block =  bsxfun(@minus, this_block, pcaData.mu{b});    

    [V, ~, D] = princomp(this_block', 'econ');
    if ~whitening
        pcaData.proj{b} = V(:,1:dim_pca)';
    else
        pcaData.proj{b} = diag(1./sqrt(D(1:dim_pca) + 1e-5)) * V(:,1:dim_pca)' ;
    end  
    t = toc;
    fprintf('%s: pca of block %d trained, elapsing %.2f minutes.\n', mfilename, b, t/60) ;
end
clear train_data

for c = 1:numChunks
   chunkPath = fullfile(opts.cacheDir, sprintf('chunk-%03d.mat',c)) ;
   load(chunkPath, 'data') ;
   data_ori = data;
   
   data = cell(num_block, 1);
    for b = 1:num_block  
        lo = (b-1)*dim +1;
        hi = b*dim;
        data{b} = pcaData.proj{b} * bsxfun(@minus, data_ori(lo:hi,:), pcaData.mu{b});
    end
    clear data_ori
    data = cat(1, data{:});
    newPath = fullfile(opts.pcaDir, sprintf('chunk-%03d.mat',c)) ;
    if ~exist(opts.pcaDir, 'dir')
        mkdir(opts.pcaDir);
    end
    save(newPath, 'data', '-v7.3');

end


