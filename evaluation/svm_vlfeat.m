function acc = svm_vlfeat(imdb, opts, dataDir, classRange, train_idx, test_idx)

iter_max = 100;
switch opts.dataset
    case 'voc07', iter_max = 30;
end

switch opts.dataset
case 'voc07'
    %% Generate training data and testing data                 
    opts.cacheChunkSize = 512 ;
    numChunks = ceil(numel(imdb.images.name) / opts.cacheChunkSize) ;
    train_data = cell(1,sum(imdb.images.set <= 2)) ; 
    test_data = cell(1,sum(imdb.images.set == 3));
    for c = 1:numChunks
      chunkPath = fullfile(dataDir, sprintf('chunk-%03d.mat',c)) ;
      fprintf('%s: loading descriptors from %s\n', mfilename, chunkPath) ;
      load(chunkPath, 'data') ;
      lo = opts.cacheChunkSize * (c-1) + 1;
      hi = min(opts.cacheChunkSize * c,numel(imdb.images.name));
      chunk_split = imdb.images.set(lo:hi);
      train_data{c} = data(:,chunk_split <= 2);
      test_data{c} = data(:,chunk_split == 3);  
      clear data ;
    end
    train_data = cat(2,train_data{:}) ;
    test_data = cat(2,test_data{:}) ;    

    %% train and test
    lambda = 1 / (opts.C*numel(train_idx)) ;
    par = {'Solver', 'sdca', 'Verbose', ...
           'BiasMultiplier', 1, ...
           'Epsilon', 0.001, ...
           'MaxNumIterations', iter_max * numel(train_idx)} ;

    scores = cell(1, numel(classRange)) ;
    ap = zeros(1, numel(classRange)) ;
    ap11 = zeros(1, numel(classRange)) ;
    w = cell(1, numel(classRange)) ;
    b = cell(1, numel(classRange)) ;
    for c = 1:numel(classRange)
      if isfield(imdb.images, 'class')
        y = 2 * (imdb.images.class == classRange(c)) - 1 ;
      else
        y = - ones(1, numel(imdb.images.id)) ;
        [~,loc] = ismember(imdb.classes.imageIds{classRange(c)}, imdb.images.id) ;
        y(loc) = 1 - imdb.classes.difficult{classRange(c)} ;
      end
      if all(y <= 0), continue ; end

      [w{c},b{c}] = vl_svmtrain(train_data, y(train_idx), lambda, par{:}) ;
      scores{c} = w{c}' * test_data + b{c} ;

      [~,~,info] = vl_pr(y(test_idx), scores{c}) ;
      ap(c) = info.ap ;
      ap11(c) = info.ap_interp_11 ;
      fprintf('class %d %s AP %.2f; AP 11 %.2f\n', c, imdb.meta.classes{classRange(c)}, ...
              ap(c) * 100, ap11(c)*100) ;
    end

    mAP = sprintf('mAP: %.2f %%; mAP 11: %.2f', mean(ap) * 100, mean(ap11) * 100) ;
    disp(mAP) ;
    acc = mean(ap11) * 100; 

case {'caltech256','scene67','sun397'}             
    %% Generate training data                
    opts.cacheChunkSize = 512 ;
    numChunks = ceil(numel(imdb.images.name) / opts.cacheChunkSize) ;
    train_data = cell(1,sum(imdb.images.set <= 2)) ; 
    for c = 1:numChunks
      chunkPath = fullfile(dataDir, sprintf('chunk-%03d.mat',c)) ;
      fprintf('%s: loading descriptors from %s\n', mfilename, chunkPath) ;
      load(chunkPath, 'data') ;
      lo = opts.cacheChunkSize * (c-1) + 1;
      hi = min(opts.cacheChunkSize * c,numel(imdb.images.name));
      chunk_split = imdb.images.set(lo:hi);
      train_data{c} = data(:,chunk_split <= 2);
      clear data ;
    end
    train_data = cat(2,train_data{:}) ; 

    %% train SVM model
    lambda = 1 / (opts.C*numel(train_idx)) ;
    par = {'Solver','sdca', 'Verbose', ...
           'BiasMultiplier', 1, ...
           'Epsilon', 0.001, ...
           'MaxNumIterations', iter_max * numel(train_idx)} ;

    scores = cell(1, numel(classRange)) ; 
    w = cell(1, numel(classRange)) ; 
    b = cell(1, numel(classRange)) ;
    for c = 1:numel(classRange)
      if isfield(imdb.images, 'class')
        y = 2 * (imdb.images.class == classRange(c)) - 1 ;
      else
        y = - ones(1, numel(imdb.images.id)) ;
        [~,loc] = ismember(imdb.classes.imageIds{classRange(c)},imdb.images.id) ; 
        y(loc) = 1 - imdb.classes.difficult{classRange(c)} ;
      end
      if all(y <= 0)
          continue ;
      end

      [w{c},b{c}] = vl_svmtrain(train_data, y(train_idx), lambda, par{:}) ; 
      fprintf('class %d %s svm trained...\n', c, imdb.meta.classes{classRange(c)}) ;
    end            
    clear train_data

    %% Generate testing data                 
    test_data = cell(1,sum(imdb.images.set == 3));
    for c = 1:numChunks
      chunkPath = fullfile(dataDir, sprintf('chunk-%03d.mat',c)) ;
      fprintf('%s: loading descriptors from %s\n', mfilename, chunkPath) ;
      load(chunkPath, 'data') ;
      lo = opts.cacheChunkSize * (c-1) + 1;
      hi = min(opts.cacheChunkSize * c,numel(imdb.images.name));
      chunk_split = imdb.images.set(lo:hi);
      test_data{c} = data(:,chunk_split == 3);  
      clear data ;
    end
    test_data = cat(2,test_data{:}) ;     

   %% testing and calculate accuracy     
   for c = 1:numel(classRange)
      scores{c} = w{c}' * test_data + b{c} ;
      fprintf('class %d %s completed...\n', c, imdb.meta.classes{classRange(c)}) ;
   end
    scores = cat(1,scores{:}) ;
    [~,preds] = max(scores, [], 1) ;

    test_label = imdb.images.class(test_idx);            
    acc = sum(preds == test_label) / length(test_idx) * 100;
end
