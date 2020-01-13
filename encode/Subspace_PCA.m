function dictionary_pca = Subspace_PCA(dictionary, num_pca, dictionarySize, descrs, index)

dictionary_pca.proj = cell(dictionarySize,1);
dictionary_pca.err= cell(dictionarySize,1);
dictionary_pca.log_coef = cell(dictionarySize,1);
dictionary_pca.pcaNum = zeros(dictionarySize, 1);
dictionary_pca.mu = zeros(num_pca, dictionarySize);
dictionary_pca.b  = zeros(num_pca, dictionarySize);

x = zeros(dictionarySize,1);
for i = 1:dictionarySize
    curr_idx = (index ==i);
    sift_sets = single(descrs(:, curr_idx));  
    x(i) = size(sift_sets,2);
    sift_sets = bsxfun(@minus, sift_sets, dictionary(:,i));  
    
    %%% for ridge regression %%   
    cov_matrix = cov(sift_sets') + 1e-3 * eye(size(sift_sets,1));    
    [vectors,values] = eig(cov_matrix);  
    [max_value,max_idx] = sort(diag(values),'descend');
    vectors = vectors(:,max_idx);
    values = diag(max_value);   
    
    max_vectors = vectors(:, 1:num_pca); 
    max_values = values(1:num_pca, 1:num_pca);    
    dictionary_pca.pcaNum(i) = num_pca;    
    
    dictionary_pca.proj{i} = max_values^(-1/2) * max_vectors';
    dictionary_pca.log_coef{i} = log(2*pi)*ones(num_pca,1)+log(max_value(1:num_pca));
    dictionary_pca.err{i} = eye(size(descrs,1)) - max_vectors * max_vectors'; 
     
%     %%  for elastic net  %%
%      sift_sets = bsxfun(@times, sift_sets, 1./max(sqrt(sum(sift_sets.^2)), eps(4))) ;
%      param.lambda  = 0.05;
%      param.lambda2 = 0.0;
%      param.mode = 2;
% 
%      param.modeParam = 0;
%      param.gamma1 = 0.1;
%      param.modeD = 1;
%      param.K = num_pca;   
%      param.numThreads = 1;
%      param.batchsize = 64;
%      param.verbose = false;
%      param.iter = 100;       
%      D = mexTrainDL(sift_sets, param);
% 
%      param.pos = false;
%      beta = mexLasso(sift_sets, D, param);
%      dictionary_pca.mu(:,i) = median(beta, 2);
%      dictionary_pca.b(:,i) = mean(abs(bsxfun(@minus, beta, dictionary_pca.mu(:,i))), 2);
%      dictionary_pca.proj{i} = D';
%      dictionary_pca.pcaNum(i) = num_pca;    

end    

