function z = lasc_encode(descrs, encoder)

soft_num = 5;

%%% proximity measure (Euclidean distance, point to center)  
dist_matrix = sp_dist2(descrs', encoder.words');
[dist2,min_ind] = sort(dist_matrix,2, 'ascend');
soft_idx = min_ind(:,1:soft_num);
dist2 = dist2(:,1:soft_num);

% %% for retrieval task 
% lambda = 10;
% weight = exp(-lambda * dist2);
%%% for classification task 
lambda = 5;
weight = ones(size(dist2))./(1+exp(lambda*dist2)); 

weight = bsxfun(@rdivide, weight, sum(weight,2));
weight(isnan(weight)) = 0;

% %% proximity measure (Mahalanobis distance, rebuilt error)
% dist_matrix = zeros(size(descrs, 2),encoder.numWords);
% for i = 1:encoder.numWords 
%     err_matrix = encoder.pca.err{i} * bsxfun(@minus, descrs, encoder.words(:,i));
%     dist_matrix(:,i) = sum((err_matrix).^2);
% end
% [dist2, min_ind] = sort(dist_matrix,2, 'ascend');
% soft_idx = min_ind(:,1:soft_num);
% dist2 = dist2(:,1:soft_num);
% lambda = 1;
% weight = exp(-lambda * dist2);
% % weight = ones(size(dist2))./(1+exp(lambda*dist2));   
% weight = bsxfun(@rdivide, weight, sum(weight,2));
% weight(isnan(weight)) = 0;

% %% proximity measure (Probabilistic Model, posterior probability)
% log_u = zeros(size(descrs, 2),encoder.numWords);
% for i = 1:encoder.numWords 
%     mu_sigma2 = (encoder.pca.proj{i} * bsxfun(@minus, descrs, encoder.words(:,i))).^2; 
%     log_u(:,i) = -0.5 * sum(bsxfun(@plus, mu_sigma2, encoder.pca.log_coef{i}));
% end
% log_lambda = log_sum(log_u);
% weight = exp(bsxfun(@minus, log_u, log_lambda));
% [weight, min_ind] = sort(weight,2, 'descend');
% soft_idx = min_ind(:,1:soft_num);
% weight = weight(:,1:soft_num);
% weight = bsxfun(@rdivide, weight, sum(weight,2));
% weight(isnan(weight)) = 0;


%%% coding features and pooling
z = cell(1, encoder.numWords*2);
for i = 1:encoder.numWords       
    curr_idx = find(soft_idx == i);
    curr_weight = weight(curr_idx);
    curr_idx = mod(curr_idx,size(descrs, 2));
    curr_idx (curr_idx == 0) = size(descrs, 2);
    curr_num = length(curr_idx);
    if curr_num > 0
       %%% for ridge regression %%
        beta = encoder.pca.proj{i} * bsxfun(@minus, descrs(:,curr_idx), encoder.words(:,i));
        weight_soft = repmat(curr_weight,1,encoder.pca.pcaNum(i))'; 
        z{i} = mean(beta.* weight_soft, 2);   
        z{i} = z{i}./(norm(z{i})+eps);
        z{i+encoder.numWords} = mean((beta.^2 -1).* weight_soft, 2);
        z{i+encoder.numWords} = z{i+encoder.numWords}./(norm(z{i+encoder.numWords})+eps);       
             
%        %% for elastic net %%
%         weight_soft = repmat(curr_weight,1,encoder.pca.pcaNum(i))'; 
%         param.lambda  = 0.05; 
%         param.lambda2 = 0.0;
%         param.mode = 2;   
%         param.pos = false;
%         sift_set = bsxfun(@minus, descrs(:,curr_idx), encoder.words(:,i));  
%         sift_set = bsxfun(@times, sift_set, 1./max(sqrt(sum(sift_set.^2)), eps(4))) ;
%         beta = mexLasso(sift_set, encoder.pca.proj{i}', param);
%         beta = full(beta);
%         z{i} = mean(beta.* weight_soft, 2);
%         z{i} = z{i}./(norm(z{i})+eps); 
%        
%         mu = encoder.pca.mu(:,i);
%         b  = encoder.pca.b(:,i);
%         beta2 = bsxfun(@times, abs(bsxfun(@minus, beta, mu)), 1./b) - 1;
%         z{i+encoder.numWords} = mean(beta2.* weight_soft, 2);
%         z{i+encoder.numWords} = z{i+encoder.numWords}./(norm(z{i+encoder.numWords})+eps); 
        
    else
        z{i} = zeros(encoder.pca.pcaNum(i), 1);
        z{i+encoder.numWords} = z{i};
    end
end
z1 = cat(1, z{1:encoder.numWords});
z1 = z1./(norm(z1)+eps);
z2 = cat(1, z{encoder.numWords+1:end});
z2 = z2./(norm(z2)+eps);
gama = 0.8; % weight of first-order LASC and second-order LASC, 1 for retrieval task.
z = [gama*z1;(1-gama)*z2];
end


function log_lambda = log_sum(log_u_matrix)
    log_lambda = zeros(size(log_u_matrix,1),1);
    for n = 1:size(log_u_matrix,1)
        log_u = log_u_matrix(n,:);    
        log_lambda(n) = log_u(1);
        for d = 1:length(log_u)-1
            log_lambda(n) = log_lambda(n) + log(1+exp((log_u(d+1)-log_lambda(n))));
        end
    end
end
