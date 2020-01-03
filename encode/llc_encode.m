function llc_codes = llc_encode(descrs, encoder, knn, beta)

llc_codes = zeros(encoder.numWords, size(descrs,2), 'single');
dist_matrix = sp_dist2(descrs', encoder.words');
[~,min_ind] = sort(dist_matrix,2, 'ascend');
soft_idx = min_ind(:,1:knn);
for i = 1:size(descrs,2)
   idx = soft_idx(i,:);
   z = encoder.words(:,idx)' - repmat(descrs(:,i)', knn, 1); % shift ith pt to origin
   C = z * z';                               % local covariance
   C = C + eye(knn) * beta * trace(C);       % regularlization (K>D)
%    C = C + eye(knn) * beta ;       % regularlization (K>D)
   w = C \ ones(knn,1);
   w = w / sum(w);                          % enforce sum(w)=1
   llc_codes(idx,i) = w;
end
