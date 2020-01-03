function z = bl_pooling(descrs)
y  = cell(1,3);
for si = 1:numel(descrs)
    feats = descrs{si};
    [h,w,ch] = size(feats) ;
    feats = reshape(feats, h*w, ch);
    feats = reshape(feats'*feats,[1,ch*ch])/(h*w);
    feats = sign(feats).* sqrt(abs(feats)) ;
    feats = bsxfun(@times, feats, 1./(sqrt(sum(feats.^2))+eps)) ;
    y{si} = feats ;
       
end
rate = [1 1 1] ;
z= rate(1)*y{1}+rate(2)*y{2}+rate(3)*y{3} ;
end

