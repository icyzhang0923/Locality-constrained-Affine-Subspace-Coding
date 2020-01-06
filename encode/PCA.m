function [eigvector, eigvalue, meanData, new_data] = PCA(data, options)
%PCA	Principal component analysis
%	Usage:
%	[EIGVECTOR, EIGVALUE, MEANDATA, NEW_DATA] = PCA(DATA, OPTIONS)
%
%	DATA: Rows of vectors of data points
%   options - Struct value in Matlab. The fields in options
%             that can be set:
%           ReducedDim   -  The dimensionality of the
%                           reduced subspace. If 0,
%                           all the dimensions will be
%                           kept. Default is 0.
%           PCARatio     -  The percentage of principal
%                           component kept. The percentage is
%                           calculated based on the
%                           eigenvalue. Default is 1
%                           (100%, all the non-zero
%                           eigenvalues will be kept.
%          Please do not set both these two fields. If both of them are
%          set, PCARatio will be used. 
%          
%   MEANDATA: Mean of all the data. 
%	NEW_DATA: The data after projection (mean removed)
%	EIGVECTOR: Each column of this matrix is a eigenvector of DATA'*DATA
%	EIGVALUE: Eigenvalues of DATA'*DATA
%
%	Examples:
% 			data = rand(7,10);
% 			options = [];
% 			options.ReducedDim = 4;
% 			[eigvector,eigvalue,meanData,new_data] = PCA(data,options);
% 
% 			data = rand(7,10);
% 			options = [];
% 			options.PCARatio = 0.98;
% 			[eigvector,eigvalue,meanData,new_data] = PCA(data,options);
%
%
% Note: Be aware of the "mean". For classification, you can use this code by:
% 	
%			fea_Train = fea(trainIdx,:);
%
%			[eigvector, eigvalue] = PCA(fea_Train, options)
%			fea_New = fea*eigvector;
%		or
%			[eigvector, eigvalue, meanData] = PCA(fea_Train, options)
%			fea_New_2 = (fea - repmat(meanData,nSmp,1))*eigvector;
%
%	 Then classification is then performed on "fea_new" or "fea_new_2". 
%	 Since we use Euclidean distance, the result will be the same for 
%	 nearest neighbor classifier on "fea_new" or "fea_new_2". 
%
%    If you call PCA by:
%			[eigvector, eigvalue, meanData, fea_Train_new] = PCA(fea_Train, options);
%	  Since "fea_Train_new" is "mean removed", you should also subtract the "meanData" 
%	  from each testing example, like
%			fea_Test = fea(testIdx,:);
%			fea_Test_new = (fea_Test - repmat(meanData,nSmpTest,1))*eigvector;

%
% 
%    Written by Deng Cai (dengcai@gmail.com), April/2004, Feb/2006

if (~exist('options','var'))
   options = [];
else
   if ~strcmpi(class(options),'struct') 
       error('parameter error!');
   end
end

bRatio = 0;
if isfield(options,'PCARatio')
    bRatio = 1;
    eigvector_n = min(size(data));
elseif isfield(options,'ReducedDim')
    eigvector_n = options.ReducedDim;
else
    eigvector_n = min(size(data));
end
    

[nSmp, nFea] = size(data);

meanData = mean(data);
data = data - repmat(meanData,nSmp,1);

if nSmp >= nFea
    ddata = data'*data;
    ddata = (ddata + ddata')/2;
    if issparse(ddata)
        ddata = full(ddata);
    end

    if size(ddata, 1) > 100 & eigvector_n < size(ddata, 1)/2  % using eigs to speed up!
        option = struct('disp',0);
        [eigvector, d] = eigs(ddata,eigvector_n,'la',option);
        eigvalue = diag(d);
    else
        [eigvector, d] = eig(ddata);
        eigvalue = diag(d);
        % ====== Sort based on descending order
        [junk, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        eigvector = eigvector(:, index);
    end
    
    clear ddata;
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
    eigvalue (eigIdx) = [];
    eigvector (:,eigIdx) = [];

else	% This is an efficient method which computes the eigvectors of
	% of A*A^T (instead of A^T*A) first, and then convert them back to
	% the eigenvectors of A^T*A.
    if nSmp > 700
        ddata = zeros(nSmp,nSmp);
        for i = 1:ceil(nSmp/100)
            if i == ceil(nSmp/100)
                ddata((i-1)*100+1:end,:) = data((i-1)*100+1:end,:)*data';
            else
                ddata((i-1)*100+1:i*100,:) = data((i-1)*100+1:i*100,:)*data';
            end
        end
    elseif nSmp > 400
        ddata = zeros(nSmp,nSmp);
        for i = 1:ceil(nSmp/200)
            if i == ceil(nSmp/200)
                ddata((i-1)*200+1:end,:) = data((i-1)*200+1:end,:)*data';
            else
                ddata((i-1)*200+1:i*200,:) = data((i-1)*200+1:i*200,:)*data';
            end
        end
    else
        ddata = data*data';
    end
   
    ddata = (ddata + ddata')/2;
    if issparse(ddata)
        ddata = full(ddata);
    end
    
    if size(ddata, 1) > 100 & eigvector_n < size(ddata, 1)/2  % using eigs to speed up!
        option = struct('disp',0);
        [eigvector1, d] = eigs(ddata,eigvector_n,'la',option);
        eigvalue = diag(d);
    else
        [eigvector1, d] = eig(ddata);
        
        eigvalue = diag(d);
        % ====== Sort based on descending order
        [junk, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        eigvector1 = eigvector1(:, index);
    end

    clear ddata;
    
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
    eigvalue (eigIdx) = [];
    eigvector1 (:,eigIdx) = [];

    eigvector = data'*eigvector1;		% Eigenvectors of A^T*A
    clear eigvector1;
	eigvector = eigvector*diag(1./(sum(eigvector.^2).^0.5)); % Normalization

end

if bRatio
    if options.PCARatio >= 1 | options.PCARatio <= 0
        idx = length(eigvalue);
    else
        sumEig = sum(eigvalue);
        sumEig = sumEig*options.PCARatio;
        sumNow = 0;
        for idx = 1:length(eigvalue)
            sumNow = sumNow + eigvalue(idx);
            if sumNow >= sumEig
                break;
            end
        end
    end

    eigvalue = eigvalue(1:idx);
    eigvector = eigvector(:,1:idx);
else
    if eigvector_n < length(eigvalue)
        eigvalue = eigvalue(1:eigvector_n);
        eigvector = eigvector(:, 1:eigvector_n);
    end
end

if nargout == 4
    new_data = data*eigvector;
end

