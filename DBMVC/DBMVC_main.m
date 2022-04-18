function [res] = DBMVC_main(fea, numClust, gt, r1Temp, r2Temp, r3Temp, r4Temp, anc_idx)

X = fea;
v = length(fea);
index = anc_idx{r4Temp};
for it = 1 : v  
    Anchor{it} = fea{it}(index(it,:),:); %select landmark points
    dist = EuDist2(X{it},Anchor{it},0);
    sigma = mean(min(dist,[],2).^0.5)*2;
    feaVec = exp(-dist/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feaVec', mean(feaVec',2));% Centered data
end
    
beta = r1Temp; 
gamma = r2Temp;  
lambda = r3Temp;   
viewNum = v;
res_label = DBMVC(X, numClust, viewNum, beta, gamma, lambda);
res = zeros(2, 8);
res(1,:) = Clustering8Measure(gt, res_label);
end

