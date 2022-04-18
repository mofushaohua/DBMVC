
function [pred_label] = DBMVC(Z,n_cluster, viewNum,beta, gamma, lambda)
%--------------------Parameters--------------------------------------------
MaxIter = 7;       % 5 iterations are okay, but better results for 10
innerMax = 10;
r = 5;              % r is the power of alpha_i
L = 128;            % Hashing code length
%--------------------End Parameters----------------------------------------


%--------------------Initialization----------------------------------------
%viewNum = size(Z,2); % number of views
N = size(Z{1},2);
alpha = ones(viewNum,1) / viewNum;


rand('seed',100);
%init some need matrix
if length(Z)<4
    sel_sample = Z{viewNum}(:,randsample(N, 100),:);
    [pcaW, ~] = eigs(cov(sel_sample'), L);
    B = sign(pcaW'*Z{viewNum});
else
    sel_sample = Z{4}(:,randsample(N, 100),:);
    [pcaW, ~] = eigs(cov(sel_sample'), L);
    B = sign(pcaW'*Z{4});
end




P = cell(1,viewNum);%

Pw=cell(1,viewNum);%init projection matrix
for v = 1:viewNum
     W = constructW_PKN_anchor(Z{v}', Z{v}', 15);
     D=diag(sum(W));
     Pw{v}=D-W;
 end
    
 
rand('seed',500);
C = B(:,randsample(N, n_cluster));
HamDist = 0.5*(L - B'*C);
[~,ind] = min(HamDist,[],2);
G = sparse(ind,1:N,1,n_cluster,N,N);
G = full(G);
CG = C*G;

ZZT = cell(1,viewNum);
for view = 1:viewNum
    ZZT{view} = Z{view}*Z{view}';
end
clear HamDist ind initInd n_randm pcaW sel_sample view
%--------------------End Initialization------------------------------------


%--------------------The proposed method-----------------------------------
%disp('----------The proposed method (multi-view)----------');
  
%diagA = ones(1,N)*Zstar*Zstar';
Mr = cell(1,viewNum);
for i=1:viewNum
    [x,~]=size(Z{i});
     Mr{i}=ones(x,1);%初始化对角阵元素
end
MR = cell(1,viewNum);
for iter = 1:MaxIter
   % fprintf('The %d-th iteration...\n',iter);
    
    %---------Update Pv--------------
    alpha_r = alpha.^r;
    PX = zeros(L,N);
    for v = 1:viewNum
      [x,~]=size(Mr{v});
      MR{v}=spdiags(Mr{v},0,x,x);
        P{v} = B*Z{v}'/(ZZT{v}+beta*MR{v}+gamma*Pw{v});
        PX   = PX+alpha_r(v)*P{v}*Z{v};
        Pi=sqrt(sum(P{v}.*P{v},1)+eps);
        Mr{v}=0.5./Pi;
    end
    
    %---------Update B--------------
    B=sign(PX+lambda*CG);B(B==0)=-1;
 
    %---------Update C and G--------------
    for iterInner = 1:innerMax
        C = sign(B*G'); C(C==0) = 1;
        rho = .001; mu = .01; % Preferred for this dataset
        for iterIn = 1:3
            grad = -B*G' + rho*repmat(sum(C),L,1);
            C    = sign(C-1/mu*grad); C(C==0) = 1;
        end
        
        HamDist = 0.5*(L - B'*C); 
        [~,indx] = min(HamDist,[],2);
        G = sparse(indx,1:N,1,n_cluster,N,N);
    end
    CG = C*G;
  
    %---------Update alpha--------------
    h = zeros(viewNum,1);
    for view = 1:viewNum
        h(view) = norm(B-P{view}*Z{view},'fro')^2 + beta*trace(P{view}*diag(Mr{view})*P{view}'+gamma*trace(P{view}*Pw{view}*P{view}'));
    end
    H = bsxfun(@power,h, 1/(1-r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,H,sum(H)); % alpha = H./sum(H);
   
end
%disp('----------Main Iteration Completed----------');
[~,pred_label] = max(G,[],1);
end