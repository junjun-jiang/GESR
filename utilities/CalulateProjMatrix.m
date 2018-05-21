


function A_P = CalulateProjMatrix(Dl,Dh,lambda,K)
%%% Dl : LR training patches
%%% Dh : HR training patches

% normalize the dictionarDh
norm_Dl = sqrt(sum(Dl.^2, 1)); 
Dl = Dl./repmat(norm_Dl, size(Dl, 1), 1);

norm_Dh = sqrt(sum(Dh.^2, 1)); 
Dh = Dh./repmat(norm_Dh, size(Dh, 1), 1);

TrainNum = size(Dh,2);

 
options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = K;
[W, elapse] = constructW(Dh',options);
W = (W + W')/2;

I = eye(TrainNum,TrainNum);
G = Dl*(I-W)*(I-W)'*Dl';

% A_P = pinv(Dl*Dl' + lambda*G)*Dl*Dh';

% A_P = pinv(Dl*Dl'+1e-9)*Dl*Dh';

U = Dl*Dl' + lambda*G;
V = Dl*Dh';
for i = 1:size(V,2)
    A_P(:,i) = lsqnonneg(U,V(:,i));
end

% [A_P,E] = lrra(V,U,0.1);
