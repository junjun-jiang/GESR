function [Dh, Dl] = patch_clustering(Dh, Dl, centerNum)

%%% perform clustering
options = foptions;
options(1) = 1; % display
options(2) = 1;
options(3) = 0.01; % precision
options(5) = 1; % initialization
options(14) = size(Dh,2); % maximum iterations
centers = zeros(centerNum,size(Dl, 1)+size(Dh, 1));
%%% run kmeans
fprintf('\nRunning k-means\n');
sift_all = [Dh; Dl]';
dictionary = sp_kmeans(centers, sift_all, options);
Dh = dictionary(:,1:size(Dh, 1))';
Dl = dictionary(:,size(Dh, 1)+1:end)';
