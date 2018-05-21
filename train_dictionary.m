

% All training sets were performed on a PC with a Dual-Core 3.20GZ CPU and 16GB RAM.
addpath('.\utilities');

TR_IMG_PATH = 'Data/Training';
dict_size   = 1024;          % dictionary size
nSmp        = 100000;         % number of patches to sample
patch_size  = 5;            % image patch size
upscale     = 4;            % upscaling factor

% randomly sample image patches
[Xh, Xl] = rnd_smp_patch(TR_IMG_PATH, '*.bmp', patch_size, nSmp, upscale);
 
% prune patches with small variances, threshould chosen based on the training data
[Xh, Xl] = patch_pruning(Xh, Xl, 10);
[Dh, Dl] = patch_clustering(Xh, Xl, dict_size);

str = strcat('.\Training\dictionary_Dh_Dl_',num2str(nSmp),'(',num2str(dict_size),')_',num2str(upscale),'.mat');
save(str,'Dh','Dl');