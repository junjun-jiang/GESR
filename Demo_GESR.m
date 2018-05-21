% =========================================================================
% image super-resolution via GESR
% 
% All results reported in the paper were performed on a PC with a Dual-Core 3.20GZ CPU and 2GB RAM in MATLAB R2009a.
% =========================================================================

clear all; clc;
close all;
addpath('.\utilities')

% load dictionary
load('.\Training\dictionary_Dh_Dl_100000(2048)_4.mat');

% set parameters
patch_size  = 5;            % image patch size
upscale     = 4;            % upscaling factor
maxIter     = 20;          % if 0, do not use backprojection
lambda      = 0.4;          % regularization parameter
K           = 35;           % neighbors for constructing the graph
overlap     = 4;            % the more overlap the better (patch size 5x5)

% training phase
fprintf('training...Obtain the projection matrix \n');
A_P =  CalulateProjMatrix(Dl,Dh,lambda,K); % obtain the projection matrix A

% read test image
fprintf('read the test image and super-resolved it...\n');

% Names = {'bike','Butterfly','flower','girl','hat','leaves','lena','Parrots','Plants','raccoon'};
str = strcat('.\Data\Testing\girl.tif');
im = imread(str); % ground truth
im_l = imresize(im, 1/upscale, 'bicubic'); % LR image

tic

% change color geace, work on illuminance only
im_l_ycbcr = rgb2ycbcr(im_l);
im_l_y = im_l_ycbcr(:, :, 1);
im_l_cb = im_l_ycbcr(:, :, 2);
im_l_cr = im_l_ycbcr(:, :, 3);

% image super-resolution based on gearse representation
[im_h_y] = GESR(im_l_y, upscale, Dh, Dl, overlap, A_P);
[im_h_y] = backprojection(im_h_y, im_l_y, maxIter);

% upscale the chrominance simply by "bicubic" 
[nrow, ncol] = size(im_h_y);
im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

im_h_ycbcr = zeros([nrow, ncol, 3]);
im_h_ycbcr(:, :, 1) = im_h_y;
im_h_ycbcr(:, :, 2) = im_h_cb;
im_h_ycbcr(:, :, 3) = im_h_cr;
im_h = ycbcr2rgb(uint8(im_h_ycbcr));

fprintf('done.');
fprintf('the running time is %f s \n',toc);

% bicubic interpolation for reference
im_b = imresize(im_l, [nrow, ncol], 'bicubic');

% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(im, im_b);
ge_rmse = compute_rmse(im, im_h);

bb_psnr = 20*log10(255/bb_rmse);
ge_psnr = 20*log10(255/ge_rmse);

fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for GESR Recovery: %f dB\n', ge_psnr);

% show the images
figure, imshow(im_h);
title('GESR Recovery');
figure, imshow(im_b);
title('Bicubic Interpolation');

str = strcat('.\Data\Testing\Results\girl.tif');
imwrite(uint8(im_h),str);


