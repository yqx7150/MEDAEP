close all;
clear all;clc;

% add MatCaffe path
addpath ../mnt/data/siavash/caffe/matlab;
getd = @(p)path(path,p);% Add some directories to the path
addpath(genpath('../MEDAEP_imagedeblur/image/'));
addpath(genpath('../MEDAEP_imagedeblur/loadNET/'));
addpath(genpath('../MEDAEP_imagedeblur/quality_assess/'));
addpath(genpath('../MEDAEP_imagedeblur/model/'));
% set to 0 if you want to run on CPU (very slow)
use_gpu = 1;

%% Deblurring demo

% load image and kernel
load('Levin09.mat');
% gt = double(imread('barbara.tif'));
% gt = double(imread('boats.tif'));
% gt= double(imread('cameraman.tif'));
% gt = double(imread('baboon.tif'));
  gt = double(imread('peppers.tif'));
% gt = double(imread('straw.tif'));

kernel = kernels{1};%19%17%15%27%13%21%23%%25
sigma_d = 255 * .01;

pad = floor(size(kernel)/2);   
gt_extend = padarray(gt, pad, 'replicate', 'both');
degraded = convn(gt_extend, rot90(kernel,2), 'valid');
noise = randn(size(degraded));
degraded = degraded + noise * sigma_d;
figure(11);imshow(gt,[])
figure(22); imshow(degraded,[]);
%% load net% 
params.net =loadNet_single_real([size(gt_extend),1], use_gpu);
params.gt = gt;

%% run DAEP
params.num_iter =5000;
params.sigma_net = 25; 
map_deblur = DAEP_Deblursingle(degraded, kernel, sigma_d, params);


figure(222);
% subplot(131);
% imshow(gt/255); title('Ground Truth')
% subplot(132);
% imshow(degraded/255); title('Blurry')
% subplot(133);
imshow(map_deblur/255); title('Restored')

