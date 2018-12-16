function net = loadNet_single_real(img_size, use_gpu)
% Loads a Caffe 'net' object for a specific image dimensions
%
%
% Input:
% img_size: MAP Image size [Height, Width].
% use_gpu: GPU flag: use 1 if you use GPU, use 0 to run on CPU.
%
% Output:
% map: Caffe 'net' object.

%%
net_size = [1, img_size(2), img_size(1)];

caffe.reset_all();

if use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

 net_model = './model/deploy_DAE_base_singlereal.prototxt';
 net_weights = './model\N25_1D_iter_100000.caffemodel';

%   net_weights = 'C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N25_real_1D\modelDAEN25_real_1D\N25_1D_iter_35000.caffemodel';
%net_weights = 'C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N25_real_1D_CImageNet_Netlayer5\modelDAEN25_real_1D_Netlayer5\N25_1D_C_iter_100000.caffemodel';

FID_base = fopen(net_model, 'r');
Str_base = fread(FID_base, [1, inf]);
fclose(FID_base);
%FID_net = fopen('MRIComplex_Denoising_Qiegenliu_20180324/DAE25/modelDAE25/deploy_DAE_resized_20180324.prototxt', 'w');
 FID_net = fopen('./model/deploy_DAE_resized_singlereal.prototxt', 'w');
%FID_net = fopen('C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N25_real_1D_CImageNet_Netlayer5/deploy_DAE_resize_net.prototxt', 'w');

fprintf(FID_net, char(Str_base), net_size);
fclose(FID_net);
% net_model = 'MRIComplex_Denoising_Qiegenliu_20180324/DAE25/modelDAE25/deploy_DAE_resized_20180324.prototxt';
 net_model = './model/deploy_DAE_resized_singlereal.prototxt';
% net_model = 'C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N25_real_1D_CImageNet_Netlayer5/deploy_DAE_resize_net.prototxt';

net = caffe.Net(net_model, net_weights, 'test');
