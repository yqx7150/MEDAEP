function net2 = loadNet_qx3channel_diffSigma3(img_size, use_gpu)
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
net_size = [3, img_size(2), img_size(1)];

  caffe.reset_all();

if use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

net_model = './model/deploy_DAE_base.prototxt';
 net_weights='.\model/DAE_sigma25.caffemodel';
%   net_weights = 'F:\MRIComplex_Denoising_Qiegenliu_180327\DAE_N25_real_3D\modelDAEN25_real_3D/N25_real3D_iter_100000.caffemodel';
%     net_weights = 'C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N30_real_3D\modelDAEN30_real_3D/N30_real3D_iter_200000.caffemodel';
%     net_weights = 'F:\MRIComplex_Denoising_Qiegenliu_180327\DAE_N20_real_3D\modelDAEN20_real_3D/N20_real3D_iter_100000.caffemodel';
%     net_weights = 'C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N15_real_3D\modelDAEN15_real_3D/N15_real3D_iter_100000.caffemodel';
%    net_weights = 'C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N11_real_3D\modelDAEN11_real_3D/N11_real3D_iter_100000.caffemodel';

%  net_weights = 'C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N25_real_3D\BSDS500_modelDAEN25_real_3D2/N25_real3D_iter_190000.caffemodel';
%net_weights = 'C:\Users\个人电脑\Desktop\MRIComplex_Denoising_Qiegenliu_180327\DAE_N25_real_3D_CImageNet_1copyto3\modelDAEN25_real_3D_1repmat3\N25_3D_C_1copyto3_iter_100000.caffemodel';

FID_base = fopen(net_model, 'r');
Str_base = fread(FID_base, [1, inf]);
fclose(FID_base);
FID_net = fopen('./model/deploy_DAE_resized.prototxt', 'w');
fprintf(FID_net, char(Str_base), net_size);
fclose(FID_net);
net_model = './model/deploy_DAE_resized.prototxt';

net2 = caffe.Net(net_model, net_weights, 'test');
