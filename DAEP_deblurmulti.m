%function map = DAEP_deblurmulti(degraded, kernel, sigma_d, params, params2)
% Input:
% degraded: Observed degraded RGB input image in range of [0, 255].
% kernel: Blur kernel (internally flipped for convolution).
% sigma_d: Noise standard deviation.
% params1: Set of parameters with regard to network 1.
% params2: Set of parameters with regard to network 2.
% params1.net: The DAE Network1 object loaded from MatCaffe.
% params2.net: The DAE Network2 object loaded from MatCaffe.
% Optional parameters:
% params.sigma_net: The standard deviation of the network training noise. default: 25
% params.num_iter: Specifies number of iterations.
% params.gamma: Indicates the relative weight between the data term and the prior. default: 6.875
% params.mu: The momentum for SGD optimization. default: 0.9
% params.alpha the step length in SGD optimization. default: 0.1
% Outputs:
% map: Solution.


