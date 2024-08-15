function [out] = batchnorm_forward_test(x, gamma, beta, mu, var)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
  x = x';
  xmu = x - mu;
  
  sqrtvar = sqrt(var + 1e-8);
  ivar = 1./sqrtvar;
  
  xhat = xmu .* ivar;

  %step6: scale and shift
  gammax = gamma .* xhat;
  out = gammax + beta;
  
  out = out';

end

