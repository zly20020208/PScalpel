function [dx, dgamma, dbeta] = batchnorm_backward(dout, cache)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
   %解压中间变量
  xhat = cache.xhat;
  gamma = cache.gamma;
  xmu = cache.xmu;
  ivar = cache.ivar;
  sqrtvar = cache.sqrtvar;
  var = cache.var;
  
  dout = dout';

  [N,D] = size(dout);

  %step6
  dbeta = sum(dout);
  dgammax = dout;
  dgamma = sum(dgammax.*xhat);
  dxhat = dgammax .* gamma;

  %step5
  divar = sum(dxhat.*xmu);
  dxmu1 = dxhat .* ivar; %注意这是xmu的一个支路

  %step4
  dsqrtvar = -1./(sqrtvar.^2) .* divar;
  dvar = 0.5 * 1./sqrt(var+1e-8) .* dsqrtvar;

  %step3
  dsq = 1./N * ones(N,D) .* dvar;
  dxmu2 = 2 * xmu .* dsq; %注意这是xmu的第二个支路

  %step2
  dx1 = (dxmu1 + dxmu2); %注意这是x的一个支路


  %step1
  dmu = -1 * sum(dxmu1+dxmu2);
  dx2 = 1. /N * ones(N,D) .* dmu; %注意这是x的第二个支路

  %step0 done!
  dx = dx1 + dx2;
  
  dx = dx';
  
end

