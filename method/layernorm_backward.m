function [dx, dgamma, dbeta] = layernorm_backward(dout, cache)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
   %解压中间变量
  D = cache.D;
  gamma = cache.gamma;
  xmu = cache.xmu;
  sqrtvar = cache.sqrtvar;

  
  dgamma = sum(dout.*xmu./sqrtvar,2);
  dbeta = sum(dout,2);
  
  dlxhat = dout.*gamma;
  dxhatx = 1./sqrtvar;
  dlvar = -0.5*sum(gamma.*xmu.*sqrtvar.^(-3).*dout);
  dlvarx = 2*xmu/D;
  dlmu = -1.*sum(dlxhat./sqrtvar)-2.*sum(dlvar.*xmu)/D;
  
  dx = dlxhat.*dxhatx + dlvar.*dlvarx + dlmu/D;

  
end

