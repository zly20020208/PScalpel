function [out, Cache] = layernorm_forward(x, gamma, beta)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
  %为了后向传播求导方便，这里都是分步进行的
  %step1: 计算均值
  mu = mean(x);

  %step2: 减均值
  xmu = x - mu;

  %step3: 计算方差
  sq = xmu .^ 2;
  var = mean(sq);

  %step4: 计算x^的分母项
  sqrtvar = sqrt(var + 1e-8);
  ivar = 1./sqrtvar;

  %step5: normalization->x^
  xhat = xmu .* ivar;

  %step6: scale and shift
  gammax = gamma .* xhat;
  out = gammax + beta;
  

  %存储中间变量
    Cache.D = size(x,1);   
    Cache.xmu = xmu;
    Cache.sqrtvar = sqrtvar;
    Cache.gamma = gamma;
end

