function [G] = generation_G(net_betafold, data, threshold)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
    n_size = size(data,2);
    [output,~] = forwordprop(net_betafold,data,n_size,0);
    G = output>threshold;
end

