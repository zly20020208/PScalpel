function [g] = dsoftmax(g,y)
%UNTITLED11 此处显示有关此函数的摘要
%   此处显示详细说明
    dout = diag(y) - y * y';
    g = dout * g;
end

