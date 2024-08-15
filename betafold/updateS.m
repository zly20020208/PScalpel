function [new_s] = updateS(parameter,dw, old_s)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
new_s = parameter.beta1*old_s+ (1 - parameter.beta1)*dw; 
end

