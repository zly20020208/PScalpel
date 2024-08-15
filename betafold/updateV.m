function [new_v] = updateV(parameter, dw, old_v)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
new_v = max(parameter.beta2*old_v + (1 - parameter.beta2)*dw.*dw,old_v); 
end

