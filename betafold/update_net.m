function [new_w] = update_net(parameter, s, V, global_step, old_w)
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明
new_w = old_w - parameter.learning_rate*(s/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(V./(1 - parameter.beta2.^global_step)));
end

