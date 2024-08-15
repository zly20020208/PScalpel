function [y] = sigmoid(x)
%sigmoid sigmoid激活函数
%   此处显示详细说明
y = 1./(1 + exp(-x));
end
