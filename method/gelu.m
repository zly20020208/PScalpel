function [y] = gelu(x)
%relu 激活函数
%   此处显示详细说明
    y = 0.5*x.*(1+tanh(sqrt(2/pi)*(x+0.044715*x.^3)));
end

