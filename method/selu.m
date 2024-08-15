function [y] = selu(x)
%relu 激活函数
%   此处显示详细说明
    alpha = 1.6732632423543772848170429916717;
    scale = 1.0507009873554804934193349852946;
    p = (x > 0);
    y = x.*p + alpha*(exp(x)-1).*(1-p);
    y = y * scale;
end

