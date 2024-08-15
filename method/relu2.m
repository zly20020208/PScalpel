function [y] = relu2(x)
%relu 激活函数
%   此处显示详细说明
p = (x > 0);
y = x.*p;
y = y.^2;
end

