function [y] = relu(x)
%relu 激活函数
%   此处显示详细说明
p = (x > 0);
y = x.*p;
end

