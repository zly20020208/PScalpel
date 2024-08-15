function [g] = dtanh(g,y)
%relu 激活函数
%   此处显示详细说明
    g = g.*(1-y.^2);
end

