function [g] = drelu(g,y)
%relu 激活函数
%   此处显示详细说明
    g = g.*(y>0);
end

