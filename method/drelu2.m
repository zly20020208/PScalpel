function [g] = drelu2(g,y)
%relu 激活函数
%   此处显示详细说明
    g = g.*sqrt(y)*2;
end

