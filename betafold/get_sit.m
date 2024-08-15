function [sit] = get_sit(i,j,len)
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明
sit = (i-1)*len - (i-1)*(i-2)/2 +j;
end

