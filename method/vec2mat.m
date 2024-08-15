function [mat] = vec2mat(vec,len)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
    mat = zeros(len,len);
    index = 1;
    for i=1:len-1
        l = len-i;
        mat(i,i+1:end) = vec(index:index+l-1);
        index = index + l;
    end
    mat = mat + mat';
end

