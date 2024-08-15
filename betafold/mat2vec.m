function [vec] = mat2vec(mat)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    len = length(mat);
    vec = [];
    for i=1:len-1
        vec = [vec,mat(i,i+1:end)];
    end
end
