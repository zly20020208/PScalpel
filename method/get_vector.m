function [vector] = get_vector(sequence)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
load('dict.mat')
vector = ones(100,1)*0.01;
seq = char(sequence);
len = length(seq);
for i=1:len-2
    g3 = seq(i:i+2);
    if isfield(dict,g3)
        vector = [vector,dict.(g3)'];
    else
        vector = [vector,dict.U0N0K'];
    end
end
vector = [vector,ones(100,1)*0.02];
    
end

