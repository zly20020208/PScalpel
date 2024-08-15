function [auc] = get_auc_multi(y,label)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    sum=0;
    len = size(label,1);
    for i=1:len
       sum = sum + get_auc(y(i,:),label(i,:)); 
    end
    auc = sum/len;
end

