function [auc] = get_auc(y,label)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    len = length(y);
    L = label;
    m = sum(L);
    n = len - m;
    if m==0
       m=1; 
    end
    if n==0
        n=1;
    end
    [~,auc_ind] = sort(-y);
    sum_p_rank = 0;
    for auc_i=1:len
        if L(auc_ind(auc_i))==1
            sum_p_rank = sum_p_rank + len + 1 - auc_i;
        end
    end
    auc = (sum_p_rank-m*(1+m)/2)/(m*n);
end

