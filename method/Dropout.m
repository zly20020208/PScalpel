function [out] = Dropout(out,level)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    if level < 0 || level >1
        raise ValueError('Dropout level must be in interval [0, 1].')
    end
    retain_prob = 1 - level;
    shape = size(out,1);
    random_tensor = rand(shape,1)<=retain_prob;
 
    out = out.*random_tensor;
    out = out / retain_prob;
end

