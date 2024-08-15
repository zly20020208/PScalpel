function [output] = position_embedding(input)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    [dims,length] = size(input);
    PE = zeros(dims,length);
    for pos=1:length
        for i=1:dims
            val = pos/10000^(2*i/dims);
            if mod(i,2)==0
                PE(i,pos) = sin(val);
            else
                PE(i,pos) = cos(val);
            end
        end
    end
    output = input + PE*0.2;
end

