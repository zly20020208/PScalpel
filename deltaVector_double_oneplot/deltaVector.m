function [y, Y] = deltaVector(net,data,dropout)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    L = size(net,2);
    n_size = size(data,2);
    z = [];
    %[y, Y] = FNN(dnn,x,dropout)
    data = position_embedding(data);
    [x, Y{1}.Y] = FNN(net{1},data,dropout);
    for i=2:L
        % [output, Y] = selfAttention(attentionLayer, input, n_size, dropout)
        [x, Y{i}.Y] = selfAttention(net{i}, x, n_size, dropout);
        z = [z;sum(x,2)];
    end
    y = z;

end
