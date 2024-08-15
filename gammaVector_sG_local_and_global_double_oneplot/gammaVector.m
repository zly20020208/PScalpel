function [y, Y] = gammaVector(net,data,G,dropout)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
%      x = position_embedding(x);
    L = size(net,2);
    z = [];
    %[y, Y] = FNN(dnn,x,dropout)
    [x, Y{1}.Y] = FNN(net{1},data,dropout);

    for i=2:L  
        % [output, Y] = GAT(net, input, G, dropout)
        [x, Y{i}.Y] = GAT(net{i}, x, G, dropout);
        z = [z;sum(x,2)];
    end

    y = z;

end
