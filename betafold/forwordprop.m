function [y, Y] = forwordprop(net,x,n_size,dropout)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
     x = position_embedding(x);
     L = size(net,2);
     %[y, Y] = FNN(dnn,x,dropout)
     [x, Y{1}] = FNN(net{1},x,dropout);
     for i=2:L-1
         % [y, Y] = block(net,x,dropout)        
         [x, Y{i}] = block(net{i}, x, n_size, dropout);  
     end
     % [output, Y] = application(applicationLayer, input)
     [y, Y{L}] = application(net{L}, x, n_size, dropout);
end
