function [y, Y] = block(net,x,n_size,dropout)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    normalize = net.normalize;
     
     X = x;
     
     
     if normalize
        [x,Y.Cache{1}] = layernorm_forward(x,net.gamma{1},net.beta{1});
     else
         [x,Y.Cache{1}] = layernorm_forward(x,1,0);
     end
     % [output, Y] = selfAttention(attentionLayer, input, dropout)
     [x, Y.Y{1}] = selfAttention(net.layers{1}, x, n_size, dropout); 
     
     x = x + X;
     X = x;
     
     if normalize
        [x,Y.Cache{2}] = layernorm_forward(x,net.gamma{2},net.beta{2});
     else
         [x,Y.Cache{2}] = layernorm_forward(x,1,0);
     end
     % [y, Y] = FNN(dnn,x,dropout)
     [x, Y.Y{2}] = FNN(net.layers{2},x,dropout);
     y = x + X;
end
