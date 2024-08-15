function [y, Y] = FNN(dnn,x,dropout)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    normalize = dnn.normalize;
    m = size(x,2);
    Y.Y{1} = x;
    
    z = dnn.W*x + repmat(dnn.b,1,m);
    if normalize
        [z,Y.Cache] = batchnorm_forward(z,dnn.gamma,dnn.beta);
    else
        [z,Y.Cache] = batchnorm_forward(z,1,0);
    end
    y = selu(z);
    y = Dropout(y,dropout);
    Y.Y{2} = y;
end
