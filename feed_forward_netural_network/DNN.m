function [y, Y] = DNN(net,x,dropout)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    dnn = net.dnn;
    L = size(dnn,2);
    m = size(x,2);
    Y.Y{1} = x;
    for i = 1:L
        z = dnn{i}.W*x + repmat(dnn{i}.b,1,m);
        if m>1
            [z,Y.Cache{i}] = batchnorm_forward(z,dnn{i}.gamma,dnn{i}.beta);
        else
            z = batchnorm_forward_test(z,dnn{i}.gamma,dnn{i}.beta,dnn{i}.mu,dnn{i}.var);
        end
        
%         [z,Y.Cache{i}] = batchnorm_forward(z,dnn{i}.gamma,dnn{i}.beta);
        
        if dnn{i}.function == "relu"
            y = relu(z);
        end
        if dnn{i}.function == "selu"
            y = selu(z);
        end
        if dnn{i}.function == "sigmoid"
            y = sigmoid(z);
        end
        if dnn{i}.function == "tanh"
            y = tanh(z);
        end
        if dnn{i}.function == "softmax"
            y = softmax(z);
        end
        if dnn{i}.function == "no"
            y = z;
        end
        if i~=L
            y = Dropout(y,dropout);
        end
        Y.Y{i+1} = y;
        x = y;
    end
end
