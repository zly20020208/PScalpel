function [y, Y] = DNN(dnn,x,dropout)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    L = size(dnn,2);
    m = size(x,2);
    Y.Y{1} = x;
    for i = 1:L
        z = dnn{i}.W*x + repmat(dnn{i}.b,1,m);
        [z,Y.Cache{i}] = batchnorm_forward(z,dnn{i}.gamma,dnn{i}.beta);
%         [z,Y.Cache{i}] = batchnorm_forward(z,1,0);
%         [z,Y.Cache{i}] = batchnorm_forward(z,1,0);

        if dnn{i}.function == "relu2"
            y = relu2(z);
        end
        if dnn{i}.function == "relu"
            y = relu(z);
        end
        if dnn{i}.function == "selu"
            y = selu(z);
        end
        if dnn{i}.function == "sigmoid"
            y = sigmoid(z);
        end
        if dnn{i}.function == "softmax"
            z = z/0.7;
            z = z - max(z);
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
