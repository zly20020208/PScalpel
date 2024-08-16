function [net, adaptive_var, parameter] = creatnn(K)
    [net,adaptive_var] = creatDNN(K);
    

    %% 训练参数设置
    parameter.learning_rate = 0.01;
    parameter.delta = 1e-8;
    parameter.beta1 = 0.9;
    parameter.beta2 = 0.999;
    parameter.dropout = 0;
    
end
