function [net, adaptive_var, parameter] = creatnn(input_dim,mid_dim,appli_dim,layer_num,K,use_gpu)
%UNTITLED6 此处显示有关此函数的摘要
    %% 创建网络
    [net{1},adaptive_var{1}] = creatFFN(input_dim,mid_dim, true);
    for i=2:layer_num+1
        % [net, adv] = creatBlock(wide)
        [net{i},adaptive_var{i}] = creatBlock(mid_dim, true); 
        if use_gpu
           net{i} = net_gpu(net{i}); 
           adaptive_var{i} = net_gpu(adaptive_var{i});
        end
    end
    [net{layer_num+2},adaptive_var{layer_num+2}] = createApplicationLayer(mid_dim, appli_dim, true, true,K, use_gpu); 
    if use_gpu
       net{layer_num+2} = net_gpu(net{layer_num+2}); 
       adaptive_var{layer_num+2} = net_gpu(adaptive_var{layer_num+2});
    end

    %% 训练参数设置
    parameter.learning_rate = 0.01;
    parameter.delta = 1e-8;
    parameter.beta1 = 0.9;
    parameter.beta2 = 0.999;
    parameter.dropout = 0;
    if use_gpu
        parameter = net_gpu(parameter); 
    end
    
end
