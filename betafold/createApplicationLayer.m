function [applicationLayer, adaptive_var] = createApplicationLayer(seq_dim, key_size, nolinear, normalize, K, use_gpu)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    applicationLayer.normalize = normalize;%true表示该层需要对输出结果进行标准化
    applicationLayer.seq_dim = seq_dim;%输入序列的特征维度
    applicationLayer.key_size = key_size;%K的特征维度
    applicationLayer.nolinear = nolinear;%true表示该层加入了偏置，可以进行非线性变换
    applicationLayer.use_gpu = use_gpu;%true表示该层使用了gpu
    
    [applicationLayer.dnn,adaptive_var.adv] = creatDNN(K);
    
    
    %% 初始化K的相关权重
    applicationLayer.KW = randn(key_size,seq_dim);
    if normalize
        applicationLayer.Kgamma = 1;
        applicationLayer.Kbeta = 0;
    end    
    
    if nolinear
        applicationLayer.Kb = randn(key_size,1);
    end
    

    
    

    
    %% 定义自适应变量
    %定义V

    adaptive_var.V.vKW = applicationLayer.KW * 0;
    if normalize
        adaptive_var.V.vKgamma = 0;
        adaptive_var.V.vKbeta = 0;
    end
    if nolinear
        adaptive_var.V.vKb = applicationLayer.Kb * 0;
    end

    
    %定义s

    adaptive_var.s.vKW = applicationLayer.KW * 0;
    if normalize
        adaptive_var.s.vKgamma = 0;
        adaptive_var.s.vKbeta = 0;
    end
    if nolinear
        adaptive_var.s.vKb = applicationLayer.Kb * 0;
    end

    

end

