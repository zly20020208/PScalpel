function [applicationLayer, adaptive_var, gI, loss] = train_application(applicationLayer, parameter, adaptive_var, Y, label, global_step, n_size)
%UNTITLED9 此处显示有关此函数的摘要
%   此处显示详细说明
    nolinear = applicationLayer.nolinear;
    normalize = applicationLayer.normalize;%true表示该层需要对输出结果进行标准化
    
    
    %% 欠采样
    m = size(label,2);
    probability = sum(label)/(m-sum(label));
    index = [];
    for i=1:m
        if label(i)==1
           index = [index,i];
        elseif rand < probability
           index = [index,i];
        end
    end
    label = label(index);
    for i=1:length(Y.Y.Y)
        Y.Y.Y{i} = Y.Y.Y{i}(:,index);
        if i ~= length(Y.Y.Y)
            Y.Y.Cache{i}.xmu = Y.Y.Cache{i}.xmu(index,:);
            Y.Y.Cache{i}.xhat = Y.Y.Cache{i}.xhat(index,:);
%             Y.Y.Cache{i}.sqrtvar = Y.Y.Cache{i}.sqrtvar(:,index);
        end
    end
    
    

    %% 计算dnn的梯度
    [applicationLayer.dnn, adaptive_var.adv, g, loss] = train_DNN(applicationLayer.dnn, parameter, adaptive_var.adv, Y.Y, label, global_step);

    
    if applicationLayer.use_gpu
        loss = gather(loss);
    end
    %% 计算K的梯度
    gK = [];
    k=1;
    l = length(index);
    nn_sum = 0;
    for t=1:length(n_size)
        len = n_size(t);
        nn_size = len*(len-1)/2;
        if applicationLayer.use_gpu
            gk = gpuArray(single(zeros(applicationLayer.key_size,len)));
        else
            gk = zeros(applicationLayer.key_size,len);
        end
        
        
        i=1;
        while k<=l&&index(k)<=nn_sum + nn_size
            [i,j] = get_ij(index(k)-nn_sum,len-1,i);
            j = j + i;
            gk(:,i) = gk(:,i) + g(1:applicationLayer.key_size,k);
            gk(:,j) = gk(:,j) + g(applicationLayer.key_size+1:applicationLayer.key_size*2,k);
            
            k = k+1; 
        end
        gK = [gK,gk];
        nn_sum = nn_sum + nn_size;
    end
    

    
    
    

    %% 梯度继续反向传播
    gK = drelu2(gK,Y.K);
%     gK = batchCentra_backward(gK);
    if normalize
        [gK, dKgamma, dKbeta] = batchnorm_backward(gK, Y.KCache);
    else
        [gK, ~, ~] = batchnorm_backward(gK, Y.KCache);
    end
    
    
    %% 计算I的梯度
    len = sum(n_size);
%     gI = zeros(applicationLayer.seq_dim,len);
    gI = applicationLayer.KW' * gK;
    
    %% 对KW进行梯度下降
    dw = gK * Y.I'/len;
    
    
    adaptive_var.s.vKW = updateS(parameter, dw, adaptive_var.s.vKW);
    if normalize
        adaptive_var.s.vKgamma = updateS(parameter, dKgamma, adaptive_var.s.vKgamma);
        adaptive_var.s.vKbeta = updateS(parameter, dKbeta, adaptive_var.s.vKbeta);    
    end
    
    adaptive_var.V.vKW = updateV(parameter, dw, adaptive_var.V.vKW);
    if normalize
        adaptive_var.V.vKgamma = updateV(parameter, dKgamma, adaptive_var.V.vKgamma);
        adaptive_var.V.vKbeta = updateV(parameter, dKbeta, adaptive_var.V.vKbeta);     
    end
    
    applicationLayer.KW = update_net(parameter, adaptive_var.s.vKW, adaptive_var.V.vKW, global_step, applicationLayer.KW);   
    if normalize
        applicationLayer.Kgamma = update_net(parameter, adaptive_var.s.vKgamma, adaptive_var.V.vKgamma, global_step, applicationLayer.Kgamma);
        applicationLayer.Kbeta = update_net(parameter, adaptive_var.s.vKbeta, adaptive_var.V.vKbeta, global_step, applicationLayer.Kbeta);    
    end
        
    if nolinear
        db = sum(gK,2)/len;
        
        adaptive_var.s.vKb = updateS(parameter, db, adaptive_var.s.vKb);
        adaptive_var.V.vKb = updateV(parameter, db, adaptive_var.V.vKb);
        applicationLayer.Kb = update_net(parameter, adaptive_var.s.vKb, adaptive_var.V.vKb, global_step, applicationLayer.Kb); 
    end 
    
end

