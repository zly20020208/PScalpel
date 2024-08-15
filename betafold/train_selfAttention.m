function [attentionLayer, adaptive_var, gI] = train_selfAttention(attentionLayer, parameter, adaptive_var, Y, g, global_step, n_size)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    heads = attentionLayer.heads;
    nolinear = attentionLayer.nolinear;
    normalize = attentionLayer.normalize;%true表示该层需要对输出结果进行标准化
    m = size(g,2);
    
    %% 对OW的进行梯度下降
    g = drelu2(g,Y.O);
    if normalize
        [g, dgamma, dbeta] = batchnorm_backward(g, Y.OCache);
    else
        [g, ~, ~] = batchnorm_backward(g, Y.OCache);
    end
    
    dw = g * Y.B'/m;
    gB = attentionLayer.OW' * g;
    
    adaptive_var.s.vOW = updateS(parameter, dw, adaptive_var.s.vOW);
    if normalize
        adaptive_var.s.vOgamma = updateS(parameter, dgamma, adaptive_var.s.vOgamma);
        adaptive_var.s.vObeta = updateS(parameter, dbeta, adaptive_var.s.vObeta);
    end
    
    
    adaptive_var.V.vOW = updateV(parameter, dw, adaptive_var.V.vOW);
    if normalize
        adaptive_var.V.vOgamma = updateV(parameter, dgamma, adaptive_var.V.vOgamma);
        adaptive_var.V.vObeta = updateV(parameter, dbeta, adaptive_var.V.vObeta); 
    end
       
    
    attentionLayer.OW = update_net(parameter, adaptive_var.s.vOW, adaptive_var.V.vOW, global_step, attentionLayer.OW);
    if normalize
        attentionLayer.Ogamma = update_net(parameter, adaptive_var.s.vOgamma, adaptive_var.V.vOgamma, global_step, attentionLayer.Ogamma);
        attentionLayer.Obeta = update_net(parameter, adaptive_var.s.vObeta, adaptive_var.V.vObeta, global_step, attentionLayer.Obeta);
    end
    
    
    if nolinear
        db = sum(g,2)/m;
        
        adaptive_var.s.vOb = updateS(parameter, db, adaptive_var.s.vOb);
        adaptive_var.V.vOb = updateV(parameter, db, adaptive_var.V.vOb);
        attentionLayer.Ob = update_net(parameter, adaptive_var.s.vOb, adaptive_var.V.vOb, global_step, attentionLayer.Ob);
    end
    
    %% 反向传播正则化gB
    [gB, dgamma, dbeta] = layernorm_backward(gB, Y.BCache);
    adaptive_var.s.vBgamma = updateS(parameter, dgamma, adaptive_var.s.vBgamma);
    adaptive_var.s.vBbeta = updateS(parameter, dbeta, adaptive_var.s.vBbeta);
    adaptive_var.V.vBgamma = updateV(parameter, dgamma, adaptive_var.V.vBgamma);
    adaptive_var.V.vBbeta = updateV(parameter, dbeta, adaptive_var.V.vBbeta); 
    attentionLayer.Bgamma = update_net(parameter, adaptive_var.s.vBgamma, adaptive_var.V.vBgamma, global_step, attentionLayer.Bgamma);
    attentionLayer.Bbeta = update_net(parameter, adaptive_var.s.vBbeta, adaptive_var.V.vBbeta, global_step, attentionLayer.Bbeta);
    
    %% 对多头进行方向传播
%     gV = gpuArray(single(zeros(attentionLayer.v_size,m)));
%     gK = gpuArray(single(zeros(attentionLayer.key_size,m)));
%     gQ = gpuArray(single(zeros(attentionLayer.key_size,m)));
    gV = zeros(attentionLayer.v_size,m);
    gK = zeros(attentionLayer.key_size,m);
    gQ = zeros(attentionLayer.key_size,m);
    for i=1:heads
        %% 计算q,k,v的梯度
        T = gB((i-1)*attentionLayer.size_per_head+1:i*attentionLayer.size_per_head,:);
        q = Y.q{i};
        k = Y.k{i};
        v = Y.v{i};
        gq = [];
        gk = [];
        gv = [];
        start = 1;
        for j=1:length(n_size)
            len = n_size(j);
            qq = q(:,start:start+len-1);
            kk = k(:,start:start+len-1);
            vv = v(:,start:start+len-1);
            gT = T(:,start:start+len-1);
            s = vv*kk';
            gqq = s'*gT;
            gs = gT*qq';
            gkk = (vv'*gs)';
            gvv = gs*kk;
            gq = [gq,gqq];
            gk = [gk,gkk];
            gv = [gv,gvv];
            
            start = start+len;
        end
        
        
        %% 计算V的梯度
        gv = drelu2(gv,Y.v{i});
        if normalize
            [gv, dgamma, dbeta] = batchnorm_backward(gv, Y.vCache{i});
        else
            [gv, ~, ~] = batchnorm_backward(gv, Y.vCache{i});
        end
        gV = gV + attentionLayer.vW{i}' * gv;
        
        %% 对vW进行梯度下降
        dw = gv * Y.V'/m;
        
        adaptive_var.s.vvW{i} = updateS(parameter, dw, adaptive_var.s.vvW{i});
        if normalize
            adaptive_var.s.vvgamma{i} = updateS(parameter, dgamma, adaptive_var.s.vvgamma{i});
            adaptive_var.s.vvbeta{i} = updateS(parameter, dbeta, adaptive_var.s.vvbeta{i});
        end
        
        
        adaptive_var.V.vvW{i} = updateV(parameter, dw, adaptive_var.V.vvW{i});
        if normalize
            adaptive_var.V.vvgamma{i} = updateV(parameter, dgamma, adaptive_var.V.vvgamma{i});
            adaptive_var.V.vvbeta{i} = updateV(parameter, dbeta, adaptive_var.V.vvbeta{i});
        end
        
        
        attentionLayer.vW{i} = update_net(parameter, adaptive_var.s.vvW{i}, adaptive_var.V.vvW{i}, global_step, attentionLayer.vW{i});
        if normalize
            attentionLayer.vgamma{i} = update_net(parameter, adaptive_var.s.vvgamma{i}, adaptive_var.V.vvgamma{i}, global_step, attentionLayer.vgamma{i});
            attentionLayer.vbeta{i} = update_net(parameter, adaptive_var.s.vvbeta{i}, adaptive_var.V.vvbeta{i}, global_step, attentionLayer.vbeta{i});
        end

        
        if nolinear
            db = sum(gv,2)/m;
            
            adaptive_var.s.vvb{i} = updateS(parameter, db, adaptive_var.s.vvb{i});
            adaptive_var.V.vvb{i} = updateV(parameter, db, adaptive_var.V.vvb{i});
            attentionLayer.vb{i} = update_net(parameter, adaptive_var.s.vvb{i}, adaptive_var.V.vvb{i}, global_step, attentionLayer.vb{i});           
        end
        
        %% 计算Q和K的梯度
        gq = drelu2(gq,Y.q{i});
        if normalize
            [gq, dqgamma, dqbeta] = batchnorm_backward(gq, Y.qCache{i});
        else
            [gq, ~, ~] = batchnorm_backward(gq, Y.qCache{i});
        end
        
        gk = drelu2(gk,Y.k{i});
        if normalize
            [gk, dkgamma, dkbeta] = batchnorm_backward(gk, Y.kCache{i});
        else
            [gk, ~, ~] = batchnorm_backward(gk, Y.kCache{i});
        end
        
        
        gQ = gQ + attentionLayer.qW{i}' * gq;
        gK = gK + attentionLayer.kW{i}' * gk;
        
        %% 对qW进行梯度下降
        dw = gq * Y.Q'/m;
        
        adaptive_var.s.vqW{i} = updateS(parameter, dw, adaptive_var.s.vqW{i});
        if normalize
            adaptive_var.s.vqgamma{i} = updateS(parameter, dqgamma, adaptive_var.s.vqgamma{i});
            adaptive_var.s.vqbeta{i} = updateS(parameter, dqbeta, adaptive_var.s.vqbeta{i});
        end
        
        
        adaptive_var.V.vqW{i} = updateV(parameter, dw, adaptive_var.V.vqW{i});
        if normalize
            adaptive_var.V.vqgamma{i} = updateV(parameter, dqgamma, adaptive_var.V.vqgamma{i});
            adaptive_var.V.vqbeta{i} = updateV(parameter, dqbeta, adaptive_var.V.vqbeta{i});
        end
        
        
        attentionLayer.qW{i} = update_net(parameter, adaptive_var.s.vqW{i}, adaptive_var.V.vqW{i}, global_step, attentionLayer.qW{i}); 
        if normalize
            attentionLayer.qgamma{i} = update_net(parameter, adaptive_var.s.vqgamma{i}, adaptive_var.V.vqgamma{i}, global_step, attentionLayer.qgamma{i});
            attentionLayer.qbeta{i} = update_net(parameter, adaptive_var.s.vqbeta{i}, adaptive_var.V.vqbeta{i}, global_step, attentionLayer.qbeta{i});
        end

        
        if nolinear
            db = sum(gq,2)/m;
            
            adaptive_var.s.vqb{i} = updateS(parameter, db, adaptive_var.s.vqb{i});
            adaptive_var.V.vqb{i} = updateV(parameter, db, adaptive_var.V.vqb{i});
            attentionLayer.qb{i} = update_net(parameter, adaptive_var.s.vqb{i}, adaptive_var.V.vqb{i}, global_step, attentionLayer.qb{i});           
        end   
        %% 对kW进行梯度下降
        dw = gk * Y.K'/m;
        
        adaptive_var.s.vkW{i} = updateS(parameter, dw, adaptive_var.s.vkW{i});
        if normalize
            adaptive_var.s.vkgamma{i} = updateS(parameter, dkgamma, adaptive_var.s.vkgamma{i});
            adaptive_var.s.vkbeta{i} = updateS(parameter, dkbeta, adaptive_var.s.vkbeta{i});
        end
        
        
        adaptive_var.V.vkW{i} = updateV(parameter, dw, adaptive_var.V.vkW{i});
        if normalize
            adaptive_var.V.vkgamma{i} = updateV(parameter, dkgamma, adaptive_var.V.vkgamma{i});
            adaptive_var.V.vkbeta{i} = updateV(parameter, dkbeta, adaptive_var.V.vkbeta{i});
        end
        
        
        attentionLayer.kW{i} = update_net(parameter, adaptive_var.s.vkW{i}, adaptive_var.V.vkW{i}, global_step, attentionLayer.kW{i});  
        if normalize
            attentionLayer.kgamma{i} = update_net(parameter, adaptive_var.s.vkgamma{i}, adaptive_var.V.vkgamma{i}, global_step, attentionLayer.kgamma{i});
            attentionLayer.kbeta{i} = update_net(parameter, adaptive_var.s.vkbeta{i}, adaptive_var.V.vkbeta{i}, global_step, attentionLayer.kbeta{i});
        end

        
        if nolinear
            db = sum(gk,2)/m;
            
            adaptive_var.s.vkb{i} = updateS(parameter, db, adaptive_var.s.vkb{i});
            adaptive_var.V.vkb{i} = updateV(parameter, db, adaptive_var.V.vkb{i});
            attentionLayer.kb{i} = update_net(parameter, adaptive_var.s.vkb{i}, adaptive_var.V.vkb{i}, global_step, attentionLayer.kb{i});                  
        end        
        
    end
    %% 计算I的梯度
    gQ = drelu2(gQ,Y.Q);
    gK = drelu2(gK,Y.K);
    gV = drelu2(gV,Y.V);
    if normalize
        [gQ, dQgamma, dQbeta] = batchnorm_backward(gQ, Y.QCache);
        [gK, dKgamma, dKbeta] = batchnorm_backward(gK, Y.KCache);
        [gV, dVgamma, dVbeta] = batchnorm_backward(gV, Y.VCache);
    else
        [gQ, ~, ~] = batchnorm_backward(gQ, Y.QCache);
        [gK, ~, ~] = batchnorm_backward(gK, Y.KCache);
        [gV, ~, ~] = batchnorm_backward(gV, Y.VCache);
    end
    
    
%     gI = gpuArray(single(zeros(attentionLayer.seq_dim,m)));
    gI = zeros(attentionLayer.seq_dim,m);
    gI = gI + attentionLayer.QW' * gQ;
    gI = gI + attentionLayer.KW' * gK;
    gI = gI + attentionLayer.VW' * gV;
    %% 对KW进行梯度下降
    dw = gK * Y.I'/m;
    
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
    
    
    attentionLayer.KW = update_net(parameter, adaptive_var.s.vKW, adaptive_var.V.vKW, global_step, attentionLayer.KW);    
    if normalize
        attentionLayer.Kgamma = update_net(parameter, adaptive_var.s.vKgamma, adaptive_var.V.vKgamma, global_step, attentionLayer.Kgamma);
        attentionLayer.Kbeta = update_net(parameter, adaptive_var.s.vKbeta, adaptive_var.V.vKbeta, global_step, attentionLayer.Kbeta);    
    end
    
        
    if nolinear
        db = sum(gK,2)/m;
        
        adaptive_var.s.vKb = updateS(parameter, db, adaptive_var.s.vKb);
        adaptive_var.V.vKb = updateV(parameter, db, adaptive_var.V.vKb);
        attentionLayer.Kb = update_net(parameter, adaptive_var.s.vKb, adaptive_var.V.vKb, global_step, attentionLayer.Kb); 
    end  
    %% 对QW进行梯度下降
    dw = gQ * Y.I'/m;
    
    adaptive_var.s.vQW = updateS(parameter, dw, adaptive_var.s.vQW);
    if normalize
        adaptive_var.s.vQgamma = updateS(parameter, dQgamma, adaptive_var.s.vQgamma);
        adaptive_var.s.vQbeta = updateS(parameter, dQbeta, adaptive_var.s.vQbeta);    
    end
    
    
    adaptive_var.V.vQW = updateV(parameter, dw, adaptive_var.V.vQW);
    if normalize
        adaptive_var.V.vQgamma = updateV(parameter, dQgamma, adaptive_var.V.vQgamma);
        adaptive_var.V.vQbeta = updateV(parameter, dQbeta, adaptive_var.V.vQbeta);     
    end
    
    
    attentionLayer.QW = update_net(parameter, adaptive_var.s.vQW, adaptive_var.V.vQW, global_step, attentionLayer.QW);  
    if normalize
        attentionLayer.Qgamma = update_net(parameter, adaptive_var.s.vQgamma, adaptive_var.V.vQgamma, global_step, attentionLayer.Qgamma);
        attentionLayer.Qbeta = update_net(parameter, adaptive_var.s.vQbeta, adaptive_var.V.vQbeta, global_step, attentionLayer.Qbeta);    
    end
    
    
    if nolinear
        db = sum(gQ,2)/m;
        
        adaptive_var.s.vQb = updateS(parameter, db, adaptive_var.s.vQb);
        adaptive_var.V.vQb = updateV(parameter, db, adaptive_var.V.vQb);
        attentionLayer.Qb = update_net(parameter, adaptive_var.s.vQb, adaptive_var.V.vQb, global_step, attentionLayer.Qb); 
    end  
    
    
    %% 对VW进行梯度下降
    dw = gV * Y.I'/m;
    
    adaptive_var.s.vVW = updateS(parameter, dw, adaptive_var.s.vVW);
    if normalize
        adaptive_var.s.vVgamma = updateS(parameter, dVgamma, adaptive_var.s.vVgamma);
        adaptive_var.s.vVbeta = updateS(parameter, dVbeta, adaptive_var.s.vVbeta);    
    end
    
    
    adaptive_var.V.vVW = updateV(parameter, dw, adaptive_var.V.vVW);
    if normalize
        adaptive_var.V.vVgamma = updateV(parameter, dVgamma, adaptive_var.V.vVgamma);
        adaptive_var.V.vVbeta = updateV(parameter, dVbeta, adaptive_var.V.vVbeta);     
    end
    
    
    attentionLayer.VW = update_net(parameter, adaptive_var.s.vVW, adaptive_var.V.vVW, global_step, attentionLayer.VW);    
    if normalize
        attentionLayer.Vgamma = update_net(parameter, adaptive_var.s.vVgamma, adaptive_var.V.vVgamma, global_step, attentionLayer.Vgamma);
        attentionLayer.Vbeta = update_net(parameter, adaptive_var.s.vVbeta, adaptive_var.V.vVbeta, global_step, attentionLayer.Vbeta);    
    end
    
    
    if nolinear
        db = sum(gV,2)/m;
        
        adaptive_var.s.vVb = updateS(parameter, db, adaptive_var.s.vVb);
        adaptive_var.V.vVb = updateV(parameter, db, adaptive_var.V.vVb);
        attentionLayer.Vb = update_net(parameter, adaptive_var.s.vVb, adaptive_var.V.vVb, global_step, attentionLayer.Vb); 
    end      
end

