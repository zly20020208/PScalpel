function [attentionLayer, adaptive_var] = createAttentionLayer(seq_dim, key_size, v_size, heads, size_per_head, output_size, nolinear, normalize)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    attentionLayer.normalize = normalize;%true表示该层需要对输出结果进行标准化
    attentionLayer.seq_dim = seq_dim;%输入序列的特征维度
    attentionLayer.heads = heads;%多头自注意力机制的头数
    attentionLayer.size_per_head = size_per_head;%中间层，每一头的特征维度
    attentionLayer.key_size = key_size;%K的特征维度
    attentionLayer.v_size = v_size;%V的特征维度
    attentionLayer.nolinear = nolinear;%true表示该层加入了偏置，可以进行非线性变换
    
    attentionLayer.QW = randn(key_size,seq_dim);
    if normalize
        attentionLayer.Qgamma = 1;
        attentionLayer.Qbeta = 0;
    end
    
    
    attentionLayer.KW = randn(key_size,seq_dim);
    if normalize
        attentionLayer.Kgamma = 1;
        attentionLayer.Kbeta = 0;
    end
    
    
    attentionLayer.VW = randn(v_size,seq_dim);
    if normalize
        attentionLayer.Vgamma = 1;
        attentionLayer.Vbeta = 0; 
    end
       
    
    attentionLayer.OW = randn(output_size,heads*size_per_head);
    if normalize
        attentionLayer.Ogamma = 1;
        attentionLayer.Obeta = 0;
        attentionLayer.Bgamma = 1;
        attentionLayer.Bbeta = 0;
    end
         
    
    
    if nolinear
        attentionLayer.Qb = randn(key_size,1);
        attentionLayer.Kb = randn(key_size,1);
        attentionLayer.Vb = randn(v_size,1);
        attentionLayer.Ob = randn(output_size,1);
    end
    for i=1:heads
        attentionLayer.vW{i} = randn(size_per_head,v_size);
        if normalize
            attentionLayer.vgamma{i} = 1;
            attentionLayer.vbeta{i} = 0; 
        end
        
        
        %这里默认分出多头后的q,k的size和v的size是一样的
        attentionLayer.qW{i} = randn(size_per_head,key_size);
        if normalize
            attentionLayer.qgamma{i} = 1;
            attentionLayer.qbeta{i} = 0; 
        end
        
        
        attentionLayer.kW{i} = randn(size_per_head,key_size);
        if normalize
            attentionLayer.kgamma{i} = 1;
            attentionLayer.kbeta{i} = 0; 
        end
        
        if nolinear
            attentionLayer.vb{i} = randn(size_per_head,1);
            attentionLayer.qb{i} = randn(size_per_head,1);
            attentionLayer.kb{i} = randn(size_per_head,1);
        end
    end
    

    
    %% 定义自适应变量
    %定义V
    adaptive_var.V.vQW = attentionLayer.QW * 0;
    if normalize
        adaptive_var.V.vQgamma = 0;
        adaptive_var.V.vQbeta = 0;
    end
    
    
    adaptive_var.V.vKW = attentionLayer.KW * 0;
    if normalize
        adaptive_var.V.vKgamma = 0;
        adaptive_var.V.vKbeta = 0;
    end
    
    
    adaptive_var.V.vVW = attentionLayer.VW * 0;
    if normalize
        adaptive_var.V.vVgamma = 0;
        adaptive_var.V.vVbeta = 0;
    end
    
    
    adaptive_var.V.vOW = attentionLayer.OW * 0;
    if normalize
        adaptive_var.V.vOgamma = 0;
        adaptive_var.V.vObeta = 0;
        adaptive_var.V.vBgamma = 0;
        adaptive_var.V.vBbeta = 0;
    end
    
    if nolinear
        adaptive_var.V.vQb = attentionLayer.Qb * 0;
        adaptive_var.V.vKb = attentionLayer.Kb * 0;
        adaptive_var.V.vVb = attentionLayer.Vb * 0;
        adaptive_var.V.vOb = attentionLayer.Ob * 0;
    end
    for i = 1:heads
        adaptive_var.V.vqW{i} = attentionLayer.qW{i}*0;
        if normalize
            adaptive_var.V.vqgamma{i} = 0;
            adaptive_var.V.vqbeta{i} = 0;
        end
        
        
        adaptive_var.V.vkW{i} = attentionLayer.kW{i}*0;
        if normalize
            adaptive_var.V.vkgamma{i} = 0;
            adaptive_var.V.vkbeta{i} = 0;
        end
        
        
        adaptive_var.V.vvW{i} = attentionLayer.vW{i}*0;
        if normalize
            adaptive_var.V.vvgamma{i} = 0;
            adaptive_var.V.vvbeta{i} = 0;
        end
        
        if nolinear
            adaptive_var.V.vqb{i} = attentionLayer.qb{i} * 0;
            adaptive_var.V.vkb{i} = attentionLayer.kb{i} * 0;
            adaptive_var.V.vvb{i} = attentionLayer.vb{i} * 0;
        end
    end
    
    %定义s
    adaptive_var.s.vQW = attentionLayer.QW * 0;
    if normalize
        adaptive_var.s.vQgamma = 0;
        adaptive_var.s.vQbeta = 0;
    end
    
            
    adaptive_var.s.vKW = attentionLayer.KW * 0;
    if normalize
        adaptive_var.s.vKgamma = 0;
        adaptive_var.s.vKbeta = 0;
    end
    
    
    adaptive_var.s.vVW = attentionLayer.VW * 0;
    if normalize
        adaptive_var.s.vVgamma = 0;
        adaptive_var.s.vVbeta = 0;
    end
    
    
    adaptive_var.s.vOW = attentionLayer.OW * 0;
    if normalize
        adaptive_var.s.vOgamma = 0;
        adaptive_var.s.vObeta = 0;
        adaptive_var.s.vBgamma = 0;
        adaptive_var.s.vBbeta = 0;
    end
    
    if nolinear
        adaptive_var.s.vQb = attentionLayer.Qb * 0;
        adaptive_var.s.vKb = attentionLayer.Kb * 0;
        adaptive_var.s.vVb = attentionLayer.Vb * 0;
        adaptive_var.s.vOb = attentionLayer.Ob * 0;
    end
    for i = 1:heads
        adaptive_var.s.vqW{i} = attentionLayer.qW{i}*0;
        if normalize
            adaptive_var.s.vqgamma{i} = 0;
            adaptive_var.s.vqbeta{i} = 0;
        end
        
        
        adaptive_var.s.vkW{i} = attentionLayer.kW{i}*0;
        if normalize
            adaptive_var.s.vkgamma{i} = 0;
            adaptive_var.s.vkbeta{i} = 0;
        end
        
        
        adaptive_var.s.vvW{i} = attentionLayer.vW{i}*0;
        if normalize
            adaptive_var.s.vvgamma{i} = 0;
            adaptive_var.s.vvbeta{i} = 0;
        end
        
        if nolinear
            adaptive_var.s.vqb{i} = attentionLayer.qb{i} * 0;
            adaptive_var.s.vkb{i} = attentionLayer.kb{i} * 0;
            adaptive_var.s.vvb{i} = attentionLayer.vb{i} * 0;
        end
    end

end

