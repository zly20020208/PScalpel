function [net, adv] = creatBlock(wide, normalize)
%UNTITLED6 此处显示有关此函数的摘要
    %% 创建网络
    net.normalize = normalize;
    % [attentionLayer, adaptive_var] = createAttentionLayer(seq_dim, key_size, v_size, heads, size_per_head, output_size, nolinear)
    [net.layers{1},adv.adaptive_var{1}] = createAttentionLayer(wide, wide, wide, 8, wide/8, wide, true, normalize); 
    
    %[FFN,adaptive_var] = creatFFN(len)
    [net.layers{2},adv.adaptive_var{2}] = creatFFN(wide,wide,normalize);
    
    if normalize
        net.gamma{1} = 1;
        net.beta{1} = 0;

        adv.V.vgamma{1} = 0;
        adv.V.vbeta{1} = 0;
        adv.s.vgamma{1} = 0;
        adv.s.vbeta{1} = 0;

        net.gamma{2} = 1;
        net.beta{2} = 0;

        adv.V.vgamma{2} = 0;
        adv.V.vbeta{2} = 0;
        adv.s.vgamma{2} = 0;
        adv.s.vbeta{2} = 0;
    end
    

    
end
