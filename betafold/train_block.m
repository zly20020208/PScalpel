function [net, adv, gI] = train_block(net, parameter, adv, Y, g, global_step, n_size)
    normalize = net.normalize;
    %% 反向传播，对FNN进行训练
    %[dnn, adaptive_var, gI] = train_FNN(dnn, parameter, adaptive_var, Y, g, global_step)
    [net.layers{2}, adv.adaptive_var{2}, gI] = train_FNN(net.layers{2}, parameter, adv.adaptive_var{2}, Y.Y{2}, g, global_step);
    if normalize
        [gI, dgamma, dbeta] = layernorm_backward(gI, Y.Cache{2});
    else
        [gI, ~, ~] = layernorm_backward(gI, Y.Cache{2});
    end
    g = gI + g;

    %% 修改gamma和beta
    if normalize
        adv.s.vgamma{2} = updateS(parameter, dgamma, adv.s.vgamma{2});
        adv.s.vbeta{2} = updateS(parameter, dbeta, adv.s.vbeta{2});

        adv.V.vgamma{2} = updateV(parameter, dgamma, adv.V.vgamma{2});
        adv.V.vbeta{2} = updateV(parameter, dbeta, adv.V.vbeta{2}); 

        net.gamma{2} = update_net(parameter, adv.s.vgamma{2}, adv.V.vgamma{2}, global_step, net.gamma{2});
        net.beta{2} = update_net(parameter, adv.s.vbeta{2}, adv.V.vbeta{2}, global_step, net.beta{2});
    end
    
    
    %% 对attentionLayer进行训练
    % [attentionLayer, adaptive_var, gI] = train_selfAttention(attentionLayer, parameter, adaptive_var, Y, g, global_step)
    [net.layers{1}, adv.adaptive_var{1}, gI] = train_selfAttention(net.layers{1}, parameter, adv.adaptive_var{1}, Y.Y{1}, g, global_step, n_size);
    if normalize
        [gI, dgamma, dbeta] = layernorm_backward(gI, Y.Cache{1});
    else
        [gI, ~, ~] = layernorm_backward(gI, Y.Cache{2});
    end
    gI = gI + g;
    
    
    %% 修改gamma和beta
    if normalize
        adv.s.vgamma{1} = updateS(parameter, dgamma, adv.s.vgamma{1});
        adv.s.vbeta{1} = updateS(parameter, dbeta, adv.s.vbeta{1});

        adv.V.vgamma{1} = updateV(parameter, dgamma, adv.V.vgamma{1});
        adv.V.vbeta{1} = updateV(parameter, dbeta, adv.V.vbeta{1}); 

        net.gamma{1} = update_net(parameter, adv.s.vgamma{1}, adv.V.vgamma{1}, global_step, net.gamma{1});
        net.beta{1} = update_net(parameter, adv.s.vbeta{1}, adv.V.vbeta{1}, global_step, net.beta{1});
    end
end
