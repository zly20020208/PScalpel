function [net, adaptive_var, loss1, loss2] = backprop(data, net, parameter, adaptive_var, global_step, G)
    [Y, result] = forwordprop(net,data,G,parameter.dropout);
    len = length(data.clinVar);
    %% 计算clinVar的损失
    sem = [];
    model_b = sqrt(sumsqr(result.clinVar{len}));
    g1 = [];
    g2 = [];
    for i=1:len-1
        ab = dot(result.clinVar{i},result.clinVar{len});
        model_a = sqrt(sumsqr(result.clinVar{i}));
        sem(i) = ab/(model_a*model_b);
        g1{i} = result.clinVar{len}/(model_a*model_b)-result.clinVar{i}*ab/(model_a^3*model_b);
        g2{i} = result.clinVar{i}/(model_b*model_a)-result.clinVar{len}*ab/(model_b^3*model_a);
        darcos = 1/sqrt(1-sem(i)^2);
        g1{i} = g1{i} * darcos;
        g2{i} = g2{i} * darcos;
    end
    sem = sem/parameter.t;
    sem = softmax(sem');
    loss1 = -log(sem(1));
    
    g_clinVar = [];
    
    g_clinVar{1} = g1{1} * (sem(1)-1);
    g_clinVar{len} = g2{1} * (sem(1)-1);
    for i=2:len-1
        g_clinVar{i} = g1{i} * sem(i);
        g_clinVar{len} = g2{i} * sem(i) + g_clinVar{len};
    end
    g_clinVar{len} = g_clinVar{len}/(len-1);
    
    %% 计算rate用于保证clinVar和func梯度的平衡
    rate1 = 0;
    for i=1:len
        rate1 = rate1 + sqrt(sumsqr(g_clinVar{i}));
    end
    rate1 = rate1 / len;
    
    
    
    
   %% 计算func的损失
    sem = [];
    model_b = sqrt(sumsqr(result.func{len}));
    g1 = [];
    g2 = [];
    for i=1:len-1
        ab = dot(result.func{i},result.func{len});
        model_a = sqrt(sumsqr(result.func{i}));
        sem(i) = ab/(model_a*model_b);
        g1{i} = result.func{len}/(model_a*model_b)-result.func{i}*ab/(model_a^3*model_b);
        g2{i} = result.func{i}/(model_b*model_a)-result.func{len}*ab/(model_b^3*model_a);
        darcos = 1/sqrt(1-sem(i)^2);
        g1{i} = g1{i} * darcos;
        g2{i} = g2{i} * darcos;
    end
    sem = softmax(sem');
    loss2 = -log(sem(1));
    
    g_func = [];
    
    g_func{1} = g1{1} * (sem(1)-1);
    g_func{len} = g2{1} * (sem(1)-1);
    for i=2:len-1
        g_func{i} = g1{i} * sem(i);
        g_func{len} = g2{i} * sem(i) + g_func{len};
    end
    g_func{len} = g_func{len}/(len-1);
    
    rate2 = 0;
    for i=1:len
        rate2 = rate2 + sqrt(sumsqr(g_func{i}));
    end
    rate2 = rate2 / len;
    
    rate = rate1/rate2*parameter.rate;
    for i=1:len
        g_func{i} = g_func{i} * rate;
    end

    clinVar_G = G.clinVar;
    func_G = G.func;
    %% 训练模型
    %% [net, adaptive_var, gI] = train_gammaVector(net, parameter, adaptive_var, Y, g, global_step, G)
    for i=1:len-1
        [t_net, t_ada_var, ~] = train_gammaVector(net, parameter, adaptive_var, Y.clinVar{i}.Y, g_clinVar{i}, global_step, clinVar_G{i});
        if i==1
            sum_net = t_net;
            sum_adv = t_ada_var;
        else
            sum_net = net_add(sum_net,t_net);
            sum_adv = net_add(sum_adv,t_ada_var);
        end
        [t_net, t_ada_var, ~] = train_gammaVector(net, parameter, adaptive_var, Y.func{i}.Y, g_func{i}, global_step, func_G{i});
        sum_net = net_add(sum_net,t_net);
        sum_adv = net_add(sum_adv,t_ada_var);
    end
    sum_net = net_divide(sum_net,(len-1)*2);
    sum_adv = net_divide(sum_adv,(len-1)*2);
    
    [sum_net1, sum_adv1, ~] = train_gammaVector(net, parameter, adaptive_var, Y.clinVar{len}.Y, g_clinVar{len}, global_step, clinVar_G{len});
    [t_net, t_ada_var, ~] = train_gammaVector(net, parameter, adaptive_var, Y.func{len}.Y, g_func{len}, global_step, func_G{len});
    
    sum_net1 = net_add(sum_net1,t_net);
    sum_adv1 = net_add(sum_adv1,t_ada_var);
    sum_net1 = net_divide(sum_net1,2);
    sum_adv1 = net_divide(sum_adv1,2);
    
    
    net = net_add(sum_net,sum_net1);
    adaptive_var = net_add(sum_adv,sum_adv1);
    net = net_divide(net,2);
    adaptive_var = net_divide(adaptive_var,2);
    
    
end
