function [net, adaptive_var, loss] = backprop(x, label, n_size, net, parameter, adaptive_var, global_step)
    L = size(net,2);
    [~, Y] = forwordprop(net,x,n_size,parameter.dropout);

    %% 反向传播，对applicationLayer进行训练
    %[applicationLayer, adaptive_var, gI, loss] = train_application(applicationLayer, parameter, adaptive_var, Y, label, global_step)
    [net{L}, adaptive_var{L}, gI, loss] = train_application(net{L}, parameter, adaptive_var{L}, Y{L}, label, global_step, n_size);

    
    %% 对block进行训练
    for i=L-1:-1:2
        % [net, adv, gI] = train_block(net, parameter, adv, Y, g, global_step)
        [net{i}, adaptive_var{i}, gI] = train_block(net{i}, parameter, adaptive_var{i}, Y{i}, gI, global_step, n_size);
    end
    %% 对FNN进行训练
    %[dnn, adaptive_var, gI] = train_FNN(dnn, parameter, adaptive_var, Y, g, global_step)
    [net{1}, adaptive_var{1}, ~] = train_FNN(net{1}, parameter, adaptive_var{1}, Y{1}, gI, global_step);
end
