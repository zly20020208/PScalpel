function [net, adaptive_var, loss] = backprop(x, label, net, parameter, adaptive_var, global_step)

    [~, Y] = forwordprop(net,x,parameter.dropout);
    %% 反向传播，对DNNLayer进行训练
    %[dnn, adaptive_var, gI, loss] = train_DNN(dnn, parameter, adaptive_var, Y, label, global_step)
    [net, adaptive_var, loss] = train_DNN(net, parameter, adaptive_var, Y, label, global_step);
end
