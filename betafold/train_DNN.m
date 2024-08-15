function [dnn, adaptive_var, gI, loss] = train_DNN(dnn, parameter, adaptive_var, Y, label, global_step)
%
    L = size(dnn,2)+1;
    y = Y.Y{L};
    m = size(y,2);
    
    label = [1-label;label];
    loss = sum(sum(-label.*log(y)))/m;
%     loss = gather(loss);



    for i = L : -1 : 2
        if dnn{i-1}.function == "relu2"
            g = drelu2(g,Y.Y{i});
        end
        if dnn{i-1}.function == "relu"
            g = g.*(Y.Y{i} > 0);
        end
        if dnn{i-1}.function == "selu"
            g = dselu(g,Y.Y{i});
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y.Y{i}.*(1 - Y.Y{i});
        end
        if dnn{i-1}.function == "softmax"
            g = Y.Y{i}-label;
            g = g/0.7;
        end
            [g, dgamma, dbeta] = batchnorm_backward(g, Y.Cache{i-1});
%             [g, ~, ~] = batchnorm_backward(g, Y.Cache{i-1});
%             [g, ~, ~] = batchnorm_backward(g, Y.Cache{i-1});
            dw = g*Y.Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;
            adaptive_var.s{i-1}.vW = updateS(parameter, dw, adaptive_var.s{i-1}.vW);
            adaptive_var.s{i-1}.vb = updateS(parameter, db, adaptive_var.s{i-1}.vb);
            adaptive_var.V{i-1}.vW = updateV(parameter, dw, adaptive_var.V{i-1}.vW);
            adaptive_var.V{i-1}.vb = updateV(parameter, db, adaptive_var.V{i-1}.vb);
            
            adaptive_var.s{i-1}.vgamma = updateS(parameter, dgamma, adaptive_var.s{i-1}.vgamma);
            adaptive_var.s{i-1}.vbeta = updateS(parameter, dbeta, adaptive_var.s{i-1}.vbeta);
            adaptive_var.V{i-1}.vgamma = updateV(parameter, dgamma, adaptive_var.V{i-1}.vgamma);
            adaptive_var.V{i-1}.vbeta = updateV(parameter, dbeta, adaptive_var.V{i-1}.vbeta);
            
            
            dnn{i-1}.W = update_net(parameter, adaptive_var.s{i-1}.vW, adaptive_var.V{i-1}.vW, global_step, dnn{i-1}.W); 
            dnn{i-1}.b = update_net(parameter, adaptive_var.s{i-1}.vb, adaptive_var.V{i-1}.vb, global_step, dnn{i-1}.b); 
            dnn{i-1}.gamma = update_net(parameter, adaptive_var.s{i-1}.vgamma, adaptive_var.V{i-1}.vgamma, global_step, dnn{i-1}.gamma);
            dnn{i-1}.beta = update_net(parameter, adaptive_var.s{i-1}.vbeta, adaptive_var.V{i-1}.vbeta, global_step, dnn{i-1}.beta);
    end
    gI = g;

end
