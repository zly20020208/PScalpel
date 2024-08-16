function [net, adaptive_var, loss] = train_DNN(net, parameter, adaptive_var, Y, label, global_step)
%
    dnn = net.dnn;
    L = size(dnn,2)+1;
    y = Y.Y{L};
    m = size(y,2);
    
    loss = sum(sum(-label.*log(y)))/m;

    
    for i = L : -1 : 2
     
        if dnn{i-1}.function == "relu"
            g = g.*(Y.Y{i} > 0);
        end
        if dnn{i-1}.function == "selu"
            g = dselu(g,Y.Y{i});
        end
        if dnn{i-1}.function == "tanh"
            g = dtanh(g,Y.Y{i});
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y.Y{i}.*(1 - Y.Y{i});
        end
        if dnn{i-1}.function == "softmax"
            g = Y.Y{i}-label;
        end
            [g, dgamma, dbeta] = batchnorm_backward(g, Y.Cache{i-1});
            
            if isnan(dnn{i-1}.mu)
               dnn{i-1}.mu = Y.Cache{i-1}.mu;
               dnn{i-1}.var = Y.Cache{i-1}.var;
            else
               dnn{i-1}.mu = (1-parameter.beta1) * Y.Cache{i-1}.mu + parameter.beta1 * dnn{i-1}.mu;
               dnn{i-1}.var = (1-parameter.beta1) * Y.Cache{i-1}.var + parameter.beta1 * dnn{i-1}.var;
            end
            
            dw = g*Y.Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;

            adaptive_var.s{i-1}.vW = parameter.beta1*adaptive_var.s{i-1}.vW + (1 - parameter.beta1)*dw; 
            adaptive_var.s{i-1}.vb = parameter.beta1*adaptive_var.s{i-1}.vb + (1 - parameter.beta1)*db;
            adaptive_var.V{i-1}.vW = max(parameter.beta2*adaptive_var.V{i-1}.vW + (1 - parameter.beta2)*dw.*dw,adaptive_var.V{i-1}.vW); 
            adaptive_var.V{i-1}.vb = max(parameter.beta2*adaptive_var.V{i-1}.vb + (1 - parameter.beta2)*db.*db,adaptive_var.V{i-1}.vb);
            
            adaptive_var.s{i-1}.vgamma = parameter.beta1*adaptive_var.s{i-1}.vgamma + (1 - parameter.beta1)*dgamma; 
            adaptive_var.s{i-1}.vbeta = parameter.beta1*adaptive_var.s{i-1}.vbeta + (1 - parameter.beta1)*dbeta;
            adaptive_var.V{i-1}.vgamma = max(parameter.beta2*adaptive_var.V{i-1}.vgamma + (1 - parameter.beta2)*dgamma.*dgamma,adaptive_var.V{i-1}.vgamma); 
            adaptive_var.V{i-1}.vbeta = max(parameter.beta2*adaptive_var.V{i-1}.vbeta + (1 - parameter.beta2)*dbeta.*dbeta,adaptive_var.V{i-1}.vbeta);
            
            
            dnn{i-1}.W = dnn{i-1}.W - parameter.learning_rate*(adaptive_var.s{i-1}.vW/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V{i-1}.vW./(1 - parameter.beta2.^global_step)));
            dnn{i-1}.b = dnn{i-1}.b - parameter.learning_rate*(adaptive_var.s{i-1}.vb/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V{i-1}.vb./(1 - parameter.beta2.^global_step)));
            dnn{i-1}.gamma = dnn{i-1}.gamma - parameter.learning_rate*(adaptive_var.s{i-1}.vgamma/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V{i-1}.vgamma./(1 - parameter.beta2.^global_step)));
            dnn{i-1}.beta = dnn{i-1}.beta - parameter.learning_rate*(adaptive_var.s{i-1}.vbeta/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V{i-1}.vbeta./(1 - parameter.beta2.^global_step)));
     
    end
    net.dnn = dnn;

end
