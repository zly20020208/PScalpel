function [dnn, adaptive_var, gI] = train_FNN(dnn, parameter, adaptive_var, Y, g, global_step)
    normalize = dnn.normalize;

    y = Y.Y{2};
    m = size(y,2);
%     g = g.*(y > 0);
    g = drelu2(g,y);
    if normalize
        [g, dgamma, dbeta] = batchnorm_backward(g, Y.Cache);
    else
        [g, ~, ~] = batchnorm_backward(g, Y.Cache);
    end
    
    dw = g*Y.Y{1}.'/m;
    db = sum(g,2)/m;
    
    gI = dnn.W'*g;
    
    adaptive_var.s.vW = parameter.beta1*adaptive_var.s.vW + (1 - parameter.beta1)*dw; 
    adaptive_var.s.vb = parameter.beta1*adaptive_var.s.vb + (1 - parameter.beta1)*db;
    adaptive_var.V.vW = max(parameter.beta2*adaptive_var.V.vW + (1 - parameter.beta2)*dw.*dw,adaptive_var.V.vW); 
    adaptive_var.V.vb = max(parameter.beta2*adaptive_var.V.vb + (1 - parameter.beta2)*db.*db,adaptive_var.V.vb);
    
    if normalize
        adaptive_var.s.vgamma = parameter.beta1*adaptive_var.s.vgamma + (1 - parameter.beta1)*dgamma; 
        adaptive_var.s.vbeta = parameter.beta1*adaptive_var.s.vbeta + (1 - parameter.beta1)*dbeta;
        adaptive_var.V.vgamma = max(parameter.beta2*adaptive_var.V.vgamma + (1 - parameter.beta2)*dgamma.*dgamma,adaptive_var.V.vgamma); 
        adaptive_var.V.vbeta = max(parameter.beta2*adaptive_var.V.vbeta + (1 - parameter.beta2)*dbeta.*dbeta,adaptive_var.V.vbeta);
    end


    dnn.W = dnn.W - parameter.learning_rate*(adaptive_var.s.vW/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V.vW./(1 - parameter.beta2.^global_step)));
    dnn.b = dnn.b - parameter.learning_rate*(adaptive_var.s.vb/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V.vb./(1 - parameter.beta2.^global_step)));
    if normalize
        dnn.gamma = dnn.gamma - parameter.learning_rate*(adaptive_var.s.vgamma/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V.vgamma./(1 - parameter.beta2.^global_step)));
        dnn.beta = dnn.beta - parameter.learning_rate*(adaptive_var.s.vbeta/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V.vbeta./(1 - parameter.beta2.^global_step)));
    end

end
