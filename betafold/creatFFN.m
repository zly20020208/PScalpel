function [dnn,adaptive_var] = creatFFN(wide1,wide2, normalize)
    dnn.normalize = normalize;
    dnn.W = randn(wide2,wide1);
    dnn.b = randn(wide2,1);
    
    adaptive_var.V.vW = dnn.W * 0;
    adaptive_var.V.vb = dnn.b * 0;
        
    adaptive_var.s.vW = dnn.W * 0;
    adaptive_var.s.vb = dnn.b * 0;
    
    if normalize
        dnn.gamma = 1;
        dnn.beta = 0;

        adaptive_var.V.vgamma = 0;
        adaptive_var.V.vbeta = 0;

        adaptive_var.s.vgamma = 0;
        adaptive_var.s.vbeta = 0;
    end
end
