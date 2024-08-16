function [net,adaptive_var] = creatDNN(K)
    L = size(K.a,2);
    for i = 1:L-1
        net.dnn{i}.W = randn(K.a(i+1),K.a(i));
        net.dnn{i}.function = K.f{i};
        net.dnn{i}.b = randn(K.a(i+1),1);
        net.dnn{i}.gamma = 1;
        net.dnn{i}.beta = 0;
        
        net.dnn{i}.mu = nan;
        net.dnn{i}.var = nan;
        
        adaptive_var.V{i}.vW = net.dnn{i}.W * 0;
        adaptive_var.V{i}.vb = net.dnn{i}.b * 0;
        adaptive_var.V{i}.vgamma = 0;
        adaptive_var.V{i}.vbeta = 0;
        
        adaptive_var.s{i}.vW = net.dnn{i}.W * 0;
        adaptive_var.s{i}.vb = net.dnn{i}.b * 0;
        adaptive_var.s{i}.vgamma = 0;
        adaptive_var.s{i}.vbeta = 0;
    end
end
