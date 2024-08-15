function [dnn,adaptive_var] = creatDNN(K)
    L = size(K.a,2);
    for i = 1:L-1
        dnn{i}.W = randn(K.a(i+1),K.a(i));
        dnn{i}.function = K.f{i};
        dnn{i}.b = randn(K.a(i+1),1);
        dnn{i}.gamma = 1;
        dnn{i}.beta = 0;
        
        adaptive_var.V{i}.vW = dnn{i}.W * 0;
        adaptive_var.V{i}.vb = dnn{i}.b * 0;
        adaptive_var.V{i}.vgamma = 0;
        adaptive_var.V{i}.vbeta = 0;
        
        adaptive_var.s{i}.vW = dnn{i}.W * 0;
        adaptive_var.s{i}.vb = dnn{i}.b * 0;
        adaptive_var.s{i}.vgamma = 0;
        adaptive_var.s{i}.vbeta = 0;
        
%         if use_GPU
%             dnn{i}.W = gpuArray(single(dnn{i}.W));
%             dnn{i}.b = gpuArray(single(dnn{i}.b));
%             dnn{i}.gamma = gpuArray(single(dnn{i}.gamma));
%             dnn{i}.beta = gpuArray(single(dnn{i}.beta));
%             
%             adaptive_var.V{i}.vW = gpuArray(single(adaptive_var.V{i}.vW));
%             adaptive_var.V{i}.vb = gpuArray(single(adaptive_var.V{i}.vb));
%             adaptive_var.V{i}.vgamma = gpuArray(single(adaptive_var.V{i}.vgamma));
%             adaptive_var.V{i}.vbeta = gpuArray(single(adaptive_var.V{i}.vbeta));
%             
%             adaptive_var.s{i}.vW = gpuArray(single(adaptive_var.s{i}.vW));
%             adaptive_var.s{i}.vb = gpuArray(single(adaptive_var.s{i}.vb));
%             adaptive_var.s{i}.vgamma = gpuArray(single(adaptive_var.s{i}.vgamma));
%             adaptive_var.s{i}.vbeta = gpuArray(single(adaptive_var.s{i}.vbeta));
%             
%         end
    end
end
