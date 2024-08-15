function [output, Y] = application(applicationLayer, input, n_size, dropout)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%     len = size(input,2);
    Y.I = input;
    
    K = applicationLayer.KW * input;
    if applicationLayer.nolinear
        K = K + applicationLayer.Kb;
    end
    if applicationLayer.normalize
        [K,Y.KCache] = batchnorm_forward(K,applicationLayer.Kgamma,applicationLayer.Kbeta);
    else
        [K,Y.KCache] = batchnorm_forward(K,1,0);
    end
    K = relu2(K);
    K = Dropout(K,dropout);
    Y.K = K;
    
    sum_nn = sum(n_size.*(n_size-1)/2);
    if applicationLayer.use_gpu
        T = gpuArray(single(zeros(applicationLayer.key_size*2,sum_nn)));
    else
        T = zeros(applicationLayer.key_size*2,sum_nn);
    end
    sum_nn=0;
    start = 0;
    for j=1:length(n_size)
        len = n_size(j);
        t=0;
        for i=1:len-1
            x = [repmat(K(:,start+i),1,len-i);K(:,start+i+1:start+len)];
            T(:,sum_nn+1+t:sum_nn+len-i+t) = x;
            t = t + len - i;
        end        
        start = start + len;
        sum_nn = sum_nn + len*(len-1)/2;
    end

    [y, Y.Y] = DNN(applicationLayer.dnn,T,dropout);
%     y = y(2,:)>0.6235;
%     y = y(2,:);
%     y = gather(y);
%     output = vec2mat(y,len);
    output = y(2,:);
%     output = gather(y(2,:));
    
    
    
    

    

end

