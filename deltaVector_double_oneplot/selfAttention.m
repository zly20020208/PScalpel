function [output, Y] = selfAttention(attentionLayer, input, n_size, dropout)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    normalize = attentionLayer.normalize;
    Y.I = input;
    Q = attentionLayer.QW * input;
    K = attentionLayer.KW * input;
    V = attentionLayer.VW * input;
    if attentionLayer.nolinear
        Q = Q + attentionLayer.Qb;
        K = K + attentionLayer.Kb;
        V = V + attentionLayer.Vb;
    end
    if normalize
        [Q,Y.QCache] = batchnorm_forward(Q,attentionLayer.Qgamma,attentionLayer.Qbeta);
        [K,Y.KCache] = batchnorm_forward(K,attentionLayer.Kgamma,attentionLayer.Kbeta);
        [V,Y.VCache] = batchnorm_forward(V,attentionLayer.Vgamma,attentionLayer.Vbeta);
    else
        [Q,Y.QCache] = batchnorm_forward(Q,1,0);
        [K,Y.KCache] = batchnorm_forward(K,1,0);
        [V,Y.VCache] = batchnorm_forward(V,1,0);
    end
    Q = selu(Q);
    K = selu(K);
    V = selu(V);
    Q = Dropout(Q,dropout);
    K = Dropout(K,dropout);
    V = Dropout(V,dropout);
    Y.Q = Q;
    Y.K = K;
    Y.V = V;
    B = [];
    for i=1:attentionLayer.heads
        q = attentionLayer.qW{i} * Q;
        k = attentionLayer.kW{i} * K;
        v = attentionLayer.vW{i} * V;
        if attentionLayer.nolinear
            q = q + attentionLayer.qb{i};
            k = k + attentionLayer.kb{i};
            v = v + attentionLayer.vb{i};
        end
        if normalize
            [q,Y.qCache{i}] = batchnorm_forward(q,attentionLayer.qgamma{i},attentionLayer.qbeta{i});
            [k,Y.kCache{i}] = batchnorm_forward(k,attentionLayer.kgamma{i},attentionLayer.kbeta{i});
            [v,Y.vCache{i}] = batchnorm_forward(v,attentionLayer.vgamma{i},attentionLayer.vbeta{i});
        else
            [q,Y.qCache{i}] = batchnorm_forward(q,1,0);
            [k,Y.kCache{i}] = batchnorm_forward(k,1,0);
            [v,Y.vCache{i}] = batchnorm_forward(v,1,0);
        end
        q = selu(q);
        k = selu(k);
        v = selu(v);
        q = Dropout(q,dropout);
        k = Dropout(k,dropout);
        v = Dropout(v,dropout);
        
        T = [];
        start = 1;
        for j=1:length(n_size)
            len = n_size(j);
            qq = q(:,start:start+len-1);
            kk = k(:,start:start+len-1);
            vv = v(:,start:start+len-1);
%             s = [s, vv*kk'];
            T = [T, vv*kk'*qq];
            start = start+len;
        end
        B = [B;T];
        
        Y.q{i} = q;
        Y.k{i} = k;
        Y.v{i} = v;
    end
    [B,Y.BCache] = layernorm_forward(B,attentionLayer.Bgamma,attentionLayer.Bbeta);
    Y.B = B;
    output = attentionLayer.OW * B;
    if attentionLayer.nolinear
        output = output + attentionLayer.Ob;
    end
    if normalize
        [output,Y.OCache] = batchnorm_forward(output,attentionLayer.Ogamma,attentionLayer.Obeta);
    else
        [output,Y.OCache] = batchnorm_forward(output,1,0);
    end
    output = selu(output);
    
    output = Dropout(output,dropout);
    Y.O = output;
    

end

