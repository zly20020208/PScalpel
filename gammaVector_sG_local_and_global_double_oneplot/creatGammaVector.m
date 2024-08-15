function [net, adaptive_var] = creatGammaVector(input_dim,mid_dim,layer_num)
%UNTITLED6 此处显示有关此函数的摘要
    %% 创建网络
    [net{1},adaptive_var{1}] = creatFFN(input_dim,mid_dim, true);
    for i=2:layer_num+1
        %[attentionLayer, adaptive_var] = createGATLayer(seq_dim, key_size, v_size, heads, size_per_head, output_size, nolinear, normalize)
        [net{i}, adaptive_var{i}] = createGATLayer(mid_dim, mid_dim, mid_dim, 4, mid_dim/2, mid_dim, true, true);
    end

    
end
