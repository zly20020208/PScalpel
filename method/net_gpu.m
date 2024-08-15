function [net] = net_gpu(net1)
%UNTITLED8 此处显示有关此函数的摘要
    if ~iscell(net1) && ~isstruct(net1)
        net = gpuArray(single(net1));
        return;
    end
    net = [];
    if isstruct(net1)
        fileds = fieldnames(net1); 
        for i=1:length(fileds)
            k = fileds(i);
            key = k{1};
            key = string(key);
            if key=="normalize" || key == "seq_dim" || key == "heads" || key == "key_size" || key == "output_size"|| key == "nolinear"|| key == "use_gpu"|| key == "size_per_head"|| key == "v_size"|| key == "function"
                net.(key) = net1.(key);
                continue;
            end
            net.(key) = net_gpu(net1.(key));
        end        
    else
        len = length(net1);
        for j=1:len
            net{j} = net_gpu(net1{j});
        end
    end
    
    
    

end

