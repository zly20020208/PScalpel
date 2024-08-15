function [net] = net_add(net1,net2)
%UNTITLED8 此处显示有关此函数的摘要
    if ~iscell(net1) && ~isstruct(net1)
        net = net1 + net2;
        return;
    end
    net = [];
    if isstruct(net1)
        fileds = fieldnames(net1); 
        for i=1:length(fileds)
            k = fileds(i);
            key = k{1};
            key = string(key);
            if key=="normalize" || key == "seq_dim" || key == "heads" || key == "key_size" || key == "output_size"|| key == "size_per_head"|| key == "v_size"|| key == "nolinear"
                net.(key) = net1.(key);
                continue;
            end
            net.(key) = net_add(net1.(key),net2.(key));
        end        
    else
        len = length(net1);
        for j=1:len
            net{j} = net_add(net1{j},net2{j});
        end
    end
    
    
    

end

