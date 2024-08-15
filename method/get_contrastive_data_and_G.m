function [data,G] = get_contrastive_data_and_G(protein_change, data_dict, G_dict)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
    origin_name = char(protein_change(1));
    len = length(protein_change);
    j=1;
    data = [];
    G = [];
    for i=2:len
        change = replace(protein_change(i),', ','');
        change = replace(change,'*','');
        name = [origin_name,'_',char(change)];
        if length(name)>63
            name = name(1:63);
        end
        if isfield(data_dict,name)
            data{j} = data_dict.(name);
            n_size = size(data{j},2);
            G{j} = vec2mat(G_dict.(name),n_size);
            j = j + 1;
        else
            data = [];
            G = [];
            return;
        end        
    end
    data{j} = data_dict.(origin_name);
    n_size = size(data{j},2);
    G{j} = vec2mat(G_dict.(origin_name),n_size);
end

