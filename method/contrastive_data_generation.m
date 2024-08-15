function [names, sequences] = contrastive_data_generation(protein_change,gene_dict)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    sequences = [];
    names = [];
    origin_seq = gene_dict.(protein_change(1));
    len = length(protein_change);
    for i=2:len
        mutation_seq = mutation(origin_seq,protein_change(i));
        if mutation_seq~="error"
            sequences = [sequences, string(mutation_seq)];
            change = replace(protein_change(i),', ','');
            change = replace(change,'*','');
            names = [names, string([char(protein_change(1)),'_',char(change)])];
            continue;
        end
        
        if i==2
           sequences = "error";
           return; 
        end
    end
    sequences = [sequences, string(origin_seq)];
    names = [names, protein_change(1)];
end

