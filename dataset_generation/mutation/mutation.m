function [mutation_seq] = mutation(origin_seq,protein_change)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    mutation_seq = char(origin_seq);
    changes = split(protein_change,', ');
    pat = '(?<from>[a-z]+)(?<sit>\d+)(?<to>[a-z]+)';
    len = length(changes);
    for i=1:len
        result = regexpi(changes(i),pat,'names');
        sit = str2double(result.sit);
        if mutation_seq(sit)==result.from
            if strcmp(result.to,"del")
                mutation_seq(sit) = '';
            else
                mutation_seq(sit) = result.to;
            end
            
        end
    end
    
    if mutation_seq==origin_seq
       origin_seq
       protein_change
    end
    
end

