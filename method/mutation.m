function [mutation_seq] = mutation(origin_seq,protein_change)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    mutation_seq = char(origin_seq);
    changes = split(protein_change,', ');
    pat = '(?<from>[a-z]+)(?<sit>\d+)(?<to>[a-z*]+)';
    len = length(changes);
    for i=1:len
        result = regexpi(changes(i),pat,'names');
        sit = str2double(result.sit);
        if sit>length(mutation_seq)
           continue; 
        end
        if mutation_seq(sit)==result.from
            if strcmp(result.to,"del")
                mutation_seq(sit) = '';
            elseif strcmp(result.to,"fs")
                continue;
            elseif strcmp(result.to,"*")
                mutation_seq = mutation_seq(1:sit-1);   
            else
                mutation_seq(sit) = result.to;
            end
            
        end
    end
    
    if mutation_seq==origin_seq
       mutation_seq = "error";
    end
    
end

