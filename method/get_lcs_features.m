function [vector] = get_lcs_features(sequence)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L','M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];
    n_window = 20;
    cutoff = 7;       
    seq = char(sequence);
    
    vector = zeros(20,1);
    len = length(seq);
    lc_bool = zeros(1,len);
    for i=1:1:len-n_window+1
        peptide = seq(i:i+n_window-1);
        if length(unique(peptide))<=cutoff
           lc_bool(i:i+n_window-1) = 1;
        end
    end
    
    l = sum(lc_bool);
    if l==0
        vector = zeros(40,1);
        return;
    end
    for i=1:length(RESIDUES)
       vector(i) = sum(lc_bool(seq==RESIDUES(i)));
    end
    vector = [vector;vector/l];
    
end

