clc
clear
load('newdata/TDP/TDP_mutations_no_seq.mat')
load('newdata/TDP/TDP_mutations_yes_seq.mat')
addpath('method')
data_mutation_no = [];
data_mutation_yes = [];

len = length(TDP_mutations_no_seq);
for i=1:len
    data_mutation_no{i} = get_vector(TDP_mutations_no_seq(i));
end

len = length(TDP_mutations_yes_seq);
for i=1:len
    data_mutation_yes{i} = get_vector(TDP_mutations_yes_seq(i));
end



save newdata/TDP/data_mutation_no.mat data_mutation_no -v7.3
save newdata/TDP/data_mutation_yes.mat data_mutation_yes -v7.3
