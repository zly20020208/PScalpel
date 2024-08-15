clc
clear
addpath('betafold')
addpath('method')
load('model/net_betafold_6L_2L.mat')
% load('betafold/data/threshold.mat')
load('data/gammaVector_data/data_dict.mat');

%% 生成data_dict的图结构
fprintf("生成data_dict的图结构...\n")
names = fieldnames(data_dict);
num = length(names);
G_dict = {};
for i=1:num
    name = names{i};
    G = generation_G(net_betafold, data_dict.(name), 0.5);
    G_dict.(name) = G;
%     train_G{i} = vec2mat(output>0.5,n_size);
    
end
save data/gammaVector_data/G_dict_20.mat G_dict -v7.3