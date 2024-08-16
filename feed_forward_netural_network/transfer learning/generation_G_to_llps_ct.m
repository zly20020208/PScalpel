clc
clear
addpath('betafold')
addpath('method')

load('newdata/CGAS/train_data_yes.mat');
load('newdata/CGAS/train_data_no.mat');

load('newdata/CGAS/test_data.mat');

% load('newdata/CGAS/muta_llps_data_no.mat');
% load('data/muta_test_data/muta_llps_data_yes.mat');

load('model/net_betafold_6L_2L.mat')
% load('betafold/data/threshold.mat')
threshold = 0.5;

%% 生成train_data_yes的图结构
fprintf("生成train_data_yes的图结构...\n")
num = length(train_data_yes);
train_G_yes = [];
for i=1:num
    train_G_yes{i} = generation_G(net_betafold, train_data_yes{i}, threshold);
end

%% 生成train_data_no的图结构
fprintf("生成train_data_no的图结构...\n")
num = length(train_data_no);
train_G_no = [];
for i=1:num
    train_G_no{i} = generation_G(net_betafold, train_data_no{i}, threshold);
end


%% 生成test_data的图结构
fprintf("生成test_data的图结构...\n")
num = length(test_data);
test_G = [];
for i=1:num
    test_G{i} = generation_G(net_betafold, test_data{i}, threshold);
end

%% 生成muta_test_data的图结构
% fprintf("生成muta_test_data的图结构...\n")
% 
% muta_test_G_yes = [];
% num = length(muta_llps_data_yes);
% for i=1:num
%     muta_test_G_yes{i} = generation_G(net_betafold, muta_llps_data_yes{i}, threshold);
% end
% 
% muta_test_G_no = [];
% num = length(muta_llps_data_no);
% for i=1:num
%     muta_test_G_no{i} = generation_G(net_betafold, muta_llps_data_no{i}, threshold);
% end


save newdata/CGAS/train_G_yes_20.mat train_G_yes -v7.3
save newdata/CGAS/train_G_no_20.mat train_G_no -v7.3
save newdata/CGAS/test_G_20.mat test_G -v7.3
% save data/muta_test_data/muta_test_G_yes_20.mat muta_test_G_yes -v7.3
% save data/muta_test_data/muta_test_G_no_20.mat muta_test_G_no -v7.3