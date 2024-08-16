clc
clear

addpath('../deltaVector_selu')
% addpath('../deltaVector_double')
% addpath('../deltaVector_double_oneplot')
% addpath('../deltaVector')
% addpath('../deltaVector_deep')
addpath('../method')
%% 加载模型
load('../deltaVector_selu/net_deltaVector_128_1L_selu_cancer_position.mat');
% load('../deltaVector_double/net_deltaVector_double_rate01_128_1L.mat');
%% load('../deltaVector_double_oneplot/net_deltaVector_double_oneplot_rate001_128_1L.mat');
% load('../deltaVector_selu/net_deltaVector_128_1L.mat');
% load('../deltaVector_selu/net_deltaVector_128_1L_selu_position.mat');
% load('../deltaVector_selu/net_deltaVector_64_2L_selu_cancer_position.mat');
% load('../deltaVector/net_deltaVector_all_data_128_1L.mat');
% load('../deltaVector/net_deltaVector_256_2L.mat');
% load('../deltaVector_deep/net_deltaVector_256_2L_selu_cancer_position_deep.mat')
load('../self-attention_TTTGCL/model/self-attention_TTTGCL_rate003_128_1L_p800.mat');
%% 加载训练和测试数据
load('../data/LLPS_data/train_data_yes.mat');
load('../data/LLPS_data/train_data_no.mat');
load('../data/LLPS_data/test_data.mat');
load('../data/LLPS_data/test_label.mat');

load('../data/muta_test_data/muta_llps_data_no.mat');
load('../data/muta_test_data/muta_llps_data_yes.mat');
load('../data/muta_test_data/muta_llps_data_label.mat')

train_yes_num = length(train_data_yes);
train_no_num = length(train_data_no);
test_num = length(test_data);

%% 处理数据
t = train_data_yes;
train_data_yes = [];
%[y, Y] = deltaVector(net,data,dropout)
for i=1:train_yes_num
    [y, ~] = deltaVector(net_deltaVector,t{i},0);
    train_data_yes = [train_data_yes,y];
end


t = train_data_no;
train_data_no = [];
for i=1:train_no_num
    [y, ~] = deltaVector(net_deltaVector,t{i},0);
    train_data_no = [train_data_no,y];
end

t = test_data;
test_data = [];
for i=1:test_num
    [y, ~] = deltaVector(net_deltaVector,t{i},0);
    test_data = [test_data,y];
end

t = muta_llps_data_no;
muta_llps_data_no = [];
num = length(t);
for i=1:num
    [y, ~] = deltaVector(net_deltaVector,t{i},0);
    muta_llps_data_no = [muta_llps_data_no,y];
end

t = muta_llps_data_yes;
muta_llps_data_yes = [];
num = length(t);
for i=1:num
    [y, ~] = deltaVector(net_deltaVector,t{i},0);
    muta_llps_data_yes = [muta_llps_data_yes,y];
end

muta_llps_data = [muta_llps_data_yes,muta_llps_data_no];
muta_llps_data_label = [[ones(1,size(muta_llps_data_yes,2));zeros(1,size(muta_llps_data_yes,2))],[zeros(1,size(muta_llps_data_no,2));ones(1,size(muta_llps_data_no,2))]];

save train_data_yes.mat train_data_yes
save train_data_no.mat train_data_no
save test_data.mat test_data
save test_label.mat test_label
save muta_llps_data.mat muta_llps_data
save muta_llps_data_label.mat muta_llps_data_label