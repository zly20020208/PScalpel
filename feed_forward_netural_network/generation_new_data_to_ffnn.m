clc
clear


addpath('../GIN')
% addpath('../GIN_TTTGCL')
addpath('../method')
%% 加载模型
load('../GIN/net_gammaVector_128_1L_cancer.mat');
% load('../GIN_TTTGCL/model/GIN_TTTGCL_rate005_128_1L_p200.mat');
%% 加载图结构
load('../data/LLPS_data/train_G_yes.mat');
load('../data/LLPS_data/train_G_no.mat');
load('../data/LLPS_data/test_G.mat');

load('../data/muta_test_data/muta_test_G_yes.mat');
load('../data/muta_test_data/muta_test_G_no.mat');
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
%[y, Y] = GIN(net,data,G,dropout)
for i=1:train_yes_num
    len = size(t{i},2);
    G = vec2mat(train_G_yes{i},len);
    [y, ~] = GIN(net_gammaVector,t{i},G,0);
    train_data_yes = [train_data_yes,y];
end


t = train_data_no;
train_data_no = [];
for i=1:train_no_num
    len = size(t{i},2);
    G = vec2mat(train_G_no{i},len);
    [y, ~] = GIN(net_gammaVector,t{i},G,0);
    train_data_no = [train_data_no,y];
end

t = test_data;
test_data = [];
for i=1:test_num
    len = size(t{i},2);
    G = vec2mat(test_G{i},len);
    [y, ~] = GIN(net_gammaVector,t{i},G,0);
    test_data = [test_data,y];
end

t = muta_llps_data_no;
muta_llps_data_no = [];
num = length(t);
for i=1:num
    len = size(t{i},2);
    G = vec2mat(muta_test_G_no{i},len);
    [y, ~] = GIN(net_gammaVector,t{i},G,0);
    muta_llps_data_no = [muta_llps_data_no,y];
end

t = muta_llps_data_yes;
muta_llps_data_yes = [];
num = length(t);
for i=1:num
    len = size(t{i},2);
    G = vec2mat(muta_test_G_yes{i},len);
    [y, ~] = GIN(net_gammaVector,t{i},G,0);
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