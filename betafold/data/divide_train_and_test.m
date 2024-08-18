clear
clc
%% 加载数据
fprintf("加载数据....\n")
load('data.mat');
load('label_20.mat');

data_num = length(data);
index = randperm(data_num);
train_rate = 0.9;
train_num = floor(data_num*train_rate);

%% 划分出训练数据
fprintf("划分出训练数据...\n")
train_data = data(index(1:train_num));
train_label = label(index(1:train_num));
%% 划分出测试数据
fprintf("划分出测试数据...\n")
test_data = data(index(train_num+1:data_num));
test_label = label(index(train_num+1:data_num));

save train_data.mat train_data -v7.3
save train_label.mat train_label -v7.3
save test_data.mat test_data -v7.3
save test_label.mat test_label -v7.3