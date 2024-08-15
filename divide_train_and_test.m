clc
clear
fprintf("加载数据....\n")
load('data/LLPS_data/data_yes.mat')
load('data/LLPS_data/data_no.mat')
load('data/LLPS_data/lcs_features_yes.mat')
load('data/LLPS_data/lcs_features_no.mat')

train_rate = 0.7;

%% 划分data_yes
fprintf("划分data_min...\n")
data_num = length(data_yes);
index = randperm(data_num);
train_num = floor(data_num*train_rate);
%% 划分出训练数据
fprintf("划分出训练数据...\n")
train_data_yes = data_yes(index(1:train_num));
train_lcs_features_yes = lcs_features_yes(:,index(1:train_num));
%% 划分出测试数据
fprintf("划分出测试数据...\n")
test_data_yes = data_yes(index(train_num+1:data_num));
test_lcs_features_yes = lcs_features_yes(:,index(train_num+1:data_num));




%% 划分data_no
fprintf("划分data_no...\n")
data_num = length(data_no);
index = randperm(data_num);
train_num = floor(data_num*train_rate);
%% 划分出训练数据
fprintf("划分出训练数据...\n")
train_data_no = data_no(index(1:train_num));
train_lcs_features_no = lcs_features_no(:,index(1:train_num));
%% 划分出测试数据
fprintf("划分出测试数据...\n")
test_data_no = data_no(index(train_num+1:data_num));
test_lcs_features_no = lcs_features_no(:,index(train_num+1:data_num));


%% 测试集合并
test_data = [];
test_yes_num = length(test_data_yes);
test_no_num = length(test_data_no);
ind = 1;
for i=1:test_yes_num
    test_data{ind} = test_data_yes{i};
    ind=ind+1;
end
for i=1:test_no_num
    test_data{ind} = test_data_no{i};
    ind=ind+1;
end
test_label = [[ones(1,test_yes_num);zeros(1,test_yes_num)],[zeros(1,test_no_num);ones(1,test_no_num)]];
test_lcs_features = [test_lcs_features_yes,test_lcs_features_no];

fprintf("保存测试数据...\n")
save data/LLPS_data/test_data.mat test_data -v7.3
save data/LLPS_data/test_label.mat test_label -v7.3
save data/LLPS_data/test_lcs_features.mat test_lcs_features -v7.3

fprintf("保存训练数据...\n")
save data/LLPS_data/train_data_yes.mat train_data_yes -v7.3
save data/LLPS_data/train_data_no.mat train_data_no -v7.3
save data/LLPS_data/train_lcs_features_yes.mat train_lcs_features_yes -v7.3
save data/LLPS_data/train_lcs_features_no.mat train_lcs_features_no -v7.3