clc
clear
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
for i=1:train_yes_num
    train_data_yes = [train_data_yes,sum(t{i},2)];
end


t = train_data_no;
train_data_no = [];
for i=1:train_no_num
    train_data_no = [train_data_no,sum(t{i},2)];
end

t = test_data;
test_data = [];
for i=1:test_num
    test_data = [test_data,sum(t{i},2)];
end

t = muta_llps_data_no;
muta_llps_data_no = [];
num = length(t);
for i=1:num
    muta_llps_data_no = [muta_llps_data_no,sum(t{i},2)];
end

t = muta_llps_data_yes;
muta_llps_data_yes = [];
num = length(t);
for i=1:num
    muta_llps_data_yes = [muta_llps_data_yes,sum(t{i},2)];
end

muta_llps_data = [muta_llps_data_yes,muta_llps_data_no];
muta_llps_data_label = [[ones(1,size(muta_llps_data_yes,2));zeros(1,size(muta_llps_data_yes,2))],[zeros(1,size(muta_llps_data_no,2));ones(1,size(muta_llps_data_no,2))]];

save train_data_yes.mat train_data_yes
save train_data_no.mat train_data_no
save test_data.mat test_data
save test_label.mat test_label
save muta_llps_data.mat muta_llps_data
save muta_llps_data_label.mat muta_llps_data_label