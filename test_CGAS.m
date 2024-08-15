clc
clear
fprintf("加载数据....\n")
load('newdata/CGAS/data_mutation_no.mat')
load('newdata/CGAS/data_mutation_yes.mat')
load('newdata/CGAS/lcs_features_no.mat')
load('newdata/CGAS/lcs_features_yes.mat')
%% 测试集合并
test_data = [];
test_yes_num = length(data_mutation_yes);
test_no_num = length(data_mutation_no);
ind = 1;
for i=1:test_yes_num
    test_data{ind} = data_mutation_yes{i};
    ind=ind+1;
end
for i=1:test_no_num
    test_data{ind} = data_mutation_no{i};
    ind=ind+1;
end
test_label = [[ones(1,test_yes_num);zeros(1,test_yes_num)],[zeros(1,test_no_num);ones(1,test_no_num)]];
test_lcs_features = [lcs_features_no,lcs_features_yes];

fprintf("保存测试数据...\n")
save newdata/CGAS/test_CGAS/test_data.mat test_data -v7.3
save newdata/CGAS/test_CGAS/test_label.mat test_label -v7.3
save newdata/CGAS/test_CGAS/test_lcs_features.mat test_lcs_features -v7.3
