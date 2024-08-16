% 这段代码首先加载数据，然后使用 MATLAB 的 crossvalind 函数生成随机的训练和测试数据索引，实现五折交叉验证。
% 每一折中，数据将根据索引被分配到训练集和测试集。最后，为每一折保存相应的数据集文件。
clc;
clear;
fprintf("加载数据....\n");
load('../newdata/CGAS/data_mutation_no.mat');
load('../newdata/CGAS/data_mutation_yes.mat');
load('../newdata/CGAS/lcs_features_no.mat');
load('../newdata/CGAS/lcs_features_yes.mat');

k_fold = 5; % 设定为五折交叉验证

%% 处理带突变的数据
fprintf("处理data_yes...\n")
data_num_yes = length(data_mutation_yes);
indices_yes = crossvalind('Kfold', data_num_yes, k_fold);

%% 处理不带突变的数据
fprintf("处理data_no...\n")
data_num_no = length(data_mutation_no);
indices_no = crossvalind('Kfold', data_num_no, k_fold);

for k = 1:k_fold
    fprintf("处理第%d折...\n", k);

    %% 选择训练数据和测试数据 - data_yes
    test_idx_yes = (indices_yes == k);
    train_idx_yes = ~test_idx_yes;
    train_data_yes = data_mutation_yes(train_idx_yes);
    test_data_yes = data_mutation_yes(test_idx_yes);
    train_lcs_features_yes = lcs_features_yes(:, train_idx_yes);
    test_lcs_features_yes = lcs_features_yes(:, test_idx_yes);

    %% 选择训练数据和测试数据 - data_no
    test_idx_no = (indices_no == k);
    train_idx_no = ~test_idx_no;
    train_data_no = data_mutation_no(train_idx_no);
    test_data_no = data_mutation_no(test_idx_no);
    train_lcs_features_no = lcs_features_no(:, train_idx_no);
    test_lcs_features_no = lcs_features_no(:, test_idx_no);

    %% 合并测试集数据
    test_data = [test_data_yes, test_data_no];
    test_label = [[ones(1,length(test_data_yes));zeros(1,length(test_data_yes))],[zeros(1,length(test_data_no));ones(1,length(test_data_no))]];
%     test_label = [ones(length(test_data_yes), 1); zeros(length(test_data_no), 1)];

    test_lcs_features = [test_lcs_features_yes, test_lcs_features_no];

    %% 保存数据
    save(sprintf('../newdata/CGAS1/train_data_cross_validation/test_data_fold%d.mat', k), 'test_data', '-v7.3');
    save(sprintf('../newdata/CGAS1/train_data_cross_validation/test_label_fold%d.mat', k), 'test_label', '-v7.3');
    save(sprintf('../newdata/CGAS1/train_data_cross_validation/test_lcs_features_fold%d.mat', k), 'test_lcs_features', '-v7.3');
    save(sprintf('../newdata/CGAS1/train_data_cross_validation/train_data_yes_fold%d.mat', k), 'train_data_yes', '-v7.3');
    save(sprintf('../newdata/CGAS1/train_data_cross_validation/train_data_no_fold%d.mat', k), 'train_data_no', '-v7.3');
    save(sprintf('../newdata/CGAS1/train_data_cross_validation/train_lcs_features_yes_fold%d.mat', k), 'train_lcs_features_yes', '-v7.3');
    save(sprintf('../newdata/CGAS1/train_data_cross_validation/train_lcs_features_no_fold%d.mat', k), 'train_lcs_features_no', '-v7.3');
end

fprintf("五折交叉验证数据准备完成。\n");
