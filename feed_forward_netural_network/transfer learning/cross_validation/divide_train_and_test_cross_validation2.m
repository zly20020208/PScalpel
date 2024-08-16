clc;
clear;
fprintf("加载数据....\n");
% load('../newdata/CGAS/data_mutation_no.mat');
% load('../newdata/CGAS/data_mutation_yes.mat');
% load('../newdata/CGAS/lcs_features_no.mat');
% load('../newdata/CGAS/lcs_features_yes.mat');
load('../newdata/TDP43/data_mutation_no.mat');
load('../newdata/TDP43/data_mutation_yes.mat');
load('../newdata/TDP43/lcs_features_no.mat');
load('../newdata/TDP43/lcs_features_yes.mat');
train_rate = 0.9; % 训练数据占90%
k_fold = 5; % 五折交叉验证

%% 处理带突变的数据
fprintf("处理data_yes...\n")
data_num_yes = length(data_mutation_yes);
train_num_yes = floor(data_num_yes * train_rate);
indices_yes = randperm(data_num_yes);

%% 处理不带突变的数据
fprintf("处理data_no...\n")
data_num_no = length(data_mutation_no);
train_num_no = floor(data_num_no * train_rate);
indices_no = randperm(data_num_no);

for k = 1:k_fold
    fprintf("处理第%d折...\n", k);

    %% 确定训练和测试索引
    % 对于data_yes
    test_idx_yes = indices_yes((train_num_yes + 1):end);
    train_idx_yes = indices_yes(1:train_num_yes);
    
    % 对于data_no
    test_idx_no = indices_no((train_num_no + 1):end);
    train_idx_no = indices_no(1:train_num_no);

    %% 提取训练和测试数据
    train_data_yes = data_mutation_yes(train_idx_yes);
    test_data_yes = data_mutation_yes(test_idx_yes);
    train_lcs_features_yes = lcs_features_yes(:, train_idx_yes);
    test_lcs_features_yes = lcs_features_yes(:, test_idx_yes);

    train_data_no = data_mutation_no(train_idx_no);
    test_data_no = data_mutation_no(test_idx_no);
    train_lcs_features_no = lcs_features_no(:, train_idx_no);
    test_lcs_features_no = lcs_features_no(:, test_idx_no);

    %% 合并测试集数据
    test_data = [test_data_yes, test_data_no];
    test_label = [[ones(1,length(test_data_yes)); zeros(1,length(test_data_yes))], [zeros(1,length(test_data_no)); ones(1,length(test_data_no))]];
    test_lcs_features = [test_lcs_features_yes, test_lcs_features_no];

    %% 保存数据
    save(sprintf('../newdata/TDP43/train_data_cross_validation/test_data_fold%d.mat', k), 'test_data', '-v7.3');
    save(sprintf('../newdata/TDP43/train_data_cross_validation/test_label_fold%d.mat', k), 'test_label', '-v7.3');
    save(sprintf('../newdata/TDP43/train_data_cross_validation/test_lcs_features_fold%d.mat', k), 'test_lcs_features', '-v7.3');
    save(sprintf('../newdata/TDP43/train_data_cross_validation/train_data_yes_fold%d.mat', k), 'train_data_yes', '-v7.3');
    save(sprintf('../newdata/TDP43/train_data_cross_validation/train_data_no_fold%d.mat', k), 'train_data_no', '-v7.3');
    save(sprintf('../newdata/TDP43/train_data_cross_validation/train_lcs_features_yes_fold%d.mat', k), 'train_lcs_features_yes', '-v7.3');
    save(sprintf('../newdata/TDP43/train_data_cross_validation/train_lcs_features_no_fold%d.mat', k), 'train_lcs_features_no', '-v7.3');
end

fprintf("五折交叉验证数据准备完成。\n");
