clc;
clear;

% 添加必要的路径
addpath('../../gammaVector_sG_local_and_global');
addpath('../../gammaVector_sG_local_and_global_double_oneplot');
addpath('../../method');

% 设置五折交叉验证
k_fold = 5;

for fold = 1:k_fold
    fprintf('处理第 %d 折...\n', fold);
    
    % 加载训练和测试数据及图结构
%     load(sprintf('../newdata/CGAS/train_data_cross_validation/train_data_yes_fold%d.mat', fold));
%     load(sprintf('../newdata/CGAS/train_data_cross_validation/train_data_no_fold%d.mat', fold));
%     load(sprintf('../newdata/CGAS/train_data_cross_validation/test_data_fold%d.mat', fold));
%     load(sprintf('../newdata/CGAS/train_data_cross_validation/test_label_fold%d.mat', fold));

    load(sprintf('../newdata/TDP43/train_data_cross_validation/train_data_yes_fold%d.mat', fold));
    load(sprintf('../newdata/TDP43/train_data_cross_validation/train_data_no_fold%d.mat', fold));
    load(sprintf('../newdata/TDP43/train_data_cross_validation/test_data_fold%d.mat', fold));
    load(sprintf('../newdata/TDP43/train_data_cross_validation/test_label_fold%d.mat', fold));

%     load(sprintf('../newdata/CGAS/train_G_cross_validation/train_G_yes_fold%d.mat', fold));
%     load(sprintf('../newdata/CGAS/train_G_cross_validation/train_G_no_fold%d.mat', fold));
%     load(sprintf('../newdata/CGAS/train_G_cross_validation/test_G_fold%d.mat', fold));

    load(sprintf('../newdata/TDP43/train_G_cross_validation/train_G_yes_fold%d.mat', fold));
    load(sprintf('../newdata/TDP43/train_G_cross_validation/train_G_no_fold%d.mat', fold));
    load(sprintf('../newdata/TDP43/train_G_cross_validation/test_G_fold%d.mat', fold));

    % 加载模型
    load('../../model/net_gammaVector_sG_local_and_global_double_oneplot_rate005_128_1L_p300.mat');

    % 处理训练数据 yes
    fprintf('生成处理后的训练数据 yes...\n');
    train_yes_num = length(train_data_yes);
    processed_train_data_yes = [];
    for i = 1:train_yes_num
        len = size(train_data_yes{i}, 2);
        G = vec2mat(train_G_yes{i}, len);
        [y, ~] = gammaVector(net_gammaVector, train_data_yes{i}, G, 0);
        processed_train_data_yes = [processed_train_data_yes, y];
    end

    % 处理训练数据 no
    fprintf('生成处理后的训练数据 no...\n');
    train_no_num = length(train_data_no);
    processed_train_data_no = [];
    for i = 1:train_no_num
        len = size(train_data_no{i}, 2);
        G = vec2mat(train_G_no{i}, len);
        [y, ~] = gammaVector(net_gammaVector, train_data_no{i}, G, 0);
        processed_train_data_no = [processed_train_data_no, y];
    end

    % 处理测试数据
    fprintf('生成处理后的测试数据...\n');
    test_num = length(test_data);
    processed_test_data = [];
    for i = 1:test_num
        len = size(test_data{i}, 2);
        G = vec2mat(test_G{i}, len);
        [y, ~] = gammaVector(net_gammaVector, test_data{i}, G, 0);
        processed_test_data = [processed_test_data, y];
    end

    % 保存处理后的数据
%     save(sprintf('../newdata/CGAS/true_train_data/train_data_yes_fold%d.mat', fold), 'processed_train_data_yes');
%     save(sprintf('../newdata/CGAS/true_train_data/train_data_no_fold%d.mat', fold), 'processed_train_data_no');
%     save(sprintf('../newdata/CGAS/true_train_data/test_data_fold%d.mat', fold), 'processed_test_data');
%     save(sprintf('../newdata/CGAS/true_train_data/test_label_fold%d.mat', fold), 'test_label');  % 假设标签不变

    save(sprintf('../newdata/TDP43/true_train_data/train_data_yes_fold%d.mat', fold), 'processed_train_data_yes');
    save(sprintf('../newdata/TDP43/true_train_data/train_data_no_fold%d.mat', fold), 'processed_train_data_no');
    save(sprintf('../newdata/TDP43/true_train_data/test_data_fold%d.mat', fold), 'processed_test_data');
    save(sprintf('../newdata/TDP43/true_train_data/test_label_fold%d.mat', fold), 'test_label');  % 假设标签不变
end

fprintf('五折交叉验证数据处理完成。\n');
