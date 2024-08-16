clc;
clear;
addpath('../../betafold');
addpath('../../method');

k_fold = 5; % 五折交叉验证

for fold = 1:k_fold
    fprintf("加载第 %d 折数据...\n", fold);
%     load(sprintf('../newdata/CGAS/train_data_cross_validation/train_data_yes_fold%d.mat', fold));
%     load(sprintf('../newdata/CGAS/train_data_cross_validation/train_data_no_fold%d.mat', fold));
%     load(sprintf('../newdata/CGAS/train_data_cross_validation/test_data_fold%d.mat', fold));
    load(sprintf('../newdata/TDP43/train_data_cross_validation/train_data_yes_fold%d.mat', fold));
    load(sprintf('../newdata/TDP43/train_data_cross_validation/train_data_no_fold%d.mat', fold));
    load(sprintf('../newdata/TDP43/train_data_cross_validation/test_data_fold%d.mat', fold));

    load('../../model/net_betafold_6L_2L.mat');
    threshold = 0.5;

    %% 生成train_data_yes的图结构
    fprintf("生成第 %d 折train_data_yes的图结构...\n", fold);
    num = length(train_data_yes);
    train_G_yes = cell(1, num);
    for i = 1:num
        train_G_yes{i} = generation_G(net_betafold, train_data_yes{i}, threshold);
    end

    %% 生成train_data_no的图结构
    fprintf("生成第 %d 折train_data_no的图结构...\n", fold);
    num = length(train_data_no);
    train_G_no = cell(1, num);
    for i = 1:num
        train_G_no{i} = generation_G(net_betafold, train_data_no{i}, threshold);
    end

    %% 生成test_data的图结构
    fprintf("生成第 %d 折test_data的图结构...\n", fold);
    num = length(test_data);
    test_G = cell(1, num);
    for i = 1:num
        test_G{i} = generation_G(net_betafold, test_data{i}, threshold);
    end

    %% 保存生成的图结构
    save(sprintf('../newdata/TDP43/train_G_cross_validation/train_G_yes_fold%d.mat', fold), 'train_G_yes', '-v7.3');
    save(sprintf('../newdata/TDP43/train_G_cross_validation/train_G_no_fold%d.mat', fold), 'train_G_no', '-v7.3');
    save(sprintf('../newdata/TDP43/train_G_cross_validation/test_G_fold%d.mat', fold), 'test_G', '-v7.3');
end

fprintf("五折交叉验证的图结构生成完毕。\n");
