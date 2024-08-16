clc;
clear;

k_fold = 5; % 五折交叉验证
aucs1_folds = cell(1, k_fold); % 存储每个折叠的AUC结果
accuracy_folds = cell(1, k_fold); % 存储每个折叠的准确率结果
prf_folds = cell(1, k_fold); % 存储每个折叠的precision, recall, F-measure结果
mean_p_folds = cell(1, k_fold); % 存储每个折叠的平均precision
mean_r_folds = cell(1, k_fold); % 存储每个折叠的平均recall
mean_f_folds = cell(1, k_fold); % 存储每个折叠的平均F-measure

figure; % 创建一个新的图形窗口

for fold = 1:k_fold
    fprintf("加载第 %d 折训练和测试数据...\n", fold);
%     load(sprintf('newdata/CGAS/true_train_data/train_data_yes_fold%d.mat', fold),'processed_train_data_yes');
%     load(sprintf('newdata/CGAS/true_train_data/train_data_no_fold%d.mat', fold),'processed_train_data_no');
%     load(sprintf('newdata/CGAS/true_train_data/test_data_fold%d.mat', fold),'processed_test_data');
%     load(sprintf('newdata/CGAS/true_train_data/test_label_fold%d.mat', fold));

    load(sprintf('newdata/TDP43/true_train_data/train_data_yes_fold%d.mat', fold),'processed_train_data_yes');
    load(sprintf('newdata/TDP43/true_train_data/train_data_no_fold%d.mat', fold),'processed_train_data_no');
    load(sprintf('newdata/TDP43/true_train_data/test_data_fold%d.mat', fold),'processed_test_data');
    load(sprintf('newdata/TDP43/true_train_data/test_label_fold%d.mat', fold));
% 
%     load(sprintf('newdata/CGAS/train_data_cross_validation/train_lcs_features_yes_fold%d.mat', fold));
%     load(sprintf('newdata/CGAS/train_data_cross_validation/train_lcs_features_no_fold%d.mat', fold));
%     load(sprintf('newdata/CGAS/train_data_cross_validation/test_lcs_features_fold%d.mat', fold));

    load(sprintf('newdata/TDP43/train_data_cross_validation/train_lcs_features_yes_fold%d.mat', fold));
    load(sprintf('newdata/TDP43/train_data_cross_validation/train_lcs_features_no_fold%d.mat', fold));
    load(sprintf('newdata/TDP43/train_data_cross_validation/test_lcs_features_fold%d.mat', fold));

    % 扩展特征
    train_data_yes = [processed_train_data_yes; train_lcs_features_yes];
    train_data_no = [processed_train_data_no; train_lcs_features_no];
    test_data = [processed_test_data; test_lcs_features];

    train_yes_num = size(train_data_yes, 2);
    train_no_num = size(train_data_no, 2);
    test_num = size(test_data, 2);

    % 数据归一化
    [test_data, PS] = mapminmax(test_data);
    train_data_yes = mapminmax('apply', train_data_yes, PS);
    train_data_no = mapminmax('apply', train_data_no, PS);

    % 加载或初始化网络
    fprintf("初始化网络...\n");
    load('net_ffnn_for_gamma.mat');
%     load("../model/net_GAT_TTTGCL_rate005_128_1L_p100.mat")
    net_ffnn.dnn{end}.W = randn(2, size(net_ffnn.dnn{end}.W, 2)); % 两个输出神经元
    net_ffnn.dnn{end}.b = randn(2, 1);
    net_ffnn.dnn{end}.function = "softmax";

    % 设置训练参数
    parameter.learning_rate = 0.005;
    parameter.dropout = 0.5;
    parameter.beta1 = 0.9;
    parameter.beta2 = 0.999;
    parameter.delta = 1e-8;

    % 冻结网络的大部分层
    L = length(net_ffnn.dnn);
    for i = 1:L-1
        net_ffnn.dnn{i}.frozen = true;
    end
    
    adaptive_var = {};
    for i = 1:L
        adaptive_var.s{i} = struct('vW', zeros(size(net_ffnn.dnn{i}.W)), ...
                                   'vb', zeros(size(net_ffnn.dnn{i}.b)), ...
                                   'vgamma', zeros(size(net_ffnn.dnn{i}.gamma)), ...
                                   'vbeta', zeros(size(net_ffnn.dnn{i}.beta)));
        adaptive_var.V{i} = struct('vW', zeros(size(net_ffnn.dnn{i}.W)), ...
                                   'vb', zeros(size(net_ffnn.dnn{i}.b)), ...
                                   'vgamma', zeros(size(net_ffnn.dnn{i}.gamma)), ...
                                   'vbeta', zeros(size(net_ffnn.dnn{i}.beta)));
    end

    % 训练与测试
    fprintf("开始训练第 %d 折...\n", fold);
    global_step = 0;
    batch_size = min(128*2, min(size(train_data_yes, 2), size(train_data_no, 2))); % 调整批次大小以匹配小数据集
    MAX_P = 1250;
    E = zeros(1, MAX_P); % 初始化损失数组
    aucs1 = [];
    accuracies = [];
    prfs = [];
    mean_ps = [];
    mean_rs = [];
    mean_fs = [];
    for i = 1:MAX_P
        global_step = global_step + 1; 
        index_yes = randi(train_yes_num, 1, batch_size);
        index_no = randi(train_no_num, 1, batch_size);
        train_data = [];
        train_label = [];
        for j = 1:batch_size
            if mod(j, 2) == 1
                index = index_yes(j);
                train_data = [train_data, train_data_yes(:, index)];
                train_label = [train_label, [1; 0]];
            else
                index = index_no(j);
                train_data = [train_data, train_data_no(:, index)];
                train_label = [train_label, [0; 1]];
            end
        end
        [net_ffnn, adaptive_var, loss] = backprop(train_data, train_label, net_ffnn, parameter, adaptive_var, global_step);
        E(i) = loss; % 存储损失值
        
        if mod(i, 50) == 0
            % 每50个epoch测试一次
            plot(1:i, E(1:i), 'b'); % 绘制损失曲线
            title(sprintf('Loss Curve for Fold %d', fold));
            xlabel('Epochs');
            ylabel('Loss');
            drawnow;
            
            [accuracy, prf, mean_p, mean_r, mean_f, auc] = test_model2(test_data, test_label, net_ffnn);
            aucs1 = [aucs1, auc];
            accuracies = [accuracies, accuracy];
            prfs = [prfs; prf];
            mean_ps = [mean_ps, mean_p];
            mean_rs = [mean_rs, mean_r];
            mean_fs = [mean_fs, mean_f];
            fprintf("第 %d 折, Epoch %d: accuracy = %.3f, AUC = %.3f\n", fold, i, accuracy, auc);
        end
    end

    aucs1_folds{fold} = aucs1; % 保存每个折叠的AUC结果
    accuracy_folds{fold} = accuracies; % 保存每个折叠的accuracy结果
    prf_folds{fold} = prfs; % 保存每个折叠的prf结果
    mean_p_folds{fold} = mean_ps; % 保存每个折叠的mean_p结果
    mean_r_folds{fold} = mean_rs; % 保存每个折叠的mean_r结果
    mean_f_folds{fold} = mean_fs; % 保存每个折叠的mean_f结果

%     save('newdata/CGAS/net_ffnn_for_gamma_128_1.mat', 'net_ffnn', '-v7.3');
    save('newdata/TDP43/transfer_net_ffnn_for_gamma_128_1.mat', 'net_ffnn', '-v7.3');
end

% 输出每折的性能指标
fprintf("五折交叉验证完成，每折的性能指标如下:\n");
for fold = 1:k_fold
    fprintf("第 %d 折 AUC: %.3f, Accuracy: %.3f\n", fold, mean(aucs1_folds{fold}), mean(accuracy_folds{fold}));
    fprintf("Mean Precision: %.3f, Mean Recall: %.3f, Mean F-measure: %.3f\n", mean(mean_p_folds{fold}), mean(mean_r_folds{fold}), mean(mean_f_folds{fold}));
end
