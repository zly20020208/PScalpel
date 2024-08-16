clc;
clear;

k_fold = 5; % 五折交叉验证
aucs1_folds = cell(1, k_fold); % 存储每个折叠的AUC结果

for fold = 1:k_fold
    fprintf("加载第 %d 折训练和测试数据...\n", fold);
    load(sprintf('../newdata/CGAS/true_train_data/train_data_yes_fold%d.mat', fold),'processed_train_data_yes');
    load(sprintf('../newdata/CGAS/true_train_data/train_data_no_fold%d.mat', fold),'processed_train_data_no');
    load(sprintf('../newdata/CGAS/true_train_data/test_data_fold%d.mat', fold),'processed_test_data');
    load(sprintf('../newdata/CGAS/true_train_data/test_label_fold%d.mat', fold));

    load(sprintf('../newdata/CGAS/train_data_cross_validation/train_lcs_features_yes_fold%d.mat', fold));
    load(sprintf('../newdata/CGAS/train_data_cross_validation/train_lcs_features_no_fold%d.mat', fold));
    load(sprintf('../newdata/CGAS/train_data_cross_validation/test_lcs_features_fold%d.mat', fold));

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
    net_ffnn.dnn{end}.W = randn(2, size(net_ffnn.dnn{end}.W, 2)); % 两个输出神经元
    net_ffnn.dnn{end}.b = randn(2, 1);
    net_ffnn.dnn{end}.function = "softmax";

    % 设置训练参数
    parameter.learning_rate = 0.001;
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
    MAX_P = 5000;
    aucs1 = [];
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
        if mod(i, 50) == 0
            % 每50个epoch测试一次
            [accuracy, prf, mean_p, mean_r, mean_f, auc] = test_model2(test_data, test_label, net_ffnn);
            aucs1 = [aucs1, auc];
            fprintf("第 %d 折, Epoch %d: accuracy = %.3f, AUC = %.3f\n", fold, i, accuracy, auc);
            fprintf("测试集的预测AUC:%f\n", auc);
            fprintf("测试集的预测Accuracy:%f\n", accuracy);
            fprintf("测试集的预测F-Measure:%f\n", prf);
            fprintf("LLPS_yes...\n");
            fprintf("测试集的平均预测Precision:%f\n", mean_p);
            fprintf("测试集的平均预测recall:%f\n", mean_r);
            fprintf("测试集的平均预测F-Measure:%f\n", mean_f);
            fprintf("\n\n");
        end
    end

    aucs1_folds{fold} = aucs1; % 保存每个折叠的结果
    save('../newdata/CGAS/net_ffnn_for_gamma_128_1.mat', 'net_ffnn', '-v7.3');
end

% 输出每折的AUC
fprintf("五折交叉验证完成，每折的AUCs:\n");
for fold = 1:k_fold
    fprintf("第 %d 折 AUC: %.3f\n", fold, mean(aucs1_folds{fold}));
end
