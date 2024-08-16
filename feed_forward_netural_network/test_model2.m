function [accuracy, prf, mean_p, mean_r, mean_f, auc] = test_model2(test_data_mat, test_label, model)

   % 确保使用的是完整的测试数据集
    if size(test_data_mat, 2) ~= size(test_label, 2)
        error('测试数据和测试标签的样本数量不匹配。');
    end

    % 前向传播以获取预测结果
    [y, ~] = forwordprop(model, test_data_mat, 0);

    % 计算 AUC 等其他指标，根据需要调整...
    auc = get_auc_multi(y, test_label);
    % 转换预测结果为类别索引
    [~, predicted_classes] = max(y, [], 1);

    % 转换独热编码的测试标签为类别索引
    [~, actual_classes] = max(test_label, [], 1);


    % 计算准确率
    accuracy = sum(predicted_classes == actual_classes) / length(actual_classes);

    % 初始化精确度、召回率和F1分数数组
    p = zeros(1, size(y, 1));
    r = zeros(1, size(y, 1));
    f = zeros(1, size(y, 1));
    prf = 0;
    xisu = [1/2 1/2];
    % 计算每个类别的精确度、召回率和F1分数
    for i = 1:size(y, 1)
        y_true = actual_classes == i;
        y_pred = predicted_classes == i;
        tp = sum(y_true & y_pred);
        fp = sum(~y_true & y_pred);
        fn = sum(y_true & ~y_pred);
        if tp > 0
            p(i) = tp / (tp + fp);
            r(i) = tp / (tp + fn);
            f(i) = 2 * p(i) * r(i) / (p(i) + r(i));
        end
        prf = prf +f(i)*xisu(i);
    end

    % 计算平均精确度、召回率和F1分数
    mean_p = mean(p);
    mean_r = mean(r);
    mean_f = mean(f);
end
