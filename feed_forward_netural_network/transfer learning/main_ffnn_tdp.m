clear
clc
%% 加载训练和测试数据
load('../newdata/TDP/LLPS_data/train_data_yes.mat');
load('../newdata/TDP/LLPS_data/train_data_no.mat');
load('../newdata/TDP/LLPS_data/test_data.mat');
load('../newdata/TDP/LLPS_data/test_label.mat');

% load('muta_llps_data.mat')
% load('muta_llps_data_label.mat')

%% 加载lcs_features
load('../newdata/TDP/train_lcs_features_yes.mat');
load('../newdata/TDP/train_lcs_features_no.mat');
load('../newdata/TDP/test_lcs_features.mat');

% load('../data/muta_test_data/muta_llps_lcs_features_no.mat');
% load('../data/muta_test_data/muta_llps_lcs_features_yes.mat');

%% 为数据集扩充特征
train_data_yes = [train_data_yes;train_lcs_features_yes];
train_data_no = [train_data_no;train_lcs_features_no];
test_data = [test_data;test_lcs_features];
% muta_llps_data = [muta_llps_data;[muta_llps_lcs_features_yes,muta_llps_lcs_features_no]];



train_yes_num = size(train_data_yes,2);
train_no_num = size(train_data_no,2);
test_num = size(test_data,2);

%%  数据归一化
[test_data,PS] = mapminmax(test_data);
train_data_yes = mapminmax('apply',train_data_yes,PS);
train_data_no = mapminmax('apply',train_data_no,PS);
% muta_llps_data = mapminmax('apply',muta_llps_data,PS);
%% 创建网络
addpath('../method')
fprintf("创建网络...\n")
% K.f = ["selu","selu","softmax"];
% % K.a = [128+40,64,32,16,2];
% K.a = [128+40,64,16,2];
% [net_ffnn, adaptive_var, parameter] = creatnn(K);

% load('../model/net_ffnn_for_gamma_double_rate005_p300.mat')
% load('../model/net_ffnn_for_gamma_double_rate005_p350_auc07226.mat')
load('net_ffnn_for_gamma.mat')
%% 调整网络结构以适应新的分类任务
% 假定网络的最后一层是softmax层，并调整其权重和偏差的大小
net_ffnn.dnn{end}.W = randn(2, size(net_ffnn.dnn{end}.W, 2)); % 两个输出神经元
net_ffnn.dnn{end}.b = randn(2, 1);
net_ffnn.dnn{end}.function = "softmax";
%% 设置训练参数
parameter.learning_rate = 0.001; % 适合小数据集的较低学习率
parameter.dropout = 0.5; % dropout
parameter.beta1 = 0.9; % 动量
parameter.beta2 = 0.999; % RMSprop参数
parameter.delta = 1e-8; % 防止除以0

%% 初始化adaptive_var用于动量和RMSprop.初始化了adaptive_var结构，这将存储用于动量和RMSprop算法的累积梯度和梯度平方。请根据实际情况调整这些参数。



L = length(net_ffnn.dnn); % 获取神经网络层数
% 冻结除了最后一层以外的所有层
for i = 1:L-1
    net_ffnn.dnn{i}.frozen = true;
end
adaptive_var = {};
% adaptive_var.V = cell(1,L);
% adaptive_var.s = cell(1,L);
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

% % 为最后一层初始化 V 和 s
% adaptive_var.s{L} = struct('vW', zeros(size(net_ffnn.dnn{L}.W)), ...
%                            'vb', zeros(size(net_ffnn.dnn{L}.b)));
% adaptive_var.V{L} = struct('vW', zeros(size(net_ffnn.dnn{L}.W)), ...
%                            'vb', zeros(size(net_ffnn.dnn{L}.b)));
% 
% % 如果最后一层有批归一化的操作，还需要初始化 gamma 和 beta
% if isfield(net_ffnn.dnn{L}, 'gamma')
%     adaptive_var.s{L}.vgamma = zeros(size(net_ffnn.dnn{L}.gamma));
%     adaptive_var.V{L}.vgamma = zeros(size(net_ffnn.dnn{L}.gamma));
% end
% 
% if isfield(net_ffnn.dnn{L}, 'beta')
%     adaptive_var.s{L}.vbeta = zeros(size(net_ffnn.dnn{L}.beta));
%     adaptive_var.V{L}.vbeta = zeros(size(net_ffnn.dnn{L}.beta));
% end
% 




% Now initialize the V and s fields for the last (unfrozen) layer
% adaptive_var.V{L} = struct('vW', zeros(size(net_ffnn.dnn{L}.W)), ...
%                            'vb', zeros(size(net_ffnn.dnn{L}.b)));
% 
% adaptive_var.s{L} = struct('vW', zeros(size(net_ffnn.dnn{L}.W)), ...
%                            'vb', zeros(size(net_ffnn.dnn{L}.b)));
% 
% if isfield(net_ffnn.dnn{L}, 'gamma')
%     adaptive_var.V{L}.vgamma = zeros(size(net_ffnn.dnn{L}.gamma));
%     adaptive_var.s{L}.vgamma = zeros(size(net_ffnn.dnn{L}.gamma));
% end
% 
% if isfield(net_ffnn.dnn{L}, 'beta')
%     adaptive_var.V{L}.vbeta = zeros(size(net_ffnn.dnn{L}.beta));
%     adaptive_var.s{L}.vbeta = zeros(size(net_ffnn.dnn{L}.beta));
% end

%% 定义global_step
global_step = 0;

%% 设置训练的批次及迭代次数
% batch_size = 128*2;
batch_size = min(128*2, min(size(train_data_yes, 2), size(train_data_no, 2)));% 调整批次大小以匹配小数据集
MAX_P = 2500;
E = [];
aucs1 = [];


fprintf("开始训练...\n")
for i = 1:MAX_P
    
    global_step = global_step + 1; 
    index_yes = randi(train_yes_num,1,batch_size);
    index_no = randi(train_no_num,1,batch_size);
    train_data = [];
    train_label = [];
    for j=1:batch_size
        if mod(j,2)==1
            index = index_yes(j);
            train_data = [train_data,train_data_yes(:,index)];
            train_label = [train_label,[1;0]];
        else
            index = index_no(j);
            train_data = [train_data,train_data_no(:,index)];
            train_label = [train_label,[0;1]];
        end
    end
%     x = [train_data_yes(:, index_yes), train_data_no(:, index_no)];  % 合并样本数据
%     labels = [repmat([1; 0], 1, length(index_yes)), repmat([0; 1], 1, length(index_no))];  % 创建标签
    [net_ffnn, adaptive_var, loss] = backprop(train_data, train_label, net_ffnn, parameter, adaptive_var, global_step);
    loss;
    E(i) = loss;
%     plot(E)
%     drawnow;
    if mod(i,50) == 0
        %% 对常规数据集进行测试
%         [accuracy,prf,auc,p,r,f] = test_model(test_data,test_label,net_ffnn);
         [accuracy, prf, mean_p, mean_r, mean_f, auc] = test_model2(test_data,test_label,net_ffnn);
        aucs1 = [aucs1,auc];
        subplot(1,2,1)
        plot(aucs1)
        drawnow;
        fprintf("Epoch %d: accuracy = %.3f, AUC = %.3f\n", i, accuracy, auc);
        fprintf("测试集的预测AUC:%f\n",auc)
        fprintf("测试集的预测Accuracy:%f\n",accuracy);
        fprintf("测试集的预测F-Measure:%f\n",prf)
        fprintf("LLPS_yes...\n")
        fprintf("测试集的平均预测Precision:%f\n",mean_p)
        fprintf("测试集的平均预测recall:%f\n",mean_r)
        fprintf("测试集的平均预测F-Measure:%f\n",mean_f)
        fprintf("\n\n")
       %% 对突变数据集进行测试
%         [accuracy,prf,auc,p,r,f] = test_model(muta_llps_data,muta_llps_data_label,net_ffnn);
% 
%         aucs2 = [aucs2,auc];
%         subplot(1,2,2)
%         plot(aucs2)
%         drawnow;
%         fprintf("测试集的预测AUC:%f\n",auc)
%         fprintf("测试集的预测Accuracy:%f\n",accuracy);
%         fprintf("测试集的预测F-Measure:%f\n",prf)
%         fprintf("muta_LLPS_yes...\n")
%         fprintf("测试集的预测Precision:%f\n",p(1))
%         fprintf("测试集的预测recall:%f\n",r(1))
%         fprintf("测试集的预测F-Measure:%f\n",f(1))
%         
%         fprintf("muta_LLPS_no...\n")
%         fprintf("测试集的预测Precision:%f\n",p(2))
%         fprintf("测试集的预测recall:%f\n",r(2))
%         fprintf("测试集的预测F-Measure:%f\n\n",f(2))
        
%         if length(aucs1)>2&&((length(aucs2)>5&&auc<aucs2(end-5))||(auc>0.65&&auc<aucs2(end-2))||(aucs1(end)<aucs1(end-2)&&auc<aucs2(end-2)))
%             save ../newdata/CGAS/net_ffnn_for_gamma.mat net_ffnn -v7.3
%             break;
%         end
%         if length(aucs1)>2&&(aucs1(end)<aucs1(end-2)||auc<aucs2(end-2))
%             save net_ffnn_for_gamma.mat net_ffnn -v7.3
%             break;
%         end
        
        
         save ../newdata/TDP/net_ffnn_for_gamma_128_1.mat net_ffnn -v7.3
    end
end
