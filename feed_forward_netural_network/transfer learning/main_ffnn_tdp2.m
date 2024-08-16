clear
clc
%% 加载训练和测试数据
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

%% 为数据集扩充特征
train_data_yes = [train_data_yes;train_lcs_features_yes];
train_data_no = [train_data_no;train_lcs_features_no];
test_data = [test_data;test_lcs_features];



train_yes_num = size(train_data_yes,2);
train_no_num = size(train_data_no,2);
test_num = size(test_data,2);

%%  数据归一化
[test_data,PS] = mapminmax(test_data);
train_data_yes = mapminmax('apply',train_data_yes,PS);
train_data_no = mapminmax('apply',train_data_no,PS);

%% 创建网络
addpath('../method')
fprintf("创建网络...\n")
K.f = ["selu","selu","softmax"];
% K.a = [128+40,64,32,16,2];
K.a = [128+40,64,16,2];
[net_ffnn, adaptive_var, parameter] = creatnn(K);
% load('../model/net_ffnn_for_gamma_double_rate005_p300.mat')
load('../model/net_ffnn_for_gamma_double_rate005_p350_auc07226.mat')
parameter.dropout = 0.5;
parameter.learning_rate = 0.001;
%% 定义global_step
global_step = 0;

%% 设置训练的批次及迭代次数
batch_size = 128*2;
MAX_P = 200;
E = [];
aucs1 = [];
aucs2 = [];

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
    [net_ffnn, adaptive_var, loss] = backprop(train_data, train_label, net_ffnn, parameter, adaptive_var, global_step);
    loss;
    E(i) = loss;
%     plot(E)
%     drawnow;
    if mod(i,50) == 0
        %% 对常规数据集进行测试
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
%         
%         if length(aucs1)>2&&((length(aucs2)>5&&auc<aucs2(end-5))||(auc>0.65&&auc<aucs2(end-2))||(aucs1(end)<aucs1(end-2)&&auc<aucs2(end-2)))
%             save net_ffnn_for_gamma.mat net_ffnn -v7.3
%             break;
%         end
%         if length(aucs1)>2&&(aucs1(end)<aucs1(end-2)||auc<aucs2(end-2))
%             save net_ffnn_for_gamma.mat net_ffnn -v7.3
%             break;
%         end
        
        
        save ../newdata/TDP/net_ffnn_for_gamma_yuanshi.mat net_ffnn -v7.3
    end
end
