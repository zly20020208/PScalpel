clear
clc
%% 设置gpu参数
% gpuDevice(8)
%% 加载训练和测试数据
load('train_data.mat');
load('train_label.mat');
load('test_data.mat');
load('test_label.mat');
train_num = length(train_label);
test_num = length(test_label);
%% 创建网络
fprintf("创建网络...\n")
K.f = ["relu2","softmax"];
K.a = [256,64,2];
[~, adaptive_var, parameter] = creatnn(100,128,128,6,K,false);
load('net_betafold_6L_2L_heads_8.mat');
parameter.dropout = 0.5;
parameter.learning_rate = 0.001;
%% 定义global_step
% global_step = gpuArray(single(0));
global_step = 0;

%% 设置训练的批次及迭代次数
turn_num = 64;
batch_size = 1;
MAX_P = 1e7;
E = [];

fprintf("开始训练...\n")
for i = 1:MAX_P
    Loss = 0;
    global_step = global_step + 1; 
    for k=1:turn_num
        index = randi(train_num,1,batch_size);
        train_d = [];
        train_l = [];
        n_size = [];
        for j=1:batch_size
            t_d = train_data{index(j)};
            t_l = train_label{index(j)};
            n_size(j) = size(t_d,2);
            train_d = [train_d, t_d];
            train_l = [train_l, t_l];
        end
    %     train_d =  gpuArray(single(train_d));
    %     train_l = gpuArray(single(train_l));
    %     n_size = gpuArray(single(n_size));
        [net_betafold, adaptive_var, loss] = backprop(train_d, train_l, n_size, net_betafold, parameter, adaptive_var, global_step);    
        Loss = Loss + loss;
    end

    Loss = Loss / turn_num
    E(i) = Loss;
    if mod(i,50) == 0
        p1=[];
        r1=[];
        f1=[];
        p2=[];
        r2=[];
        f2=[];
        acc1 = [];
        acc2 = [];
        
        auc = [];

        val = [];
        for j=1:test_num
            n_size = size(test_data{j},2);
            [output,~] = forwordprop(net_betafold,test_data{j},n_size,0);
%             y = mat2vec(output)>0.5;
%             val = [val,output];
            y = output>0.5;
            L = test_label{j};
            tp = sum(y==L&y==1);
            fp = sum(y~=L&y==1);
            errors = sum(y~=L);
            fn = errors-fp;
            tn = sum(y==L&y==0);
            if tp==0
                f1(j) = 0;
                p1(j) = 0;
                r1(j) = 0;
            else
                p1(j) = tp/(tp+fp);
                r1(j) = tp/(tp+fn);
                f1(j) = 2*p1(j)*r1(j)/(p1(j)+r1(j));
            end
            acc1(j) = (tp+tn)/(tp+tn+fp+fn);
            tp = sum(y==L&y==0);
            fp = sum(y~=L&y==0);
            errors = sum(y~=L);
            fn = errors-fp;
            tn = sum(y==L&y==1);
            if tp==0
                f2(j) = 0;
                p2(j) = 0;
                r2(j) = 0;
            else
                p2(j) = tp/(tp+fp);
                r2(j) = tp/(tp+fn);
                f2(j) = 2*p2(j)*r2(j)/(p2(j)+r2(j));
            end
            acc2(j) = (tp+tn)/(tp+tn+fp+fn);
            
            %% 计算AUC
            auc = [auc,get_auc(output,L)];
        end
%         save val.mat val -v7.3
%         clear val
        fprintf("正样例...\n")
        fprintf("测试集的预测Accuracy:%f +/- %f\n",mean(acc1),std(acc1))
        fprintf("测试集的预测Precision:%f +/- %f\n",mean(p1),std(p1))
        fprintf("测试集的预测recall:%f +/- %f\n",mean(r1),std(r1))
        fprintf("测试集的预测F-Measure:%f +/- %f\n",mean(f1),std(f1))
        
        fprintf("负样例...\n")
        fprintf("测试集的预测Accuracy:%f +/- %f\n",mean(acc2),std(acc2))
        fprintf("测试集的预测Precision:%f +/- %f\n",mean(p2),std(p2))
        fprintf("测试集的预测recall:%f +/- %f\n",mean(r2),std(r2))
        fprintf("测试集的预测F-Measure:%f +/- %f\n",mean(f2),std(f2))
        
        fprintf("AUC...\n")
        fprintf("测试集的预测AUC:%f +/- %f\n",mean(auc),std(auc))
        
        fprintf("F1-score...\n")
        fprintf("测试集的预测F1-score:%f\n",mean([mean(f1),mean(f2)]))
        
%         net_betafold = gpu_gather(net_betafold);
        save net_betafold_6L_2L_heads_8.mat net_betafold -v7.3
    end
end
plot(E)
