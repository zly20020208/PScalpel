clear
clc
%% 加载训练和测试数据
addpath('../method')
load('../data/gammaVector_data/data_cancer_neuro.mat')
load('../data/gammaVector_data/data_dict.mat')
load('../data/gammaVector_data/parameter.mat');
load('../data/LLPS_data/train_data_yes.mat');
load('../data/LLPS_data/train_data_no.mat');
data_num = size(data_cancer_neuro,1);
yes_num = length(train_data_yes);
no_num = length(train_data_no);
%% 创建网络
fprintf("创建网络...\n")
[net_deltaVector, adaptive_var] = creatDeltaVector(100,128,1);
% load('net_deltaVector_double_rate05_128_1L.mat');
parameter.dropout = 0;
parameter.learning_rate = 0.001;
parameter.t = 1;%0.05
parameter.rate = 0.01;
%% 定义global_step
global_step = 0;

%% 设置训练的批次及迭代次数
turn_num = 128;
MAX_P = 1e7;
E = [];

fprintf("开始训练...\n")
for i = 1:MAX_P
    Loss1 = 0;
    Loss2 = 0;
    global_step = global_step + 1; 
    indexs = randi(data_num,1,turn_num);
    ind_yes = randi(yes_num,1,turn_num);
    ind_no = randi(no_num,1,turn_num);
    num = 0;
    for k=1:turn_num
        index = indexs(k);
        d.clinVar = get_contrastive_data(data_cancer_neuro(index,:), data_dict);
        if isempty(d.clinVar)
            continue;
        end
        
        
        d.func{1} = train_data_yes{ind_yes(k)};
        d.func{2} = train_data_no{ind_no(k)};
        index=randi(yes_num,1,1);
        while index==ind_yes(k)
            index=randi(yes_num,1,1);
        end
        d.func{3} = train_data_yes{index};

        num= num + 1;
        [net_deltaVector, adaptive_var, loss1,loss2] = backprop(d, net_deltaVector, parameter, adaptive_var, global_step);    
        Loss1 = Loss1 + loss1;
        Loss2 = Loss2 + loss2;
    end
    Loss1 = Loss1/num;
    Loss2 = Loss2/num;
    Loss = [Loss1, Loss2]
    E(i) = Loss1;
    if mod(i,50) == 0
        save net_deltaVector_double_oneplot_rate001_128_1L.mat net_deltaVector -v7.3
    end
end
plot(E)
