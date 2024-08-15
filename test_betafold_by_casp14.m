clc
clear

%%
addpath('betafold')
addpath('method')
load('betafold/data/test_casp14_data.mat')
load('betafold/data/test_casp14_label.mat')
load('model/net_betafold_6L_2L.mat')
load('betafold/data/threshold.mat')


%%
tps = zeros(33,100);
tns = zeros(33,100);
fps = zeros(33,100);
fns = zeros(33,100);

aucs = zeros(33,100);
labels = zeros(33,100);

%%

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
test_num = length(test_casp14_data);

% for j=1:test_num
for j=1:1
    n_size = size(test_casp14_data{j},2);
    [output,~] = forwordprop(net_betafold,test_casp14_data{j},n_size,0);
    %% 计算top100的TP，FP，TN，FN
    [predict,I] = sort(output,'descend');
    label = test_casp14_label{j}(I);
    
    if predict(1)>=threshold && label(1)==1
        tps(j,1) = 1;
    elseif predict(1)<threshold && label(1)==0
        tns(j,1) = 1;
    elseif predict(1)<threshold && label(1)==1
        fns(j,1) = 1;
    else
        fps(j,1) = 1;
    end
    aucs(j,1) = predict(1);
    labels(j,1) = label(1);
    for i=2:100
        tps(j,i) = tps(j,i-1);
        tns(j,i) = tns(j,i-1);
        fps(j,i) = fps(j,i-1);
        fns(j,i) = fns(j,i-1);
        if predict(i)>=threshold && label(i)==1
            tps(j,i) = tps(j,i)+1;
        elseif predict(i)<threshold && label(i)==0
            tns(j,i) = tns(j,i)+1;
        elseif predict(i)<threshold && label(i)==1
            fns(j,i) = fns(j,i)+1;
        else
            fps(j,i) = fps(j,i)+1;
        end

        aucs(j,i) = predict(i);
        labels(j,i) = label(i);
        %     if label(x,y)==1
        %         sum_p_rank = sum_p_rank + l + 1 - i;
        %     end
    end
    
    
    
    
    
    %%
%             y = mat2vec(output)>0.5;
            val = [val,output];
    y = output>threshold;
    L = test_casp14_label{j};
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
        save betafold/data/val_pdb.mat val -v7.3
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

betafold_aucs = auc;
save experiment_result/betafold_aucs.mat betafold_aucs -v7.3



%% 计算置信度前10的残基对
for i=1:10
    sit = I(i);
    x = 0;
    y = 0;
    for j= n_size-1:-1:1
        x = x+1;
        if sit>j
            sit = sit - j;
        else
            y = x + sit;
            break;
        end
        
    end
    i+" "+x+" "+y
    
end