clc
clear

%%
addpath('betafold')
addpath('method')
load('betafold/data/test_pdb_data_w0.mat')
load('betafold/data/test_pdb_label_w0.mat')
load('model/net_betafold_6L_2L.mat')
load('betafold/data/threshold.mat')




p1=[];
r1=[];
f1=[];
p2=[];
r2=[];
f2=[];
acc1 = [];
acc2 = [];

auc = [];
f_score = [];

val = [];
test_num = length(test_pdb_data);

for j=1:test_num
    n_size = size(test_pdb_data{j},2);
    [output,~] = forwordprop(net_betafold,test_pdb_data{j},n_size,0);
%             y = mat2vec(output)>0.5;
            val = [val,output];
    y = output>threshold;
    L = test_pdb_label{j};
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

f_score = (f1+f2)/2;


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


len = length(auc);
numb = [];
for i=1:len
    numb = [numb,i];
end

pdb_w0 = [numb;auc;f_score];
save betafold/data/pdb_w0.mat pdb_w0 -v7.3

% auc = [numb;auc]';
% auc
% save auc.mat auc



