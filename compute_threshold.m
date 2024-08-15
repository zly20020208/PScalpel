clc
clear
% load('betafold/data/test_label.mat');
% 
% tar = [];
% len = length(test_label);
% for i=1:len
%     tar = [tar,test_label{i}];
% end
% save betafold/data/tar.mat tar -v7.3


%%
addpath('method')
load('betafold/data/tar.mat')
load('betafold/data/val.mat')

% 1-specificity = fpr

% Sensitivity = tpr;

[tpr,fpr,thresholds] =roc(tar,val);
plot(fpr,tpr)
RightIndex=(tpr+(1-fpr)-1);
[~,index]=max(RightIndex);
RightIndexVal=RightIndex(index(1));
tpr_val=tpr(index(1));
fpr_val=fpr(index(1));
threshold=thresholds(index(1));
auc = get_auc_multi(val,tar);
disp(['AUC： ',num2str(auc)])
disp(['平均准确率： ',num2str((RightIndexVal+1)*0.5)]);
disp(['最佳正确率： ',num2str(tpr_val)])
disp(['最佳错误率： ',num2str(fpr_val)])

save betafold/data/threshold.mat threshold