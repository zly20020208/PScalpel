function [accuracy,prf,auc,p,r,f] = test_model(test_data,test_label,model)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    %[y, Y] = forwordprop(net,data,dropout)
    [y, ~] = forwordprop(model,test_data,0);
    % 计算AUC
    auc = get_auc_multi(y,test_label);
    

    %计算准确率
    result = y(1,:)>0.5;
    gd = test_label(1,:);
    len = length(gd);
    count = sum(result==gd);
    accuracy = count/len;
    
    
    %计算PRF
    classs = [1 0];
    p=[];
    r=[];
    f=[];
    prf = 0;
    xisu = [1/2 1/2];
    for i=1:2
        y_true = gd==classs(i);
        y_pred = result==classs(i);
    %     y_pred(y_true==y_pred)
        tp = sum(y_pred(y_true==y_pred));
        fp = sum(y_pred(y_true~=y_pred));
        errors = sum(y_true~=y_pred);
        fn = errors-fp;
        if tp==0
            f(i) = 0;
        else
            p(i) = tp/(tp+fp);
            r(i) = tp/(tp+fn);
            f(i) = 2*p(i)*r(i)/(p(i)+r(i));
        end
        prf = prf +f(i)*xisu(i);
    end
end

