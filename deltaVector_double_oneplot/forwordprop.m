function [Y, result] = forwordprop(net,data,dropout)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    clinVar = data.clinVar;
    func = data.func;
    %[y, Y] = deltaVector(net,data,dropout)
    len = length(clinVar);
    %% data{1}为该批次中唯一正例，data{len}为原始数据
    for i=1:len
        [result.clinVar{i},Y.clinVar{i}.Y] = deltaVector(net,clinVar{i},dropout);
        [result.func{i},Y.func{i}.Y] = deltaVector(net,func{i},dropout);
    end


     
end
