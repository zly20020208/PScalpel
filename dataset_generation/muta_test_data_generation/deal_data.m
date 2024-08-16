clc
clear
load('muta_LLPS_no.mat')
load('muta_LLPS_yes.mat')
addpath('../../method')
%% 只保留单蛋白数据
index = muta_LLPS_yes(:,1)=="protein(1)";
muta_LLPS_yes = muta_LLPS_yes(index,:);

index = muta_LLPS_no(:,1)=="protein(1)";
muta_LLPS_no = muta_LLPS_no(index,:);

%% 只保留突变蛋白数据
index = ~contains(muta_LLPS_yes(:,2),"-");
muta_LLPS_yes = muta_LLPS_yes(index,3);

index = ~contains(muta_LLPS_no(:,2),"-");
muta_LLPS_no = muta_LLPS_no(index,3);

%% 保留突变蛋白序列
muta_llps_seq_yes = [];
num = length(muta_LLPS_yes);
for i=1:num
   sequence = muta_LLPS_yes(i);
   sequences = sequence.splitlines;
   len = size(sequences,1);
   sequence = "";
   for j=2:len
      sequence = sequence + sequences(j);
   end
   muta_llps_seq_yes = [muta_llps_seq_yes;sequence];
end

muta_llps_seq_no = [];
num = length(muta_LLPS_no);
for i=1:num
   sequence = muta_LLPS_no(i);
   sequences = sequence.splitlines;
   len = size(sequences,1);
   sequence = "";
   for j=2:len
      sequence = sequence + sequences(j);
   end
   muta_llps_seq_no = [muta_llps_seq_no;sequence];
end

%% 去除空序列
index = muta_llps_seq_yes~="";
muta_llps_seq_yes = muta_llps_seq_yes(index);

index = muta_llps_seq_no~="";
muta_llps_seq_no = muta_llps_seq_no(index);

%% 去除重复序列
muta_llps_seq_yes = unique(muta_llps_seq_yes);

muta_llps_seq_no = unique(muta_llps_seq_no);

%% 从muta_llps_seq_no中找出出现在muta_llps_seq_yes里的序列，去除
len = length(muta_llps_seq_no);
seq_no = [];
for i=1:len
    if sum(muta_llps_seq_yes==muta_llps_seq_no(i))==0
        seq_no = [seq_no, muta_llps_seq_no(i)];
    end
end
muta_llps_seq_no = seq_no;

%% 保存序列数据
save muta_llps_seq_no.mat muta_llps_seq_no -v7.3
save muta_llps_seq_yes.mat muta_llps_seq_yes -v7.3

%% 将序列处理成特征向量
len = length(muta_llps_seq_yes);
muta_llps_data_yes = {};
for i=1:len
    muta_llps_data_yes{i} = get_vector(muta_llps_seq_yes(i));
end

len = length(muta_llps_seq_no);
muta_llps_data_no = {};
for i=1:len
    muta_llps_data_no{i} = get_vector(muta_llps_seq_no(i));
end

%% 保存特征数据
save muta_llps_data_no.mat muta_llps_data_no -v7.3
save muta_llps_data_yes.mat muta_llps_data_yes -v7.3