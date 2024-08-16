% clc
% clear
% load('data.mat')
min = 10000;
len = length(data);
for i=1:len
   n = size(data{i},2);
   if n<min 
       min = n
       ii = i;
   end
end
ii;
min;
data{ii};