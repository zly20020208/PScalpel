clc
clear
load('data.mat')
nums = zeros(500,2);
len = length(data);
for i=1:500
   nums(i,1) = i*10; 
end
for i=1:len
   nums(floor(length(data{i})/10),2) =  nums(floor(length(data{i})/10),2)+1;
end
