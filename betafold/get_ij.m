function [i,j] = get_ij(sit,len,old_i)
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
    for i=old_i:len
       if get_sit(i,1,len)<=sit&&get_sit(i+1,1,len)>sit
           break;
       end
    end
    j = sit - get_sit(i,1,len) + 1;
end

