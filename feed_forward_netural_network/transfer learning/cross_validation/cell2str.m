function str_matrix = cell2str(cell_matrix)
% 将元素类型不同且长度不同的cell矩阵转换为字符串矩阵
% 输入参数 cell_matrix: 元素类型不同且长度不同的cell矩阵
% 输出参数 str_matrix: 与输入矩阵相同大小的string类型矩阵

% 获取输入矩阵的大小
[m, n] = size(cell_matrix);

% 初始化输出矩阵
str_matrix = strings(m, n);

% 遍历输入矩阵的每个元素，将其转换为字符串并存储到输出矩阵中
for i = 1:m
    for j = 1:n
        % 将单个元素转换为字符串，并将其存储到对应位置的输出矩阵中
        str_matrix(i,j) = string(cell_matrix{i,j});
    end
end

end