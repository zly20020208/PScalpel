% 加载 cell.mat 文件
load('../newdata/TDP/TDP_mutations_no.mat');
% 
% % 创建一个与加载的数据大小相同的新数组，用于保存字符串数据
% CGAS_mutations_no_seq = cell(size(mutations_no));
% 
% % 遍历每个单元格并将其内容转换为字符串
% for i = 1:numel(mutations_no)
%     % 将单元格内容转换为字符串
%     CGAS_mutations_no_seq{i} = mat2str(mutations_no{i});
% end
% 
% % 保存转换后的数据为 string.mat 文件
% save('CGAS_mutations_no_seq.mat', 'CGAS_mutations_no_seq');

TDP_mutations_no_seq = cell2str(mutations_no);
save('../newdata/TDP/TDP_mutations_no_seq.mat', 'TDP_mutations_no_seq');