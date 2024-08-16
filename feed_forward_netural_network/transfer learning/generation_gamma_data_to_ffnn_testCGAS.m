clc
clear

% addpath('../gammaVector_sG')
% addpath('../gammaVector_snG')
% addpath('../gammaVector_sG_embedding')
% addpath('../gammaVector_snG_embedding')
addpath('../gammaVector_sG_local_and_global')
addpath('../gammaVector_sG_local_and_global_double_oneplot')
% addpath('../gammaVector_sG_local_and_global_heads')
%% addpath('../gammaVector_sG_local_and_global_double_oneplot')
% addpath('../gammaVector_sG_reluA')
% addpath('../gammaVector_sG_residual')
% addpath('../gammaVector_sG_range20')
% addpath('../gammaVector_sG_simple')
% addpath('../gammaVector_sG_heads')
addpath('../method')
% addpath('../GAT_TTTGCL')
%% 加载模型
load('../model/net_gammaVector_sG_local_and_global_double_oneplot_rate005_128_1L_p300.mat');
% load('../gammaVector_sG_simple/net_gammaVector_sG_simple_128_1L_p50_t128.mat');
% load('../gammaVector_sG_heads/net_gammaVector_sG_128_1L_heads_6_per_64_p50_turn64.mat');
% load('../gammaVector_sG/net_gammaVector_sG_128_1L_p50_turn64.mat');
% load('../gammaVector_sG_embedding/net_gammaVector_sG_embedding_128_1L_p50_turn64.mat');
% load('../gammaVector_snG_embedding/net_gammaVector_snG_embedding_128_1L_p1e3_turn128.mat');
%%load('../gammaVector_sG_local_and_global/net_gammaVector_sG_local_and_global_128_1L_p300_turn64.mat');
% load('../gammaVector_sG_local_and_global_heads/net_gammaVector_sG_local_and_global_heads2_per64_128_1L_p350_turn64.mat');
%% load('../gammaVector_sG_local_and_global_double_oneplot/model/net_gammaVector_sG_local_and_global_double_oneplot_rate005_128_1L_p850.mat');
% load('../gammaVector_snG/net_gammaVector_snG_128_1L_p250_turn64.mat');
% load('../gammaVector_sG_reluA/net_gammaVector_sG_128_1L_reluA_p50_turn32.mat');
% load('../gammaVector_sG_residual/net_gammaVector_sG_residual_128_1L_p500_turn64.mat');
% load('../gammaVector_sG_range20/net_gammaVector_sG_range20_128_1L_p500_turn64.mat');
% load('../GAT_TTTGCL/model/net_GAT_double_oneplot_rate005_128_1L_p100.mat');
%% 加载图结构

% load('../newdata/CGAS/train_G_yes_20.mat');
% load('../newdata/CGAS/train_G_no_20.mat');
load('../newdata/CGAS/test_CGAS/test_G_20.mat');

% load('../data/muta_test_data/muta_test_G_yes.mat');
% load('../data/muta_test_data/muta_test_G_no.mat');

% load('../data/LLPS_data/train_G_yes_20.mat');
% load('../data/LLPS_data/train_G_no_20.mat');
% load('../data/LLPS_data/test_G_20.mat');
% 
% load('../data/muta_test_data/muta_test_G_yes_20.mat');
% load('../data/muta_test_data/muta_test_G_no_20.mat');
%% 加载训练和测试数据
% load('../newdata/CGAS/train_data_yes.mat');
% load('../newdata/CGAS/train_data_no.mat');
load('../newdata/CGAS/test_CGAS/test_data.mat');
load('../newdata/CGAS/test_CGAS/test_label.mat');

% load('../data/muta_test_data/muta_llps_data_no.mat');
% load('../data/muta_test_data/muta_llps_data_yes.mat');
% load('../data/muta_test_data/muta_llps_data_label.mat')

% train_yes_num = length(train_data_yes);
% train_no_num = length(train_data_no);
test_num = length(test_data);

% %% 处理数据
% t = train_data_yes;
% train_data_yes = [];
% %[y, Y] = GIN(net,data,G,dropout)
% for i=1:train_yes_num
%     len = size(t{i},2);
%     G = vec2mat(train_G_yes{i},len);
%     [y, ~] = gammaVector(net_gammaVector,t{i},G,0);
%     train_data_yes = [train_data_yes,y];
% end
% 
% 
% t = train_data_no;
% train_data_no = [];
% for i=1:train_no_num
%     len = size(t{i},2);
%     G = vec2mat(train_G_no{i},len);
%     [y, ~] = gammaVector(net_gammaVector,t{i},G,0);
%     train_data_no = [train_data_no,y];
% end

t = test_data;
test_data = [];
for i=1:test_num
    len = size(t{i},2);
    G = vec2mat(test_G{i},len);
    [y, ~] = gammaVector(net_gammaVector,t{i},G,0);
    test_data = [test_data,y];
end
% 
% t = muta_llps_data_no;
% muta_llps_data_no = [];
% num = length(t);
% for i=1:num
%     len = size(t{i},2);
%     G = vec2mat(muta_test_G_no{i},len);
%     [y, ~] = gammaVector(net_gammaVector,t{i},G,0);
%     muta_llps_data_no = [muta_llps_data_no,y];
% end

% t = muta_llps_data_yes;
% muta_llps_data_yes = [];
% num = length(t);
% for i=1:num
%     len = size(t{i},2);
%     G = vec2mat(muta_test_G_yes{i},len);
%     [y, ~] = gammaVector(net_gammaVector,t{i},G,0);
%     muta_llps_data_yes = [muta_llps_data_yes,y];
% end
% 
% muta_llps_data = [muta_llps_data_yes,muta_llps_data_no];
% muta_llps_data_label = [[ones(1,size(muta_llps_data_yes,2));zeros(1,size(muta_llps_data_yes,2))],[zeros(1,size(muta_llps_data_no,2));ones(1,size(muta_llps_data_no,2))]];

% save ../newdata/CGAS/LLPS_data/train_data_yes.mat train_data_yes
% save ../newdata/CGAS/LLPS_data/train_data_no.mat train_data_no
save ../newdata/CGAS/test_CGAS/LLPS_data/test_data.mat test_data
save ../newdata/CGAS/test_CGAS/LLPS_data/test_label.mat test_label
% save muta_llps_data.mat muta_llps_data
% save muta_llps_data_label.mat muta_llps_data_label