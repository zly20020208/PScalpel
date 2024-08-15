clc
clear
% load('model/net_ffnn_for_gamma_double_rate005_p350_auc07226.mat')
% load('model/net_ffnn_for_gamma_double_rate005_p300.mat')
% load('model/net_gammaVector_sG_local_and_global_double_oneplot_rate005_128_1L_p350.mat')
% load('model/net_gammaVector_sG_local_and_global_double_oneplot_rate005_128_1L_p300.mat')
load('model/net_deltaVector_128_1L_selu_cancer_position.mat');
load('model/net_ffnn_for_delta_128_1.mat')
load('model/net_betafold_6L_2L.mat')
load('model/threshold.mat')
addpath('method')
% addpath('gammaVector_sG_local_and_global_double_oneplot')
addpath('deltaVector_selu')
addpath('feed_forward_netural_network')
cgas = 'QPWHGKAMQRASEAGATAPKASARNARGAPMDPTESPAAPEAALPKAGKFGPARKSGSRQKKSAPDTQERPPVRATGARAKKAPQRAQDTQPSDATSAPGAEGLEPPAAREPALSRAGSCRQRGARCSTKPRPPPGPWDVPSPGLPVSAPILVRRDAAPGASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDIKDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAKEGNGFQEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFDEF';

%% 尝试改变cgas序列
cgas(203-1) = 'C'
% cgas(6-1) = 'N';
% cgas(79-1) = 'N';
cgas(203-1)

%% 计算score
len = length(cgas);
best_seq = [];
vector = get_vector(['M',cgas]);
% addpath('betafold')
% G = generation_G(net_betafold, vector, threshold);
% G = vec2mat(G,len+1);
data = deltaVector(net_deltaVector,vector,0);
% data = gammaVector(net_gammaVector,vector,G,0);
% data = [data;get_lcs_features(['M',cgas])];
addpath('feed_forward_netural_network')
[y, ~] = forwordprop(net_ffnn,data,0);
score = y(1)