clc
clear
% load('deltaVector_selu/net_deltaVector_128_1L_selu_cancer_position.mat');
% load('feed_forward_netural_network/net_ffnn_for_delta_128_1.mat')
load('model/net_ffnn_for_gamma_double_rate005_p350_auc07226.mat')
load('model/net_gammaVector_sG_local_and_global_double_oneplot_rate005_128_1L_p350.mat')
load('model/net_betafold_6L_2L.mat')
load('model/threshold.mat')
load('data/AA_species.mat')
addpath('method')
addpath('gammaVector_sG_local_and_global_double_oneplot')
addpath('feed_forward_netural_network')
cgas = 'QPWHGKAMQRASEAGATAPKASARNARGAPMDPTESPAAPEAALPKAGKFGPARKSGSRQKKSAPDTQERPPVRATGARAKKAPQRAQDTQPSDATSAPGAEGLEPPAAREPALSRAGSCRQRGARCSTKPRPPPGPWDVPSPGLPVSAPILVRRDAAPGASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDIKDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAKEGNGFQEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFDEF';
len = length(cgas);
m = length(AA_species);
best_seq = [];
vector = get_vector(['M',cgas]);
addpath('betafold')
G = generation_G(net_betafold, vector, threshold);
G = vec2mat(G,len+1);
data = gammaVector(net_gammaVector,vector,G,0);
data = [data;get_lcs_features(['M',cgas])];
addpath('feed_forward_netural_network')
[y, ~] = forwordprop(net_ffnn,data,0);
best_score = y(1);

load('current_better.mat')
%current = 1;



for i=current:len-1
   current = i;
   save current_better.mat current
   for j=0:m
        for p=i+1:len
           for q=0:m
               new_cgas = cgas;
               if q>0
                   new_cgas(p) = AA_species(q);
               else
                   new_cgas(p) = '';
               end
               if j>0
                  new_cgas(i) = AA_species(j);
              else
                  new_cgas(i) = '';
              end
               vector = get_vector(['M' new_cgas]);
               addpath('betafold')
               G = generation_G(net_betafold, vector, threshold);
               n_size = len+1;
               if j==0
                   n_size = n_size-1;
               end
               if q==0
                   n_size = n_size-1;
               end
               G = vec2mat(G,n_size);
               data = gammaVector(net_gammaVector,vector,G,0);
               data = [data;get_lcs_features(['M' new_cgas])];
               addpath('feed_forward_netural_network')
               [y, ~] = forwordprop(net_ffnn,data,0);
               score = y(1);
              if score>best_score
                  best_score = score
                  best_seq = ['M',new_cgas]
                  fp=fopen('varOfCgasBetter.txt','a');%'A.txt'为文件名；'a'为打开方式：在打开的文件末端添加数据，若文件不存在则创建。
                  fprintf(fp,'%d \n',best_score);%fp为文件句柄，指定要写入数据的文件。注意：%d后有空格。
                  fprintf(fp,'%s \n',best_seq);
                  fclose(fp);%关闭文件。
              end
           end
        end

   end
end
best_score
best_seq