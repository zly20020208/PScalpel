% index = benign(:,2)~="";
% benign = benign(index,:);
% save benign.mat benign

%%
% clc
% clear
% load('pathogenic_cancer_neuro.mat')
% index = pathogenic_cancer_neuro(:,2)~="";
% pathogenic_cancer_neuro = pathogenic_cancer_neuro(index,:);
% save pathogenic_cancer_neuro.mat pathogenic_cancer_neuro

%%
% clc
% clear
% uniprot = fastaread("uniprot-reviewed_yes.fasta");
% len = length(uniprot);
% uniprot_mat = [];
% for i=1:len
%     uniprot_mat = [uniprot_mat;string(uniprot(i).Header),string(uniprot(i).Sequence)];
% end
% save uniprot_mat.mat uniprot_mat

%%
% load('pathogenic_cancer_neuro.mat')
% index = ~contains(pathogenic_cancer_neuro(:,2),'*')&~contains(pathogenic_cancer_neuro(:,2),'fs');
% pathogenic_cancer_neuro = pathogenic_cancer_neuro(index,:);
% save pathogenic_cancer_neuro.mat pathogenic_cancer_neuro

%%
% load('pathogenic_cancer_neuro.mat')
% load('benign.mat')
% benign(:,1) = replace(benign(:,1),'-','_');
% pathogenic_cancer_neuro(:,1) = replace(pathogenic_cancer_neuro(:,1),'-','_');
% save benign.mat benign
% save pathogenic_cancer_neuro.mat pathogenic_cancer_neuro

%%
% clc
% clear
% load('benign.mat')
% load('pathogenic_cancer_neuro.mat')
% index = ~contains(benign(:,1),'|');
% benign = benign(index,:);
% index = ~contains(pathogenic_cancer_neuro(:,1),'|');
% pathogenic_cancer_neuro = pathogenic_cancer_neuro(index,:);
% genes = [benign(:,1);pathogenic_cancer_neuro(:,1)];
% genes = unique(genes);
% save genes.mat genes
% save benign.mat benign
% save pathogenic_cancer_neuro.mat pathogenic_cancer_neuro


%% 
% clc
% clear
% load('genes.mat')
% load('uniprot_mat.mat')
% len = length(genes);
% gene_dict = {};
% not_exit_genes = [];
% for i=1:len
%     P = 'GN='+ genes(i)+ ' ';
%     P = replace(P,"_","-");
%     index = contains(uniprot_mat(:,1),P,'IgnoreCase',true);
%     if sum(index)==1
%         sequence = uniprot_mat(index,2);
%     else
%         same_gene = uniprot_mat(index,:);
%         ind = contains(same_gene(:,1),"OS=Homo sapiens",'IgnoreCase',true);
%         if sum(ind)>=1
%             sequence = same_gene(ind,2);
%             sequence = sequence(end);
%         else
%             not_exit_genes = [not_exit_genes,genes(i)];
%             continue;
%         end
%     end
%     gene_dict.(genes(i)) = sequence;
% end
% save gene_dict.mat gene_dict
% save not_exit_genes.mat not_exit_genes

%%
% load('not_exit_genes.mat')
% load('benign.mat')
% load('pathogenic_cancer_neuro.mat')
% load('genes.mat')
% index = ~contains(benign(:,1),not_exit_genes);
% benign = benign(index,:);
% index = ~contains(pathogenic_cancer_neuro(:,1),not_exit_genes);
% pathogenic_cancer_neuro = pathogenic_cancer_neuro(index,:);
% index = ~contains(genes(:),not_exit_genes);
% genes = genes(index);
% save benign.mat benign
% save pathogenic_cancer_neuro.mat pathogenic_cancer_neuro
% save genes.mat genes


%%
% clc
% clear
% load('benign.mat')
% load('pathogenic_cancer_neuro.mat')
% load('genes.mat')
% len = length(genes);
% gene = [];
% be=[];
% pa=[];
% for i=1:len
%     if sum(strcmp(benign(:,1),genes(i)))>0&&sum(strcmp(pathogenic_cancer_neuro(:,1),genes(i)))>0
%         gene = [gene;genes(i)];
%         be= [be ; benign(strcmp(benign(:,1),genes(i)),:)];
%         pa = [pa ; pathogenic_cancer_neuro(strcmp(pathogenic_cancer_neuro(:,1),genes(i)),:)];
%     end
% end
% benign_cancer_neuro = be;
% pathogenic_cancer_neuro = pa;
% genes = gene;
% save benign_cancer_neuro.mat benign_cancer_neuro
% save pathogenic_cancer_neuro.mat pathogenic_cancer_neuro
% save genes.mat genes

%%
% clc
% clear
% load('benign_cancer_neuro.mat')
% load('pathogenic_cancer_neuro.mat')
% len = length(benign_cancer_neuro);
% data_cancer_neuro = [];
% for i=1:len
%     strs = benign_cancer_neuro(i,:);
%     gene = benign_cancer_neuro(i,1);
%     index = strcmp(pathogenic_cancer_neuro(:,1),gene);
%     strs = repmat(strs,sum(index),1);
%     strs = [strs,pathogenic_cancer_neuro(index,2)];
%     data_cancer_neuro = [data_cancer_neuro;strs];
% end
% save data_cancer_neuro.mat data_cancer_neuro

%%
% clc
% clear
% load('gene_dict.mat')
% t = struct2cell(gene_dict);
% len = length(t);
% l = [];
% for i=1:len
%     l = [l;length(char(t{i}))];
% end
% sum(l<100)