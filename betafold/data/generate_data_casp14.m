clc
clear
% load('seq_casp14.mat')
addpath('../../ps_ability/method')
load('../../ps_ability/method/AA_dict.mat')
str = '/home/turing/data2/wz/trainData/CASP14/';
% str = '/home/turing/data2/wangz/test_data_PDB_00-hq_after_2022_02_02/';
files = dir(strcat(str,'*.pdb'));
% files = dir(strcat(str,'*.ent'));
number_files = length(files);
kk=0;
for k=1:number_files
    gfl = pdbread(strcat(str,files(k).name)) ;
%     if ~isfield(gfl,'Sequence')||~isfield(gfl.Model(1),'Atom')
%        continue; 
%     end
%     sequence = gfl.Sequence.Sequence;
%     sequence = seq_casp14.(files(k).name(1:5));
%     len = length(sequence);
%     if len >1000
%         continue
%     end
    pos = find(strcmp({gfl.Model(1).Atom(:).AtomName},"CA"));
    len = length(pos);
    if len>1000
       continue; 
    end
    kk=kk+1
    sequence = '';
    for i=1:len
        sequence = [sequence , AA_dict.(gfl.Model(1).Atom(pos(i)).resName)];
    end
    sequence
    X=[gfl.Model(1).Atom(pos).X];
    Y=[gfl.Model(1).Atom(pos).Y];
    Z=[gfl.Model(1).Atom(pos).Z] ;
    xyz = [X',Y',Z'];
    distance_mat = zeros(len,len);
    for i=1:len
       for j=i+1:len
           distance = xyz(i,:)-xyz(j,:);
           distance = sum(distance.^2)^0.5;
           distance_mat(i,j) = distance;
%            distance_mat(j,i) = distance;
       end
    end
    test_casp14_data{kk} = get_vector(sequence);
    test_casp14_label{kk} = mat2vec(distance_mat)<10;
    
end

save test_casp14_data.mat test_casp14_data -v7.3
save test_casp14_label.mat test_casp14_label -v7.3
