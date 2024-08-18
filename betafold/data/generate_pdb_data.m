clc
clear
addpath('../../method')
% str = '/home/baode/data2/wz/trainData/HUMAN/';
% str = '/home/turing/data2/wangz/test_data_PDB_00-hq_after_2022_02_02/';
str = '/home/turing/data2/wangz/w0-zz/';
% files = dir(strcat(str,'*.pdb'));
files = dir(strcat(str,'*.ent'));
number_files = length(files);
kk=0;
for k=1:number_files
    try 
        gfl = pdbread(strcat(str,files(k).name)) ;
    catch
        continue;
    end
    
    if ~isfield(gfl,'Sequence')||~isfield(gfl.Model(1),'Atom')
       continue; 
    end
    sequence = gfl.Sequence.Sequence;
    len = length(sequence);
    pos = find(strcmp({gfl.Model(1).Atom(:).AtomName},"CA"));
    if len~=length(pos)||len<100
       continue; 
    end
    kk=kk+1
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
    test_pdb_data{kk} = get_vector(sequence);
    test_pdb_label{kk} = mat2vec(distance_mat)<10;
    
end

save test_pdb_data_r0.mat test_pdb_data -v7.3
save test_pdb_label_r0.mat test_pdb_label -v7.3
