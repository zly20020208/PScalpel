clc
clear
% str = '/home/baode/data2/wz/trainData/HUMAN/';
str = '/home/baode/data2/wz/trainData/';
dirs = dir(str);
number_dirs = length(dirs);
data = [];
label = [];
number = 1;
for r=1:number_dirs
    if ~isfolder(dirs(r))
       continue; 
    end
    dir_name = str+dirs(r)+'/'
    files = dir(strcat(dir_name,'*.pdb'));
    number_files = length(files);
    for k=1:number_files
        gfl = pdbread(strcat(str,files(k).name)) ;
        sequence = gfl.Sequence.Sequence;
        len = length(sequence);
        pos = find(strcmp({gfl.Model.Atom(:).AtomName},'CA'));
        X=[gfl.Model.Atom(pos).X];
        Y=[gfl.Model.Atom(pos).Y];
        Z=[gfl.Model.Atom(pos).Z]; 
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
        data{number} = get_vector(sequence);
        label{number} = mat2vec(distance_mat)<10;
        number = number+1;
    end    
end


save data.mat data -v7.3
save label.mat label -v7.3
