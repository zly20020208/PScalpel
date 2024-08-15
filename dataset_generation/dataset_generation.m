clc
clear
load('LLPS.mat')
load('LLPS_title.mat')
title = {};
for i=1:31
    title.(LLPS_title(i).replace(' ','_').replace('(','').replace(')','').replace('/','')) = i;
end
% title

%% Choose only single-protein systems
LLPS = LLPS((LLPS(:,title.Components_type)== 'protein(1)') | (LLPS(:,title.Components_type)== 'Protein(1)'),:);
LLPS_Pro_ID = LLPS(:,title.Protein_ID);
length = size(LLPS_Pro_ID,1);
lengths = [];
for i=1:length
   len = size(char(LLPS_Pro_ID(i)),2);
   lengths = [lengths;len];
end
LLPS = LLPS(lengths(:,1)==5,:);
% size(LLPS,1)
clear lengths LLPS_Pro_ID

%% Choose only individual experiments
LLPS = LLPS(LLPS(:,title.Phase_diagram)=="",:);

%% Get construct sequence
length = size(LLPS,1);
for i=1:length
   sequence = LLPS(i,title.Sequence);
   sequences = sequence.splitlines;
   len = size(sequences,1);
   sequence = "";
   for j=2:len
      sequence = sequence + sequences(j);
   end
   LLPS(i,title.Sequence) = sequence;
end

%% Manually correct a concentration that is in w/v format
LLPS(:,title.Solute_concentration) = LLPS(:,title.Solute_concentration).replace('1:19 w/v 5HT1A', '0.05 mg/ml 5HT1A');

%% remove the enrtries where it is unclear what the concentration is
LLPS = LLPS(LLPS(:,title.Sequence)~="",:);
%% Extract solute concentration
% First, value
length = size(LLPS,1);
title.Solute_concentration_value = 32;
title.Solute_concentration_unit = 33;
values = [];
units = [];
for i=1:length
    sol_con = LLPS(i,title.Solute_concentration).split("[");
    sol_con = sol_con(1).replace('≥', '').replace('≤', '').replace('＜', '').replace('>', '');
    sol_con = strip(sol_con);
    sol_con = sol_con.split(" ");
    if sol_con(1).contains("-")||sol_con(1).contains("–")
        rangs = sol_con(1).split("-").split("–");
        sol_con(1) = mean([str2double(rangs(1)),str2double(rangs(2))]);
    end
    values = [values;sol_con(1)];
    if size(sol_con)<2
        units = [units;""];
    else
        units = [units;sol_con(2)];
    end
end
LLPS = [LLPS,values,units];
LLPS = LLPS(LLPS(:,title.Solute_concentration_unit)~="",:);
length = size(LLPS,1);
% LLPS(:,title.Solute_concentration_value)
%% Convert molar concentrations to mass (all that are given as mass have units of mgmL-1)

for i=1:length
    unit = LLPS(i,title.Solute_concentration_unit);
    ratio = 1;
    if unit=="µM"||unit=="uM"||unit=="μM"
        continue;
    elseif unit.contains("nM")
        ratio = 0.001;
        LLPS(i,title.Solute_concentration_unit) = "uM";
    elseif unit=="mM"
        ratio = 1000;
        LLPS(i,title.Solute_concentration_unit) = "uM";
    elseif unit=="mg/ml"||unit=="mg/mL"||unit=="g/L"||unit=="mg"
        ratio = 10^6/molecular_weight(LLPS(i,title.Sequence));
        LLPS(i,title.Solute_concentration_unit) = "uM";
    else
        LLPS(i,title.Solute_concentration_unit) = "";
    end
%     LLPS(i,title.Solute_concentration_unit) = "uM";
    LLPS(i,title.Solute_concentration_value) = str2double(LLPS(i,title.Solute_concentration_value)) * ratio;
end
LLPS = LLPS(LLPS(:,title.Solute_concentration_unit)~="",:);

%% Create LLPS_plus and LLPS_minus
LLPS_No = LLPS(LLPS(:,title.Phase_separation)~="Yes",:);
LLPS_No = unique(LLPS_No(:,title.Sequence));
length = size(LLPS_No,1);
LLPS_NO = [];
for i=1:length
   group = LLPS(LLPS(:,title.Sequence)==LLPS_No(i),title.Protein_ID);
   LLPS_NO = [LLPS_NO;LLPS_No(i),group(1)];
end
LLPS_plus = [];
LLPS_minus = [];
LLPS_YES = LLPS(LLPS(:,title.Phase_separation)=="Yes",:);
LLPS_YES_seq = unique(LLPS_YES(:,title.Sequence));
length = size(LLPS_YES_seq,1);
for i=1:length
%    group = LLPS_YES(LLPS_YES(:,title.Sequence)==LLPS_YES_seq(i),title.Solute_concentration_value);
   group = LLPS_YES(LLPS_YES(:,title.Sequence)==LLPS_YES_seq(i),:);
   if mean(str2double(group(:,title.Solute_concentration_value)))<100
       LLPS_plus = [LLPS_plus;LLPS_YES_seq(i),group(1,title.Protein_ID)];
   else
       LLPS_minus = [LLPS_minus;LLPS_YES_seq(i),group(1,title.Protein_ID)];
   end
   
end
% load('protein.mat')
% length = size(protein,1);
% for i=1:length
%     protein_id_dict.(protein(i,1)) = protein(i,2);
%    sequence =  protein(i,3);
%    sequences = sequence.split(";");
%    len = size(sequences,1);
%    if len>1
%        sequence = "";
%        for j=2:len
%           sequence = sequence + sequences(j);
%        end
%    end
%    sequences = sequence.splitlines;
%    len = size(sequences,1);
%    if len>1
%        sequence = "";
%        for j=2:len
%           sequence = sequence + sequences(j);
%        end
%    end
%    sequence
%    protein_seq_dict.(protein(i,1)) = sequence;
% end
load('protein_id_dict.mat')
load('protein_seq_dict.mat')

%% classify LLPS_plus between know and unknow
LLPS_plus_id = unique(LLPS_plus(:,2));
length = size(LLPS_plus_id,1);
LLPS_plus_know = [];
LLPS_plus_unknow = [];
for i=1:length
    id = protein_id_dict.(LLPS_plus_id(i));
    seq = protein_seq_dict.(LLPS_plus_id(i));
    group = LLPS_plus(LLPS_plus(:,2)==LLPS_plus_id(i),1);
    num = size(group,1);
    if num==1&&seq==group
        LLPS_plus_know = [LLPS_plus_know;id,group];
    elseif id==""||seq=="-"
        LLPS_plus_unknow = [LLPS_plus_unknow;group];
    else
        for j=1:num
           if group(j)==seq
               LLPS_plus_know = [LLPS_plus_know;id,group(j)];
           else
               LLPS_plus_unknow = [LLPS_plus_unknow;group(j)];
           end
        end
    end
end

%% classify LLPS_minus between know and unknow
LLPS_minus_id = unique(LLPS_minus(:,2));
length = size(LLPS_minus_id,1);
LLPS_minus_know = [];
LLPS_minus_unknow = [];
for i=1:length
    id = protein_id_dict.(LLPS_minus_id(i));
    seq = protein_seq_dict.(LLPS_minus_id(i));
    group = LLPS_minus(LLPS_minus(:,2)==LLPS_minus_id(i),1);
    num = size(group,1);
    if num==1&&seq==group
        LLPS_minus_know = [LLPS_minus_know;id,group];
    elseif id==""||seq=="-"
        LLPS_minus_unknow = [LLPS_minus_unknow;group];
    else
        for j=1:num
           if group(j)==seq
               LLPS_minus_know = [LLPS_minus_know;id,group(j)];
           else
               LLPS_minus_unknow = [LLPS_minus_unknow;group(j)];
           end
        end
    end
end


%% classify LLPS_NO between know and unknow
LLPS_NO_id = unique(LLPS_NO(:,2));
length = size(LLPS_NO_id,1);
LLPS_NO_know = [];
LLPS_NO_unknow = [];
for i=1:length
    id = protein_id_dict.(LLPS_NO_id(i));
    seq = protein_seq_dict.(LLPS_NO_id(i));
    group = LLPS_NO(LLPS_NO(:,2)==LLPS_NO_id(i),1);
    num = size(group,1);
    if num==1&&seq==group
        LLPS_NO_know = [LLPS_NO_know;id,group];
    elseif id==""||seq=="-"
        LLPS_NO_unknow = [LLPS_NO_unknow;group];
    else
        for j=1:num
           if group(j)==seq
               LLPS_NO_know = [LLPS_NO_know;id,group(j)];
           else
               LLPS_NO_unknow = [LLPS_NO_unknow;group(j)];
           end
        end
    end
end
save LLPS_NO.mat LLPS_NO
save LLPS_plus.mat LLPS_plus
save LLPS_minus.mat LLPS_minus
save LLPS_plus_know.mat LLPS_plus_know
save LLPS_plus_unknow.mat LLPS_plus_unknow
save LLPS_minus_know.mat LLPS_minus_know
save LLPS_minus_unknow.mat LLPS_minus_unknow
save LLPS_NO_know.mat LLPS_NO_know
save LLPS_NO_unknow.mat LLPS_NO_unknow