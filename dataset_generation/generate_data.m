clc
clear
load('LLPS_min_seq1.mat')
load('LLPS_min_seq2.mat')
load('LLPS_plus_seq1.mat')
load('LLPS_plus_seq2.mat')
load('LLPS_plus_seq3.mat')
load('LLPS_no_seq.mat')
LLPS_min_seq = [LLPS_min_seq1;LLPS_min_seq2];
LLPS_min_seq = unique(LLPS_min_seq);

LLPS_plus_seq = [LLPS_plus_seq1;LLPS_plus_seq2;LLPS_plus_seq3];
LLPS_plus_seq = unique(LLPS_plus_seq);

LLPS_yes_seq = [LLPS_plus_seq;LLPS_min_seq];
LLPS_yes_seq = unique(LLPS_yes_seq);

LLPS_no_seq = unique(LLPS_no_seq);
len = length(LLPS_no_seq);
no_seq = [];
for i=1:len
    if sum(LLPS_yes_seq==LLPS_no_seq(i))==0
        no_seq = [no_seq;LLPS_no_seq(i)];
    end
end
LLPS_no_seq = no_seq;

save LLPS_yes_seq.mat LLPS_yes_seq
save LLPS_no_seq.mat LLPS_no_seq