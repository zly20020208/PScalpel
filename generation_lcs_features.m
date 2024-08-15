clc
clear
load('data/LLPS_data/LLPS_yes_seq.mat')
load('data/LLPS_data/LLPS_no_seq.mat')
addpath('method')

len = length(LLPS_yes_seq);
lcs_features_yes = zeros(40,len);
for i=1:len
    lcs_features_yes(:,i) = get_lcs_features(LLPS_yes_seq(i));
end

len = length(LLPS_no_seq);
lcs_features_no = zeros(40,len);
for i=1:len
    lcs_features_no(:,i) = get_lcs_features(LLPS_no_seq(i));
end

save data/LLPS_data/lcs_features_yes.mat lcs_features_yes -v7.3
save data/LLPS_data/lcs_features_no.mat lcs_features_no -v7.3


load('data/muta_test_data/muta_llps_seq_yes.mat')
load('data/muta_test_data/muta_llps_seq_no.mat')
addpath('method')

len = length(muta_llps_seq_yes);
muta_llps_lcs_features_yes = zeros(40,len);
for i=1:len
    muta_llps_lcs_features_yes(:,i) = get_lcs_features(muta_llps_seq_yes(i));
end

len = length(muta_llps_seq_no);
muta_llps_lcs_features_no = zeros(40,len);
for i=1:len
    muta_llps_lcs_features_no(:,i) = get_lcs_features(muta_llps_seq_no(i));
end

save data/muta_test_data/muta_llps_lcs_features_yes.mat muta_llps_lcs_features_yes -v7.3
save data/muta_test_data/muta_llps_lcs_features_no.mat muta_llps_lcs_features_no -v7.3