load('../newdata/CGAS/test_CGAS/LLPS_data/test_data.mat');
load('../newdata/CGAS/test_CGAS/LLPS_data/test_label.mat');
load('../newdata/CGAS/test_CGAS/test_lcs_features.mat');
test_label1 = test_label;
test_data1 = test_data;
test_lcs_features1 = test_lcs_features;


load('../data/test_data.mat');
load('../data/test_label.mat');
load('../data/LLPS_data/test_lcs_features.mat');
test_data2 = test_data;
test_label2 = test_label;
test_lcs_features2 = test_lcs_features;


test_label = [test_label2,test_label1];
test_data = [test_data2,test_data1];
test_lcs_features = [test_lcs_features1,test_lcs_features2];
save('test_data.mat',"test_data");
save('test_label.mat',"test_label");
save('test_lcs_features.mat',"test_lcs_features");