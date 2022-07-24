% EEG Classification BCI Task
% Computational Intelligence Course Final Project
% Armin Panjehpour - 98101288

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% phase1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% add to path feature functions

addpath('feature_functions/');

%% Load Datasete
clear; clc; close all;

Data = load('CI_Project_data.mat');
Train_Data = Data.TrainData;
Test_Data = Data.TestData;
Train_Label = Data.TrainLabel - 1;

disp('Train\Test Data Created!')

%% Sampling Rate
clc; close all;

Trial_Time = 1.5; % in seconds
Fs = size(Train_Data,2)/Trial_Time;

sprintf('Sampling Rate is: %f', Fs)

%% get trial numbers of two classes
clc; close all;

class1_trialnum = find(Train_Label == 0);
class2_trialnum = find(Train_Label == 1);

size(class1_trialnum)
size(class2_trialnum)

%% %%%%%%%%%%%%%%%% Apply the Feature Functions on Training Data %%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all;


all_features = {};
all_features_name = {};

all_features_test = {};

%% calculate variance of each trail and channel - one value for each channel and trial for train data
% clc; close all;
% 
% var_feature = squeeze(Var_Feature(Train_Data));
% 
% % normalize features 
% var_feature = normalize(var_feature, 0, 2);
% 
% n_class = 2;
% var_feature_classes = zeros(n_class,...
%     int16(size(var_feature, 2)/n_class), size(var_feature, 1));
% 
% var_feature_classes(1,:,:) = var_feature(:, class1_trialnum)';
% var_feature_classes(2,:,:) = var_feature(:, class2_trialnum)';
% 
% 
% 
% all_features{end+1} = var_feature_classes;
% 
% size(var_feature_classes)
% 
% % feature names
% for i = 1:size(var_feature, 1)
%     all_features_name{end+1} = "var_feature for channel " + i;
% end

%% calculate amplitude histogram - nbin values for each channel for training data
% clc; close all;
% 
% n_bins = 5;
% min_amp = -10;
% max_amp = 10;
% hist_feature = amp_hist_Feature(Train_Data, n_bins, min_amp, max_amp);
% 
% % normalize features 
% hist_feature = normalize(hist_feature, 0, 2);
% 
% n_class = 2;
% hist_feature_classes = zeros(n_class, size(hist_feature, 1),...
%     int16(size(hist_feature, 2)/n_class), size(hist_feature, 3));
% 
% hist_feature_classes(1,:,:,:) = hist_feature(:,class1_trialnum,:);
% hist_feature_classes(2,:,:,:) = hist_feature(:,class2_trialnum,:);
% 
% for i = 1:size(hist_feature, 3)
%     all_features{end+1} = permute(hist_feature_classes(:,:,:,i), [1 3 2 4]);
% end
% 
% size(hist_feature_classes)
% 
% % feature names
% for i = 1:size(hist_feature, 1)
%     for j = 1:n_bins
%         all_features_name{end+1} = "hist_feature for channel " + i + "bin " + j;
%     end
% end

%% calculate AR coefficients - order values for each trail and channel
% clc; close all;
% 
% order = 5;
% ar_feature = AR_Coeffs(Train_Data, order);
% 
% % normalize features 
% ar_feature = normalize(ar_feature, 0, 1);
% 
% n_class = 2;
% ar_feature_classes = zeros(n_class, int16(size(ar_feature, 1)/n_class),...
%     size(ar_feature, 2), size(ar_feature, 3));
% 
% ar_feature_classes(1,:,:,:) = ar_feature(class1_trialnum,:,:);
% ar_feature_classes(2,:,:,:) = ar_feature(class2_trialnum,:,:);
% 
% for i = 1:size(ar_feature, 3)
%     all_features{end+1} = ar_feature_classes(:,:,:,i);
% end
% 
% size(ar_feature_classes)
% 
% % feature names
% for i = 1:size(ar_feature, 2)
%     for j = 1:order+1
%         all_features_name{end+1} = "ar_feature for channel " + i + "coeff " + j;
%     end
% end
% 

%% calculate Form Factor -  one value for each channel and trial for training data
% clc; close all;
% 
% FF_feature = FF_Feature(Train_Data);
% 
% % normalize features 
% FF_feature = normalize(FF_feature, 0, 2);
% 
% n_class = 2;
% FF_feature_classes = zeros(n_class, int16(size(FF_feature, 2)/n_class), size(FF_feature, 1));
% 
% FF_feature_classes(1,:,:) = FF_feature(:,class1_trialnum)';
% FF_feature_classes(2,:,:) = FF_feature(:,class2_trialnum)';
% 
% all_features{end+1} =  FF_feature_classes;
% 
% size(FF_feature_classes)
% 
% % feature names
% for i = 1:size(FF_feature, 1)
%     all_features_name{end+1} = "FF_feature for channel " + i;
% end


%% calculate cov_Feature -  one covariance matrix for each trial for training data
% clc; close all;
% 
% cov_feature = cov_Feature(Train_Data);
% 
% % normalize features 
% cov_feature = normalize(cov_feature, 0, 1);
% 
% n_class = 2;
% cov_feature_classes = zeros(n_class, int16(size(cov_feature, 1)/n_class),...
%     size(cov_feature, 2), size(cov_feature, 2));
% 
% cov_feature_classes(1,:,:) = cov_feature(class1_trialnum,:);
% cov_feature_classes(2,:,:) = cov_feature(class2_trialnum,:);
% 
% for i = 1:size(cov_feature, 2)
%     all_features{end+1} =  cov_feature_classes(:,:,:,i);
% end
% 
% size(cov_feature_classes)
% 
% % feature names
% for i = 1:size(cov_feature, 2)
%     for j = 1:size(cov_feature, 2)
%         all_features_name{end+1} = "cov_feature between channel " + i + " and " + j;
%     end
% end

%% calculate kurtosis_Feature -  one number for each trial and channel
clc; close all;

kurt_feature = kurtosis_Feature(Train_Data);

% normalize features 
[kurt_featurep, PS] = mapstd(kurt_feature');
kurt_feature = kurt_featurep';

n_class = 2;
kurt_feature_classes = zeros(n_class, int16(size(kurt_feature, 1)/n_class),... 
    size(kurt_feature, 2));

kurt_feature_classes(1,:,:) = kurt_feature(class1_trialnum,:);
kurt_feature_classes(2,:,:) = kurt_feature(class2_trialnum,:);

all_features{end+1} =  kurt_feature_classes;

size(kurt_feature_classes)

% feature names
all_features_name{end+1} = "kurt_feature for channel ";


%%%%%% test data features
kurt_feature_test = kurtosis_Feature(Test_Data);

% normalize features 
kurt_feature_testp = mapstd('apply',kurt_feature_test',PS);
kurt_feature_test = kurt_feature_testp';

all_features_test{end+1} =  kurt_feature_test;

size(kurt_feature_test)

%% calculate skewness_Feature -  one number for each trial and channel
clc; close all;

skewness_feature = skewness_Feature(Train_Data);

% normalize features 
[skewness_featurep, PS] = mapstd(skewness_feature');
skewness_feature = skewness_featurep';

n_class = 2;
skewness_feature_classes = zeros(n_class, int16(size(skewness_feature, 1)/n_class),... 
    size(skewness_feature, 2));

skewness_feature_classes(1,:,:) = skewness_feature(class1_trialnum,:);
skewness_feature_classes(2,:,:) = skewness_feature(class2_trialnum,:);

all_features{end+1} =  skewness_feature_classes;

size(skewness_feature_classes)

% feature names
all_features_name{end+1} = "skew_feature for channel ";

%%%%%% test data features
skewness_feature_test = skewness_Feature(Test_Data);

% normalize features 
skewness_feature_testp = mapstd('apply',skewness_feature_test',PS);
skewness_feature_test = skewness_feature_testp';

all_features_test{end+1} =  skewness_feature_test;

size(skewness_feature_test)

%% calculate lyapunov exponent - one number for each trial and channel
clc; close all;

lyapunov_feature = lyapunov_Feature(Train_Data, Fs);

% normalize features 
[lyapunov_featurep, PS] = mapstd(lyapunov_feature');
lyapunov_feature = lyapunov_featurep';

n_class = 2;
lyapunov_feature_classes = zeros(n_class, int16(size(lyapunov_feature, 1)/n_class),...
    size(lyapunov_feature, 2));

lyapunov_feature_classes(1,:,:) = lyapunov_feature(class1_trialnum,:);
lyapunov_feature_classes(2,:,:) = lyapunov_feature(class2_trialnum,:);

all_features{end+1} =  lyapunov_feature_classes;

size(lyapunov_feature_classes)

% feature names
all_features_name{end+1} = "lyapunov_feature for channel ";


%%%%%% test data features
lyapunov_feature_test = lyapunov_Feature(Test_Data, Fs);

% normalize features 
lyapunov_feature_testp = mapstd('apply',lyapunov_feature_test',PS);
lyapunov_feature_test = lyapunov_feature_testp';

all_features_test{end+1} =  lyapunov_feature_test;

size(lyapunov_feature_test)
%%  calculate entropy - one number for each trial and channel
clc; close all;

entropy_feature = entropy_Feature(Train_Data);

% normalize features 
[entropy_featurep, PS] = mapstd(entropy_feature');
entropy_feature = entropy_featurep';


n_class = 2;
entropy_feature_classes = zeros(n_class, int16(size(entropy_feature, 1)/n_class),...
    size(entropy_feature, 2));

entropy_feature_classes(1,:,:) = entropy_feature(class1_trialnum,:);
entropy_feature_classes(2,:,:) = entropy_feature(class2_trialnum,:);

all_features{end+1} =  entropy_feature_classes;


size(entropy_feature_classes)

% feature names
all_features_name{end+1} = "entropy_feature for channel ";


%%%%%% test data features
entropy_feature_test = entropy_Feature(Test_Data);

% normalize features 
entropy_feature_testp = mapstd('apply',entropy_feature_test',PS);
entropy_feature_test = entropy_feature_testp';

all_features_test{end+1} =  entropy_feature_test;

size(entropy_feature_test)

%%  calculate correlation dimension - one number for each trial and channel
clc; close all;

cordir_feature = cordir_Feature(Train_Data);

% normalize features 
[cordir_featurep, PS] = mapstd(cordir_feature');
cordir_feature = cordir_featurep';

n_class = 2;
cordir_feature_classes = zeros(n_class, int16(size(cordir_feature, 1)/n_class),...
    size(cordir_feature, 2));

cordir_feature_classes(1,:,:) = cordir_feature(class1_trialnum,:);
cordir_feature_classes(2,:,:) = cordir_feature(class2_trialnum,:);

all_features{end+1} =  cordir_feature_classes;


size(cordir_feature_classes)

% feature names
all_features_name{end+1} = "corrDimension_feature for channel ";

%%%%%% test data features
cordir_feature_test = cordir_Feature(Test_Data);

% normalize features 
cordir_feature_testp = mapstd('apply',cordir_feature_test',PS);
cordir_feature_test = cordir_feature_testp';

all_features_test{end+1} =  cordir_feature_test;

size(cordir_feature_test)

%% calculate max_freq_Feature -  one max freq for each trail and channel for training data
% clc; close all;
% 
% [f, max_freq_feature] = max_freq_Feature(Train_Data, Fs);
% 
% normalize features 
% max_freq_feature = normalize(max_freq_feature, 2);
% 
% 
% n_class = 2;
% max_freq_feature_classes = zeros(n_class, int16(size(max_freq_feature, 1)/n_class),...
%     size(max_freq_feature, 2));
% 
% max_freq_feature_classes(1,:,:) = max_freq_feature(class1_trialnum,:);
% max_freq_feature_classes(2,:,:) = max_freq_feature(class2_trialnum,:);
% 
% all_features{end+1} =  max_freq_feature_classes;
% 
% size(max_freq_feature_classes)
% 
% feature names
% for i = 1:size(max_freq_feature_classes, 3)
%     all_features_name{end+1} = "maxfreq_feature for channel " + i;
% end
% 
% %%%%% test data features
% max_freq_feature_test = max_freq_Feature(Test_Data, Fs);
% 
% normalize features 
% max_freq_feature_test = normalize(max_freq_feature_test, 1);
% 
% all_features_test{end+1} =  max_freq_feature_test;
% 
% size(max_freq_feature_test)

%% calculate mean_freq_Feature -  one mean freq for each trail and channel for training data
clc; close all;

mean_freq_feature = mean_freq_Feature(Train_Data, Fs);

% normalize features 
[mean_freq_featurep, PS] = mapstd(mean_freq_feature');
mean_freq_feature = mean_freq_featurep';

n_class = 2;
mean_freq_feature_classes = zeros(n_class, int16(size(mean_freq_feature, 1)/n_class),...
    size(mean_freq_feature, 2));

mean_freq_feature_classes(1,:,:) = mean_freq_feature(class1_trialnum,:);
mean_freq_feature_classes(2,:,:) = mean_freq_feature(class2_trialnum,:);

all_features{end+1} =  mean_freq_feature_classes;

size(mean_freq_feature_classes)

% feature names
all_features_name{end+1} = "meanfreq_feature for channel ";


%%%%%% test data features
mean_freq_feature_test = mean_freq_Feature(Test_Data, Fs);

% normalize features 
mean_freq_feature_testp = mapstd('apply',mean_freq_feature_test',PS);
mean_freq_feature_test = mean_freq_feature_testp';

all_features_test{end+1} =  mean_freq_feature_test;

size(mean_freq_feature_test)

%% calculate median_freq_Feature -  one mean freq for each trail and channel for training data
clc; close all;

med_freq_feature = med_freq_Feature(Train_Data, Fs);

% normalize features 
[med_freq_featurep, PS] = mapstd(med_freq_feature');
med_freq_feature = med_freq_featurep';

n_class = 2;
med_freq_feature_classes = zeros(n_class, int16(size(med_freq_feature, 1)/n_class),...
    size(med_freq_feature, 2));

med_freq_feature_classes(1,:,:) = med_freq_feature(class1_trialnum,:);
med_freq_feature_classes(2,:,:) = med_freq_feature(class2_trialnum,:);

all_features{end+1} =  med_freq_feature_classes;

size(med_freq_feature_classes)


% feature names
all_features_name{end+1} = "medfreq_feature for channel ";


%%%%%% test data features
med_freq_feature_test = med_freq_Feature(Test_Data, Fs);

% normalize features 
med_freq_feature_testp = mapstd('apply',med_freq_feature_test',PS);
med_freq_feature_test = med_freq_feature_testp';

all_features_test{end+1} =  med_freq_feature_test;

size(med_freq_feature_test)

%% selected band energy - one energy for each trail and channel for training data
clc; close all;

band_energy_feature = band_energy_Feature(Train_Data, Fs);

% normalize features
[band_energy_feature_b1, PS1] = mapstd(band_energy_feature(:,:,1)');
band_energy_feature(:,:,1) = band_energy_feature_b1';
[band_energy_feature_b2, PS2] = mapstd(band_energy_feature(:,:,2)');
band_energy_feature(:,:,2) = band_energy_feature_b2';
[band_energy_feature_b3, PS3] = mapstd(band_energy_feature(:,:,3)');
band_energy_feature(:,:,3) = band_energy_feature_b3';
[band_energy_feature_b4, PS4] = mapstd(band_energy_feature(:,:,4)');
band_energy_feature(:,:,4) = band_energy_feature_b4';
[band_energy_feature_b5, PS5] = mapstd(band_energy_feature(:,:,5)');
band_energy_feature(:,:,5) = band_energy_feature_b5';


n_class = 2;
band_energy_feature_classes = zeros(n_class, int16(size(band_energy_feature, 1)/n_class),...
    size(band_energy_feature, 2), size(band_energy_feature, 3));

band_energy_feature_classes(1,:,:,:) = band_energy_feature(class1_trialnum,:,:);
band_energy_feature_classes(2,:,:,:) = band_energy_feature(class2_trialnum,:,:);

for i = 1:size(band_energy_feature, 3)
    all_features{end+1} =  band_energy_feature_classes(:,:,:,i);
end

size(band_energy_feature_classes)

band = ["All range", "Delta", "Theta", "Alpha", "Betha"];
% feature names
for j = 1:size(band_energy_feature_classes, 4)
    all_features_name{end+1} = "energy_feature for band " + band(j) + " channel ";
end


%%%%%% test data features
band_energy_feature_test = band_energy_Feature(Test_Data, Fs);

% normalize features 
band_energy_feature_test_b1 = mapstd('apply',band_energy_feature_test(:,:,1)',PS1);
band_energy_feature_test(:,:,1) = band_energy_feature_test_b1';
band_energy_feature_test_b2 = mapstd('apply',band_energy_feature_test(:,:,2)',PS2);
band_energy_feature_test(:,:,2) = band_energy_feature_test_b2';
band_energy_feature_test_b3 = mapstd('apply',band_energy_feature_test(:,:,3)',PS3);
band_energy_feature_test(:,:,3) = band_energy_feature_test_b3';
band_energy_feature_test_b4 = mapstd('apply',band_energy_feature_test(:,:,4)',PS4);
band_energy_feature_test(:,:,4) = band_energy_feature_test_b4';
band_energy_feature_test_b5 = mapstd('apply',band_energy_feature_test(:,:,5)',PS1);
band_energy_feature_test(:,:,5) = band_energy_feature_test_b5';

for i = 1:size(band_energy_feature_test, 3)
    all_features_test{end+1} =  band_energy_feature_test(:,:,i);
end

size(band_energy_feature_test)

%% Fisher score for each feature
clc; close all;

n_all_features = length(all_features);

n_channel = 30;
fisher_score = zeros(n_all_features, n_channel);


thresh = 0.01;

for i = 1:n_all_features
    for j = 1:n_channel
        selected_channel =  j;
        selected_feature = all_features{i}(:,:,j);
        fisher_score(i,j) = fisher_score_cal(selected_feature, selected_feature(1,:), selected_feature(2,:));
    end
end

[findeds_feat, finded_chan]  = find(fisher_score > thresh);
[length(fisher_score), length(finded_chan)]


%% select the best finded features
clc; close all;

n_class = 2;
n_class_trial = 60;

all_training_features = zeros(n_class, n_class_trial, length(findeds_feat));

for i = 1:length(findeds_feat)
  all_training_features(:,:,i) = all_features{findeds_feat(i)}(:,:,finded_chan(i));
end

size(all_training_features)

%% 
clc; close all;

% for training data
% lets check some features together 
group_features_class1 = squeeze(all_training_features(1,:,:));
group_features_class2 = squeeze(all_training_features(2,:,:));
group_features_both = zeros(120, length(findeds_feat));
group_features_both(class1_trialnum, :) = group_features_class1;
group_features_both(class2_trialnum, :) = group_features_class2;


[size(group_features_class1); size(group_features_class2); size(group_features_both)]

%% j score
clc; close all;

n_rep = 10000;
n_selected_features = 12;
all_selected_features = {};
scores = zeros(n_rep,1);


for j = 1:n_rep
  selected_features = randsample(1:size(group_features_both, 2), n_selected_features);
  all_selected_features{end+1} = selected_features;


  group_1 = group_features_class1(:,selected_features);
  group_2 = group_features_class2(:,selected_features);
  both_groups = group_features_both(:,selected_features);

  S1 = zeros(size(group_1, 2),size(group_1, 2));
  S2 = zeros(size(group_2, 2),size(group_2, 2));

  n_trials = 60;
  n_class = 2;

  mean_all = mean(both_groups, 1);
  mean_class1 = mean(group_1, 1);
  mean_class2 = mean(group_2, 1);
  
  for i = 1:n_trials
    S1 = S1 + 1/n_trials*(group_1(i,:) - mean_class1)' * (group_1(i,:) - mean_class1);
    S2 = S2 + 1/n_trials*(group_2(i,:) - mean_class2)' * (group_2(i,:) - mean_class2);
  end

  Sw = S1+S2;

  % between class matrix
  Sb = zeros(size(group_1, 2),size(group_1, 2));

  Sb = (mean_class1-mean_all)' * (mean_class1-mean_all) +...
 (mean_class2-mean_all)' * (mean_class2-mean_all);


  % final score
  J = trace(Sb)/trace(Sw);
  sprintf('final score is: %f', J)
  scores(j) = J;
end

max(scores)
target_feature = all_selected_features{find(scores == max(scores))};


% name these features
for i = 1:length(target_feature)
    all_features_name{findeds_feat(target_feature(i))} + finded_chan(target_feature(i))
end

%% final feature cell

best_features_found = group_features_both(:,target_feature);
% save('best_features_found','best_features_found')

%% best features index founded eventually 
clc; close all;

best_features_final = [findeds_feat(target_feature), finded_chan(target_feature)];


%% Train MLP
clc; close all;


% accuracy vectors
k_fold = 5;


val_data_starts = [1, 24, 48, 72, 96];

n_rep = 100;
train_acc_reap = zeros(1,n_rep);
val_acc_reap = zeros(1,n_rep);

for j = 1:n_rep
    train_acc = zeros(1,k_fold);
    val_acc = zeros(1,k_fold);
    
    trainFcn = 'trainlm';

    % create the network
    net = fitnet([5,4]  , trainFcn);
    
    for i = 1:k_fold
        train_ind = 1:120;

        % validation data
        val_data = best_features_found(val_data_starts(i):val_data_starts(i)+24,:);
        val_label = Train_Label(val_data_starts(i):val_data_starts(i)+24);
        train_ind(val_data_starts(i):val_data_starts(i)+24) = [];

        % training data
        training_data = best_features_found(train_ind,:);
        training_label = Train_Label(train_ind);

        % train the network and get the predicted outputs
        [net , tr] = train(net, training_data', training_label);
        predicted_y = net(best_features_found');
        predicted_y_val = predicted_y(val_data_starts(i):val_data_starts(i)+24);
        predicted_y_train = predicted_y(train_ind);

        % find the best threshold for making the outputs int
        [X,Y,T,AUC,OPTROCPT] = perfcurve(training_label,predicted_y_train,1);
        T_opt = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

        % binarize the outputs
        predicted_y_val = (predicted_y_val >= T_opt);
        predicted_y_train = (predicted_y_train >= T_opt);

        val_acc(i) = sum(predicted_y_val == val_label)/length(predicted_y_val);
        train_acc(i) = sum(predicted_y_train == training_label)/length(predicted_y_train);
    end
    train_acc_reap(j) = mean(train_acc);
    val_acc_reap(j) = mean(val_acc);
end
acc_val_kfold = mean(val_acc_reap)
train_acc_kfold = mean(train_acc_reap)
std_val_acc = std(val_acc_reap)
std_train_acc = std(train_acc_reap)

%% save the best features for test data
clc; 

n_test_trials = 40;
test_features = zeros(length(target_feature), n_test_trials);

for i = 1:length(target_feature)
    test_features(i,:) = all_features_test{best_features_final(i,1)}(:,best_features_final(i,2));
end

%% train the mlp with all data and get the outputs of MLP for the test data
clc;

n_trials_test = 40;
number_of_rep = 100;
test_labels = zeros(number_of_rep, n_trials_test);
T_opt = zeros(1,number_of_rep);

trainFcn = 'trainlm';

% training data
training_data = best_features_found;
training_label = Train_Label;

i = 1;
while i <= number_of_rep
    % create the network
    net = fitnet([5, 4]  , trainFcn);

    % train the network and get the predicted outputs
    [net , tr] = train(net, training_data', training_label);
    predicted_y = net(best_features_found');
    predicted_y_train = predicted_y;

    % find the best threshold for making the outputs int
    [X,Y,T,AUC,OPTROCPT] = perfcurve(training_label,predicted_y_train,1);
    T_opt(i) = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

    % binarize the outputs
    predicted_y_train = (predicted_y_train >= T_opt(i));
    train_acc = sum(predicted_y_train == training_label)/length(predicted_y_train)

    if(train_acc >= 0.65)
        test_labels(i,:) = net(test_features);
        i = i+1;
    end
end

predicted_y_test = (mean(test_labels, 1) >= mean(T_opt))

save('predicted_y_test_MLP','predicted_y_test')

%% Train RBF

clc; close all;

% accuracy vectors
k_fold = 5;
train_acc_rbf = zeros(1,k_fold);
val_acc_rbf = zeros(1,k_fold);

val_data_starts = [1, 24, 48, 72, 96];

n_rep = 100;
train_acc_reap = zeros(1,n_rep);
val_acc_reap = zeros(1,n_rep);

for j = 1:n_rep
    % train the network and get the predicted outputs
    net = newrb(training_data', training_label, 0.4, 5);
    for i = 1:k_fold
        train_ind = 1:120;

        % validation data
        val_data = best_features_found(val_data_starts(i):val_data_starts(i)+24,:);
        val_label = Train_Label(val_data_starts(i):val_data_starts(i)+24);
        train_ind(val_data_starts(i):val_data_starts(i)+24) = [];

        % training data
        training_data = best_features_found(train_ind,:);
        training_label = Train_Label(train_ind);

        % view(net)
        predicted_y = net(best_features_found');
        predicted_y_val = predicted_y(val_data_starts(i):val_data_starts(i)+24);
        predicted_y_train = predicted_y(train_ind);

        % find the best threshold for making the outputs int
        [X,Y,T,AUC,OPTROCPT] = perfcurve(training_label,predicted_y_train,1);
        T_opt = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

        % binarize the outputs
        predicted_y_val = (predicted_y_val >= T_opt);
        predicted_y_train = (predicted_y_train >= T_opt);

        val_acc_rbf(i) = sum(predicted_y_val == val_label)/length(predicted_y_val);
        train_acc_rbf(i) = sum(predicted_y_train == training_label)/length(predicted_y_train);
    end
    val_acc_reap(j) = mean(val_acc_rbf);
    train_acc_reap(j) = mean(train_acc_rbf);
end
acc_val_kfold_rbf = mean(val_acc_reap)
train_acc_kfold_rbf = mean(train_acc_reap)
std_val_rbf = std(val_acc_reap)
std_train_rbf = std(train_acc_reap)

%% train the rbf with all data and get the outputs of rbf for the test data

clc; close all;

n_trials_test = 40;
number_of_rep = 100;
test_labels_rbf = zeros(number_of_rep, n_trials_test);
T_opt = zeros(1,number_of_rep);


% training data
training_data = best_features_found;
training_label = Train_Label;

i = 1;
while i <= number_of_rep
    % train the network and get the predicted outputs
    net_rbf = newrb(training_data', training_label, 0.4, 5);

    % view(net)
    predicted_y = net_rbf(best_features_found');
    predicted_y_train = predicted_y;

    % find the best threshold for making the outputs int
    [X,Y,T,AUC,OPTROCPT] = perfcurve(training_label,predicted_y_train,1);
    T_opt(i) = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

    % binarize the outputs
    predicted_y_train = (predicted_y_train >= T_opt(i));

    train_acc_rbf = sum(predicted_y_train == training_label)/length(predicted_y_train)

    if(train_acc_rbf >= 0.65)
        % binarize the outputs
        test_labels_rbf(i,:) = net(test_features);
        i = i+1;
    end
end

predicted_y_test_rbf = (mean(test_labels_rbf, 1) >= mean(T_opt))


save('predicted_y_test_rbf','predicted_y_test_rbf')

%% consistancy of test labels with mlp and rbf
consistency_mlp_rbf = sum(predicted_y_test == predicted_y_test_rbf)/40

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% phase2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% feature selection using PARTICLE SWARM OPTIMAZATION

%% give all J > 0.01 features to the pso algorithm and export good features
clc; close all;

n_class = 2;
n_trials = 60;
n_channel = 30;
all_features_mat(1,:,:) = group_features_class1;
all_features_mat(2,:,:) = group_features_class2;

n_trials = 120;
n_features = 12;
swarm_PSO = PSO_feature_selection(all_features_mat, n_features);
selected_features_index = mode(swarm_PSO, 1);
selected_features_PSO(class1_trialnum,:) = all_features_mat(1,:,selected_features_index);
selected_features_PSO(class2_trialnum,:) = all_features_mat(2,:,selected_features_index);


%% save the best features for test data founded by PSO
clc; 

n_test_trials = 40;
test_features_PSO = zeros(length(selected_features_index), n_test_trials);
best_features_final_PSO = [findeds_feat(selected_features_index),...
    finded_chan(selected_features_index)];

for i = 1:length(selected_features_index)
    test_features_PSO(i,:) = all_features_test{best_features_final_PSO(i,1)}...
        (:,best_features_final_PSO(i,2));
end


%% Train MLP
%%%%%%%%%%%%%%%%%%%%%%%%% with PSO features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; close all;

% accuracy vectors
k_fold = 5;
n_rep = 100;
val_acc_pso_rep = zeros(1,n_rep);
train_acc_pso_rep = zeros(1,n_rep);

val_data_starts = [1, 24, 48, 72, 96];

for j = 1:n_rep
    train_acc_pso = zeros(1,k_fold);
    val_acc_pso = zeros(1,k_fold);
    trainFcn = 'trainlm';

    % create the network
    net = fitnet([5, 5]  , trainFcn);
    
    for i = 1:k_fold
        train_ind = 1:120;

        % validation data
        val_data = selected_features_PSO(val_data_starts(i):val_data_starts(i)+24,:);
        val_label = Train_Label(val_data_starts(i):val_data_starts(i)+24);
        train_ind(val_data_starts(i):val_data_starts(i)+24) = [];

        % training data
        training_data = selected_features_PSO(train_ind,:);
        training_label = Train_Label(train_ind);

        % train the network and get the predicted outputs
        [net , tr] = train(net, training_data', training_label);
        predicted_y_pso = net(selected_features_PSO');
        predicted_y_val_pso = predicted_y_pso(val_data_starts(i):val_data_starts(i)+24);
        predicted_y_train_pso = predicted_y_pso(train_ind);

        % find the best threshold for making the outputs int
        [X,Y,T,AUC,OPTROCPT] = perfcurve(training_label,predicted_y_train_pso,1);
        T_opt = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

        % binarize the outputs
        predicted_y_val_pso = (predicted_y_val_pso >= T_opt);
        predicted_y_train_pso = (predicted_y_train_pso >= T_opt);

        val_acc_pso(i) = sum(predicted_y_val_pso == val_label)/length(predicted_y_val_pso);
        train_acc_pso(i) = sum(predicted_y_train_pso == training_label)/length(predicted_y_train_pso);
    end
    val_acc_pso_rep(j) = mean(val_acc_pso);
    train_acc_pso_rep(j) = mean(train_acc_pso);
end

acc_val_kfold_pso = mean(val_acc_pso_rep)
train_acc_kfold_pso = mean(train_acc_pso_rep)
std_val_acc = std(val_acc_pso_rep)
std_train_acc = std(train_acc_pso_rep)

%% train the mlp with all data and get the outputs of MLP for the test data
%%%%%%%%%%%%%%%%%%%%%%%%% with PSO features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;

n_trials_test = 40;
number_of_rep = 100;
test_labels = zeros(number_of_rep, n_trials_test);
T_opt = zeros(1,number_of_rep);

trainFcn = 'trainlm';

% training data
training_data = selected_features_PSO;
training_label = Train_Label;

i = 1;
while i <= number_of_rep
    % create the network
    net = fitnet([5, 5]  , trainFcn);

    % train the network and get the predicted outputs
    [net , tr] = train(net, training_data', training_label);
    predicted_y_pso = net(selected_features_PSO');
    predicted_y_train_pso = predicted_y_pso;

    % find the best threshold for making the outputs int
    [X,Y,T,AUC,OPTROCPT] = perfcurve(training_label,predicted_y_train_pso,1);
    T_opt(i) = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

    % binarize the outputs
    predicted_y_train_pso = (predicted_y_train_pso >= T_opt(i));
    train_acc_pso = sum(predicted_y_train_pso == training_label)/length(predicted_y_train_pso)

    if(train_acc_pso >= 0.75)
        test_labels(i,:) = net(test_features_PSO);
        i = i+1;
    end
end

predicted_y_test_PSO = (mean(test_labels, 1) >= mean(T_opt))

save('predicted_y_test_MLP_PSO','predicted_y_test_PSO')

%% Train RBF
%%%%%%%%%%%%%%%%%%%%%%%%% with PSO features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; close all;

% accuracy vectors
k_fold = 5;

n_rep = 100;
val_acc_pso_rep = zeros(1,n_rep);
train_acc_pso_rep = zeros(1,n_rep);


val_data_starts = [1, 24, 48, 72, 96];

for j = 1:n_rep
    % train the network and get the predicted outputs
    train_acc_rbf_pso = zeros(1,k_fold);
    val_acc_rbf_pso = zeros(1,k_fold);
    net_rbf = newrb(training_data', training_label, 0.6, 20);
    for i = 1:k_fold
        train_ind = 1:120;

        % validation data
        val_data = selected_features_PSO(val_data_starts(i):val_data_starts(i)+24,:);
        val_label = Train_Label(val_data_starts(i):val_data_starts(i)+24);
        train_ind(val_data_starts(i):val_data_starts(i)+24) = [];

        % training data
        training_data = selected_features_PSO(train_ind,:);
        training_label = Train_Label(train_ind);

        % view(net)
        predicted_y_pso = net_rbf(selected_features_PSO');
        predicted_y_val_pso = predicted_y_pso(val_data_starts(i):val_data_starts(i)+24);
        predicted_y_train_pso = predicted_y_pso(train_ind);

        % find the best threshold for making the outputs int
        [X,Y,T,AUC,OPTROCPT] = perfcurve(training_label,predicted_y_train_pso,1);
        T_opt = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

        % binarize the outputs
        predicted_y_val_pso = (predicted_y_val_pso >= T_opt);
        predicted_y_train_pso = (predicted_y_train_pso >= T_opt);

        val_acc_rbf_pso(i) = sum(predicted_y_val_pso == val_label)/length(predicted_y_val_pso);
        train_acc_rbf_pso(i) = sum(predicted_y_train_pso == training_label)/length(predicted_y_train_pso);
    end
    val_acc_pso_rep(j) = mean(val_acc_pso);
    train_acc_pso_rep(j) = mean(train_acc_pso);
end

acc_val_kfold_pso = mean(val_acc_pso_rep)
train_acc_kfold_pso = mean(train_acc_pso_rep)
std_val_acc = std(val_acc_pso_rep)
std_train_acc = std(train_acc_pso_rep)

%% train the rbf with all data and get the outputs of rbf for the test data
%%%%%%%%%%%%%%%%%%%%%%%%% with PSO features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; close all;

n_trials_test = 40;
number_of_rep = 100;
test_labels_rbf = zeros(number_of_rep, n_trials_test);
T_opt = zeros(1,number_of_rep);


% training data
training_data = selected_features_PSO;
training_label = Train_Label;

i = 1;
while i <= number_of_rep
    % train the network and get the predicted outputs
    net_rbf_pso = newrb(training_data', training_label, 0.6, 20);

    % view(net)
    predicted_y_pso = net_rbf_pso(selected_features_PSO');
    predicted_y_train_pso = predicted_y_pso;

    % find the best threshold for making the outputs int
    [X,Y,T,AUC,OPTROCPT] = perfcurve(training_label,predicted_y_train_pso,1);
    T_opt(i) = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

    % binarize the outputs
    predicted_y_train_pso = (predicted_y_train_pso >= T_opt(i));

    train_acc_rbf = sum(predicted_y_train_pso == training_label)/length(predicted_y_train_pso)

    if(train_acc_rbf > 0.65)
        % binarize the outputs
        test_labels_rbf(i,:) = net(test_features_PSO);
        i = i+1;
    end
end

predicted_y_test_rbf_PSO = (mean(test_labels_rbf, 1) >= mean(T_opt))


save('predicted_y_test_rbf_PSO','predicted_y_test_rbf_PSO')

%% consistancy of test labels with mlp and mlp PSO
consistency_mlp_rbf_pso = sum(predicted_y_test_PSO == predicted_y_test_rbf_PSO)/40
consistency_two_features = sum(predicted_y_test_PSO == predicted_y_test)/40
