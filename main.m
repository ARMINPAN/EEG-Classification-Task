% EEG Classification BCI Task
% Computational Intelligence Course Final Project
% Armin Panjehpour - 98101288

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
kurt_feature = normalize(kurt_feature, 1);

n_class = 2;
kurt_feature_classes = zeros(n_class, int16(size(kurt_feature, 1)/n_class),... 
    size(kurt_feature, 2));

kurt_feature_classes(1,:,:) = kurt_feature(class1_trialnum,:);
kurt_feature_classes(2,:,:) = kurt_feature(class2_trialnum,:);

all_features{end+1} =  kurt_feature_classes;

size(kurt_feature_classes)

% feature names
for i = 1:size(kurt_feature_classes, 3)
    all_features_name{end+1} = "kurt_feature for channel " + i;
end



%% calculate skewness_Feature -  one number for each trial and channel
clc; close all;

skewness_feature = skewness_Feature(Train_Data);

% normalize features 
skewness_feature = normalize(skewness_feature, 1);

n_class = 2;
skewness_feature_classes = zeros(n_class, int16(size(skewness_feature, 1)/n_class),... 
    size(skewness_feature, 2));

skewness_feature_classes(1,:,:) = skewness_feature(class1_trialnum,:);
skewness_feature_classes(2,:,:) = skewness_feature(class2_trialnum,:);

all_features{end+1} =  skewness_feature_classes;

size(skewness_feature_classes)

% feature names
for i = 1:size(skewness_feature_classes, 3)
    all_features_name{end+1} = "skew_feature for channel " + i;
end

%% calculate lyapunov exponent - one number for each trial and channel
clc; close all;

lyapunov_feature = lyapunov_Feature(Train_Data, Fs);

% normalize features 
lyapunov_feature = normalize(lyapunov_feature, 1);

n_class = 2;
lyapunov_feature_classes = zeros(n_class, int16(size(lyapunov_feature, 1)/n_class),...
    size(lyapunov_feature, 2));

lyapunov_feature_classes(1,:,:) = lyapunov_feature(class1_trialnum,:);
lyapunov_feature_classes(2,:,:) = lyapunov_feature(class2_trialnum,:);

all_features{end+1} =  lyapunov_feature_classes;

size(lyapunov_feature_classes)

% feature names
for i = 1:size(lyapunov_feature_classes, 3)
    all_features_name{end+1} = "lyapunov_feature for channel " + i;
end

%%  calculate entropy - one number for each trial and channel
clc; close all;

entropy_feature = entropy_Feature(Train_Data);

% normalize features 
entropy_feature = normalize(entropy_feature, 1);

n_class = 2;
entropy_feature_classes = zeros(n_class, int16(size(entropy_feature, 1)/n_class),...
    size(entropy_feature, 2));

entropy_feature_classes(1,:,:) = entropy_feature(class1_trialnum,:);
entropy_feature_classes(2,:,:) = entropy_feature(class2_trialnum,:);

all_features{end+1} =  entropy_feature_classes;


size(entropy_feature_classes)

% feature names
for i = 1:size(entropy_feature_classes, 3)
    all_features_name{end+1} = "entropy_feature for channel " + i;
end

%%  calculate correlation dimension - one number for each trial and channel
clc; close all;

cordir_feature = cordir_Feature(Train_Data);

% normalize features 
cordir_feature = normalize(cordir_feature, 1);

n_class = 2;
cordir_feature_classes = zeros(n_class, int16(size(cordir_feature, 1)/n_class),...
    size(cordir_feature, 2));

cordir_feature_classes(1,:,:) = cordir_feature(class1_trialnum,:);
cordir_feature_classes(2,:,:) = cordir_feature(class2_trialnum,:);

all_features{end+1} =  cordir_feature_classes;


size(cordir_feature_classes)

% feature names
for i = 1:size(cordir_feature_classes, 3)
    all_features_name{end+1} = "corrDimension_feature for channel " + i;
end

%% calculate max_freq_Feature -  one max freq for each trail and channel for training data
% clc; close all;
% 
% [f, max_freq_feature] = max_freq_Feature(Train_Data, Fs);
% 
% % normalize features 
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
% % feature names
% for i = 1:size(max_freq_feature_classes, 3)
%     all_features_name{end+1} = "maxfreq_feature for channel " + i;
% end


%% calculate mean_freq_Feature -  one mean freq for each trail and channel for training data
clc; close all;

mean_freq_feature = mean_freq_Feature(Train_Data, Fs);

% normalize features 
mean_freq_feature = normalize(mean_freq_feature, 1);

n_class = 2;
mean_freq_feature_classes = zeros(n_class, int16(size(mean_freq_feature, 1)/n_class),...
    size(mean_freq_feature, 2));

mean_freq_feature_classes(1,:,:) = mean_freq_feature(class1_trialnum,:);
mean_freq_feature_classes(2,:,:) = mean_freq_feature(class2_trialnum,:);

all_features{end+1} =  mean_freq_feature_classes;

size(mean_freq_feature_classes)

% feature names
for i = 1:size(mean_freq_feature_classes, 3)
    all_features_name{end+1} = "meanfreq_feature for channel " + i;
end

%% calculate median_freq_Feature -  one mean freq for each trail and channel for training data
clc; close all;

med_freq_feature = med_freq_Feature(Train_Data, Fs);

% normalize features 
med_freq_feature = normalize(med_freq_feature, 1);

n_class = 2;
med_freq_feature_classes = zeros(n_class, int16(size(med_freq_feature, 1)/n_class),...
    size(med_freq_feature, 2));

med_freq_feature_classes(1,:,:) = med_freq_feature(class1_trialnum,:);
med_freq_feature_classes(2,:,:) = med_freq_feature(class2_trialnum,:);

all_features{end+1} =  med_freq_feature_classes;

size(med_freq_feature_classes)


% feature names
for i = 1:size(med_freq_feature_classes, 3)
    all_features_name{end+1} = "medfreq_feature for channel " + i;
end

%% selected band energy - one energy for each trail and channel for training data
clc; close all;

band_energy_feature = band_energy_Feature(Train_Data, Fs);

% normalize features of each band
band_energy_feature(:,:,1) = normalize(band_energy_feature(:,:,1), 1);
band_energy_feature(:,:,2) = normalize(band_energy_feature(:,:,2), 1);
band_energy_feature(:,:,3) = normalize(band_energy_feature(:,:,3), 1);
band_energy_feature(:,:,4) = normalize(band_energy_feature(:,:,4), 1);
band_energy_feature(:,:,5) = normalize(band_energy_feature(:,:,5), 1);

n_class = 2;
band_energy_feature_classes = zeros(n_class, int16(size(band_energy_feature, 1)/n_class),...
    size(band_energy_feature, 2), size(band_energy_feature, 3));

band_energy_feature_classes(1,:,:,:) = band_energy_feature(class1_trialnum,:,:);
band_energy_feature_classes(2,:,:,:) = band_energy_feature(class2_trialnum,:,:);

for i = 1:size(band_energy_feature, 3)
    all_features{end+1} =  band_energy_feature_classes(:,:,:,i);
end

size(band_energy_feature_classes)

band = [ "All range","Delta", "Theta", "Alpha", "Betha"];
% feature names
for i = 1:size(band_energy_feature_classes, 3)
    for j = 1:size(band_energy_feature_classes, 4)
        all_features_name{end+1} = "energy_feature for channel " + i + " and band " + band(j);
    end
end


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
group_features_both = zeros(120, 179);
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

%% final feature cell


best_features_found = group_features_both(:,target_feature);
save('best_features_found','best_features_found')

%% name of the features founded
all_features_name{target_feature}

%% Train MLP
clc; close all;


trainFcn = 'trainlm';

net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

% create the network
net = fitnet([20,10]  , trainFcn);
view(net)

% accuracy vectors
k_fold = 5;
train_acc = zeros(1,k_fold);
val_acc = zeros(1,k_fold);

val_data_starts = [ 1 , 24 , 48 , 72 , 96];

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

acc_val_kfold = mean(val_acc)
train_acc_kfold = mean(train_acc)


