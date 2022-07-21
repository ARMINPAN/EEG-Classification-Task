% EEG Classification BCI Task
% Computational Intelligence Course Final Project
% Armin Panjehpour - 98101288

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
clc; close all;

var_feature = squeeze(Var_Feature(Train_Data));

% normalize features 
var_feature = zscore(var_feature, 0, 2);

n_class = 2;
var_feature_classes = zeros(n_class,...
    int16(size(var_feature, 2)/n_class), size(var_feature, 1));

var_feature_classes(1,:,:) = var_feature(:, class1_trialnum)';
var_feature_classes(2,:,:) = var_feature(:, class2_trialnum)';



all_features{end+1} = var_feature_classes;

size(var_feature_classes)

% feature names
for i = 1:size(var_feature, 1)
    all_features_name{end+1} = "var_feature for channel" + i;
end

%% calculate amplitude histogram - nbin values for each channel for training data
clc; close all;

n_bins = 5;
min_amp = -10;
max_amp = 10;
hist_feature = amp_hist_Feature(Train_Data, n_bins, min_amp, max_amp);

% normalize features 
hist_feature = zscore(hist_feature, 0, 2);

n_class = 2;
hist_feature_classes = zeros(n_class, size(hist_feature, 1),...
    int16(size(hist_feature, 2)/n_class), size(hist_feature, 3));

hist_feature_classes(1,:,:,:) = hist_feature(:,class1_trialnum,:);
hist_feature_classes(2,:,:,:) = hist_feature(:,class2_trialnum,:);

for i = 1:size(hist_feature, 3)
    all_features{end+1} = permute(hist_feature_classes(:,:,:,i), [1 3 2 4]);
end

size(hist_feature_classes)

% feature names
for i = 1:size(hist_feature, 1)
    for j = 1:n_bins
        all_features_name{end+1} = "hist_feature for channel" + i + "bin " + j;
    end
end

%% calculate AR coefficients - order values for each trail and channel
clc; close all;

order = 5;
ar_feature = AR_Coeffs(Train_Data, order);

% normalize features 
ar_feature = zscore(ar_feature, 0, 1);

n_class = 2;
ar_feature_classes = zeros(n_class, int16(size(ar_feature, 1)/n_class),...
    size(ar_feature, 2), size(ar_feature, 3));

ar_feature_classes(1,:,:,:) = ar_feature(class1_trialnum,:,:);
ar_feature_classes(2,:,:,:) = ar_feature(class2_trialnum,:,:);

for i = 1:size(ar_feature, 3)
    all_features{end+1} = ar_feature_classes(:,:,:,i);
end

size(ar_feature_classes)

% feature names
for i = 1:size(ar_feature, 2)
    for j = 1:order+1
        all_features_name{end+1} = "ar_feature for channel" + i + "coeff " + j;
    end
end


%% calculate Form Factor -  one value for each channel and trial for training data
clc; close all;

FF_feature = FF_Feature(Train_Data);

% normalize features 
FF_feature = zscore(FF_feature, 0, 2);

n_class = 2;
FF_feature_classes = zeros(n_class, int16(size(FF_feature, 2)/n_class), size(FF_feature, 1));

FF_feature_classes(1,:,:) = FF_feature(:,class1_trialnum)';
FF_feature_classes(2,:,:) = FF_feature(:,class2_trialnum)';

all_features{end+1} =  FF_feature_classes;

size(FF_feature_classes)

% feature names
for i = 1:size(FF_feature, 1)
    all_features_name{end+1} = "FF_feature for channel" + i;
end


%% calculate cov_Feature -  one covariance matrix for each trial for training data
clc; close all;

cov_feature = cov_Feature(Train_Data);

% normalize features 
cov_feature = zscore(cov_feature, 0, 1);

n_class = 2;
cov_feature_classes = zeros(n_class, int16(size(cov_feature, 1)/n_class),...
    size(cov_feature, 2), size(cov_feature, 2));

cov_feature_classes(1,:,:) = cov_feature(class1_trialnum,:);
cov_feature_classes(2,:,:) = cov_feature(class2_trialnum,:);

for i = 1:size(cov_feature, 2)
    all_features{end+1} =  cov_feature_classes(:,:,:,i);
end

size(cov_feature_classes)

% feature names
for i = 1:size(cov_feature, 2)
    for j = 1:size(cov_feature, 2)
        all_features_name{end+1} = "cov_feature between channel" + i + " and " + j;
    end
end

%% calculate kurtosis_Feature -  one number for each trial and channel
clc; close all;

kurt_feature = kurtosis_Feature(Train_Data);

% normalize features 
kurt_feature = zscore(kurt_feature, 0, 1);

n_class = 2;
kurt_feature_classes = zeros(n_class, int16(size(kurt_feature, 1)/n_class),... 
    size(kurt_feature, 2));

kurt_feature_classes(1,:,:) = kurt_feature(class1_trialnum,:);
kurt_feature_classes(2,:,:) = kurt_feature(class2_trialnum,:);

all_features{end+1} =  kurt_feature_classes;

size(kurt_feature_classes)

% feature names
for i = 1:size(kurt_feature_classes, 3)
    all_features_name{end+1} = "kurt_feature for channel" + i;
end



%% calculate skewness_Feature -  one number for each trial and channel
clc; close all;

skewness_feature = skewness_Feature(Train_Data);

% normalize features 
skewness_feature = zscore(skewness_feature, 0, 1);

n_class = 2;
skewness_feature_classes = zeros(n_class, int16(size(skewness_feature, 1)/n_class),... 
    size(skewness_feature, 2));

skewness_feature_classes(1,:,:) = skewness_feature(class1_trialnum,:);
skewness_feature_classes(2,:,:) = skewness_feature(class2_trialnum,:);

all_features{end+1} =  skewness_feature_classes;

size(skewness_feature_classes)

% feature names
for i = 1:size(skewness_feature_classes, 3)
    all_features_name{end+1} = "skew_feature for channel" + i;
end

%% calculate lyapunov exponent - one number for each trial and channel
clc; close all;

lyapunov_feature = lyapunov_Feature(Train_Data, Fs);

% normalize features 
lyapunov_feature = zscore(lyapunov_feature, 0, 1);

n_class = 2;
lyapunov_feature_classes = zeros(n_class, int16(size(lyapunov_feature, 1)/n_class),...
    size(lyapunov_feature, 2));

lyapunov_feature_classes(1,:,:) = lyapunov_feature(class1_trialnum,:);
lyapunov_feature_classes(2,:,:) = lyapunov_feature(class2_trialnum,:);

all_features{end+1} =  lyapunov_feature_classes;

size(lyapunov_feature_classes)

% feature names
for i = 1:size(lyapunov_feature_classes, 3)
    all_features_name{end+1} = "lyapunov_feature for channel" + i;
end

%%  calculate entropy - one number for each trial and channel
clc; close all;

entropy_feature = entropy_Feature(Train_Data);

% normalize features 
entropy_feature = zscore(entropy_feature, 0, 1);

n_class = 2;
entropy_feature_classes = zeros(n_class, int16(size(entropy_feature, 1)/n_class),...
    size(entropy_feature, 2));

entropy_feature_classes(1,:,:) = entropy_feature(class1_trialnum,:);
entropy_feature_classes(2,:,:) = entropy_feature(class2_trialnum,:);

all_features{end+1} =  entropy_feature_classes;


size(entropy_feature_classes)

% feature names
for i = 1:size(entropy_feature_classes, 3)
    all_features_name{end+1} = "entropy_feature for channel" + i;
end

%%  calculate correlation dimension - one number for each trial and channel
clc; close all;

cordir_feature = cordir_Feature(Train_Data);

% normalize features 
cordir_feature = zscore(cordir_feature, 0, 1);

n_class = 2;
cordir_feature_classes = zeros(n_class, int16(size(cordir_feature, 1)/n_class),...
    size(cordir_feature, 2));

cordir_feature_classes(1,:,:) = cordir_feature(class1_trialnum,:);
cordir_feature_classes(2,:,:) = cordir_feature(class2_trialnum,:);

all_features{end+1} =  cordir_feature_classes;


size(cordir_feature_classes)

% feature names
for i = 1:size(cordir_feature_classes, 3)
    all_features_name{end+1} = "corrDimension_feature for channel" + i;
end

%% calculate max_freq_Feature -  one max freq for each trail and channel for training data
clc; close all;

[f, max_freq_feature] = max_freq_Feature(Train_Data, Fs);

% normalize features 
max_freq_feature = zscore(max_freq_feature, 0, 1);


n_class = 2;
max_freq_feature_classes = zeros(n_class, int16(size(max_freq_feature, 1)/n_class),...
    size(max_freq_feature, 2));

max_freq_feature_classes(1,:,:) = max_freq_feature(class1_trialnum,:);
max_freq_feature_classes(2,:,:) = max_freq_feature(class2_trialnum,:);

all_features{end+1} =  max_freq_feature_classes;

size(max_freq_feature_classes)

% feature names
for i = 1:size(max_freq_feature_classes, 3)
    all_features_name{end+1} = "maxfreq_feature for channel" + i;
end


%% calculate mean_freq_Feature -  one mean freq for each trail and channel for training data
clc; close all;

mean_freq_feature = mean_freq_Feature(Train_Data, Fs);

% normalize features 
mean_freq_feature = zscore(mean_freq_feature, 0, 1);

n_class = 2;
mean_freq_feature_classes = zeros(n_class, int16(size(mean_freq_feature, 1)/n_class),...
    size(mean_freq_feature, 2));

mean_freq_feature_classes(1,:,:) = mean_freq_feature(class1_trialnum,:);
mean_freq_feature_classes(2,:,:) = mean_freq_feature(class2_trialnum,:);

all_features{end+1} =  mean_freq_feature_classes;

size(mean_freq_feature_classes)

% feature names
for i = 1:size(mean_freq_feature_classes, 3)
    all_features_name{end+1} = "meanfreq_feature for channel" + i;
end

%% calculate median_freq_Feature -  one mean freq for each trail and channel for training data
clc; close all;

med_freq_feature = med_freq_Feature(Train_Data, Fs);

% normalize features 
med_freq_feature = zscore(med_freq_feature, 0, 1);

n_class = 2;
med_freq_feature_classes = zeros(n_class, int16(size(med_freq_feature, 1)/n_class),...
    size(med_freq_feature, 2));

med_freq_feature_classes(1,:,:) = med_freq_feature(class1_trialnum,:);
med_freq_feature_classes(2,:,:) = med_freq_feature(class2_trialnum,:);

all_features{end+1} =  med_freq_feature_classes;

size(med_freq_feature_classes)


% feature names
for i = 1:size(med_freq_feature_classes, 3)
    all_features_name{end+1} = "medfreq_feature for channel" + i;
end

%% selected band energy - one energy for each trail and channel for training data
clc; close all;

band_energy_feature = band_energy_Feature(Train_Data, Fs);

% normalize features 
band_energy_feature = zscore(band_energy_feature, 0, 1);

n_class = 2;
band_energy_feature_classes = zeros(n_class, int16(size(band_energy_feature, 1)/n_class),...
    size(band_energy_feature, 2), size(band_energy_feature, 3));

band_energy_feature_classes(1,:,:,:) = band_energy_feature(class1_trialnum,:,:);
band_energy_feature_classes(2,:,:,:) = band_energy_feature(class2_trialnum,:,:);

for i = 1:size(band_energy_feature, 3)
    all_features{end+1} =  band_energy_feature_classes(:,:,:,i);
end

size(band_energy_feature_classes)

band = ["Delta", "Theta", "Alpha", "Betha", "Gamma", "All range"];
% feature names
for i = 1:size(band_energy_feature_classes, 3)
    for j = 1:size(band_energy_feature_classes, 4)
        all_features_name{end+1} = "energy_feature for channel" + i + " and band " + band(j);
    end
end


%% Fisher score for each feature
clc; close all;

n_all_features = length(all_features);

n_channel = 30;
fisher_score = {};
high_Fscored_features = {};

thresh = 0.1;

for i = 1:n_all_features
    for j = 1:n_channel
        selected_channel =  j;
        selected_feature = all_features{i}(:,:,j);
        fisher_score{end+1} = fisher_score_cal(selected_feature, selected_feature(1,:), selected_feature(2,:));
        if(fisher_score{end} > thresh)
            high_Fscored_features{end+1} = [i, j];
        end
    end
end

[length(fisher_score), length(high_Fscored_features)]
findeds = find(cell2mat(fisher_score) > thresh);


%% feature cell to matrix and save for python

n_features = length(all_features);
n_class = 2;
n_trial = 60;
n_channels = 30;
all_features_mat = zeros(n_features, n_class, n_trial, n_channels);

for i = 1:n_features
    all_features_mat(i,:,:,:) = all_features{i};
end

size(all_features_mat)

save('all_features_mat','all_features_mat')

%% name of the features founded
all_features_name{findeds};