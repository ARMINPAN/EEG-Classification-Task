function cov_matrix = cov_Feature(data)
    % covariance between pairs of channels
    % dim1: channel number, dim2: time, dim3: trial number

    n_trials = size(data, 3);
    n_samples = size(data, 2);
    n_channels = size(data, 1);

    cov_matrix = zeros(n_trials, n_channels, n_channels);

    for i = 1:n_trials
        for j = 1:n_channels
            for k = 1:n_channels
                selected_data_ch1 = data(j,:,i);
                selected_data_ch2 = data(k,:,i);
                cov_matrix(i, j, k) = cov_calculator(selected_data_ch1,selected_data_ch2);
            end
        end
    end
end