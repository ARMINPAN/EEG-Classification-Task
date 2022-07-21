function  skew = skewness_Feature(data)
    % dim1: channel number, dim2: time, dim3: trial number
    
    n_trials = size(data, 3);
    n_samples = size(data, 2);
    n_channels = size(data, 1);

    skew = zeros(n_trials, n_channels);
    
    for i = 1:n_trials
        for j = 1:n_channels
            selected_data = data(j,:,i);
            skew(i,j) =  skewness(selected_data);
        end
    end
end