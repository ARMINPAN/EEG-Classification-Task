function med_freq = med_freq_Feature(data, Fs)
    % find the normalized weighted mean of frequencies
    % dim1: channel number, dim2: time, dim3: trial number

    n_trials = size(data, 3);
    n_samples = size(data, 2);
    n_channels = size(data, 1);

    med_freq = zeros(n_trials, n_channels);

    for i = 1:n_trials
        for j = 1:n_channels
            selected_data = data(j,:,i);
            med_freq(i,j) = medfreq(selected_data, Fs);
        end
    end

end