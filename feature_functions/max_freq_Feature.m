function [f, max_freq] = max_freq_Feature(data, Fs)
    % find the frequency with the maximum amplitude for each channel
    % dim1: channel number, dim2: time, dim3: trial number

    n_trials = size(data, 3);
    n_samples = size(data, 2);
    n_channels = size(data, 1);

    max_freq = zeros(n_trials, n_channels);

    for i = 1:n_trials
        for j = 1:n_channels
          selected_data = data(j,:,i);
          [f, fft_selected_data] = fft_calculator(selected_data, Fs);
          max_freq(i,j) = f(find(fft_selected_data == max(fft_selected_data)));
        end
    end
  
end