function hist_vals = amp_hist_Feature(data, n_bins, min_amp, max_amp)
  % amplitude histogram
  % dim1: channel number, dim2: time, dim3: trial number
  n_channels = size(data, 1);
  n_trials = size(data, 3);
  
  hist_vals = zeros(n_channels,n_trials,n_bins);
  
    for i = 1:n_channels
        for j = 1:n_trials
            in_range = find(min_amp <= data(i,:,j) & data(i,:,j) <= max_amp);
            selected_chan_data = data(i, in_range, j);
            hist_vals(i,j,:) = histogram(selected_chan_data, n_bins).Values;
        end
    end
    close all;
end