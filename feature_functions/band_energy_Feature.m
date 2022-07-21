function band_energy = band_energy_Feature(data, Fs)
    % calculate energy of 5 bands , all freq energy
    % dim1: channel number, dim2: time, dim3: trial number
    n_trials = size(data, 3);
    n_samples = size(data, 2);
    n_channels = size(data, 1);
    n_bands = 6;

    band_energy = zeros(n_trials, n_channels, n_bands);
    freqs = [[0.5, 3]; [4, 7]; [8, 12]; [12, 30]; [30, 100]; [0.5, 100]];

    for i = 1:n_trials
        for j = 1:n_channels
            for k = 1:n_bands
                selected_data = data(j,:,i);
                band_energy(i,j,k) = bandpower(selected_data, Fs, freqs(k,:));
            end
        end
    end
end