function  kurt = kurtosis_Feature(data)
    % dim1: channel number, dim2: time, dim3: trial number
    kurt = squeeze(kurtosis(data, 0, 2))';
end