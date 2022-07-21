function  skew = skewness_Feature(data)
    % dim1: channel number, dim2: time, dim3: trial number
    skew = squeeze(skewness(data, 0, 2))';
end