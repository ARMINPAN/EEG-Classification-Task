function variance = Var_Feature(data)
    % variance of each channel
    % dim1: channel number, dim2: time, dim3: trial number
    variance = var(data, 0, 2);
end