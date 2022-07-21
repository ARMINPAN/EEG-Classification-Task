function FF = FF_Feature(data)
    % form factor
    % dim1: channel number, dim2: time, dim3: trial number
    signal_std = squeeze(std(data, 0, 2));
    first_deriv_Std = squeeze(std(diff(data, 1, 2), 0, 2));
    second_deriv_Std = squeeze(std(diff(diff(data, 1, 2), 1, 2), 0, 2));
    FF = (second_deriv_Std./first_deriv_Std)./(first_deriv_Std./signal_std);
end