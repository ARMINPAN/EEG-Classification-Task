function Coeffs = AR_Coeffs(data, order)
    % autoregressive model coefficients
    % dim1: channel number, dim2: time, dim3: trial number
    n_trials = size(data, 3);
    n_samples = size(data, 2);
    n_channels = size(data, 1);

    Y = zeros(n_channels, n_samples-order, n_trials);
    Y = data(:, order:n_samples-1, :);

    X = zeros(n_trials, n_channels, n_samples-order, order+1);
    X(:,:,:,1) = 1;

    Coeffs = zeros(n_trials, n_channels, order+1);

    for i = 1:n_trials
        for j = 1:n_channels
            for k = 1:n_samples-order
                for z = 1:order
                    if(k-z >= 1)
                        X(i, j, k, order-z+1) = data(j, k-z+1, i);
                    else
                        X(i, j, k, order-z+1) = 0;
                    end
                end
            end
            a = ((squeeze(X(i,j,:,:))' * squeeze(X(i,j,:,:))) + (0.00001*rand(order+1)));
            Coeffs(i,j,:) = inv(a) * (squeeze(X(i,j,:,:))') * Y(j,:,i)';
        end
    end

end