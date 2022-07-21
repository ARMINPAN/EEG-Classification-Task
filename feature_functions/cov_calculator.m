function covariance =  cov_calculator(data1, data2)
    % covariancce calculator between two vector signals
    cov_mat = cov(data1, data2);
    covariance = cov_mat(1,2);
end 