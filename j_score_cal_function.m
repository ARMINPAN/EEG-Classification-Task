function J = j_score_cal_function(group_1, group_2, both_groups)

  S1 = zeros(size(group_1, 2),size(group_1, 2));
  S2 = zeros(size(group_2, 2),size(group_2, 2));

  n_trials = 60;
  n_class = 2;

  mean_all = mean(both_groups, 1);
  mean_class1 = mean(group_1, 1);
  mean_class2 = mean(group_2, 1);
  
  for i = 1:n_trials
    S1 = S1 + 1/n_trials*(group_1(i,:) - mean_class1)' * (group_1(i,:) - mean_class1);
    S2 = S2 + 1/n_trials*(group_2(i,:) - mean_class2)' * (group_2(i,:) - mean_class2);
  end

  Sw = S1+S2;

  % between class matrix
  Sb = zeros(size(group_1, 2),size(group_1, 2));

  Sb = (mean_class1-mean_all)' * (mean_class1-mean_all) +...
 (mean_class2-mean_all)' * (mean_class2-mean_all);


  % final score
  J = trace(Sb)/trace(Sw);
end