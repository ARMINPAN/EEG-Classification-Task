function score = fisher_score_cal(bothclass_data,class1_data,class2_data)
  % calculate fisher score for each channel
  
  mean_bothclass = mean(mean(bothclass_data));
  mean_class1 = mean(class1_data);
  mean_class2 = mean(class2_data);

  var_class1 = var(class1_data);
  var_class2 = var(class2_data);

  score = ((mean_bothclass - mean_class1)^2 + (mean_bothclass - mean_class2)^2)/...
      (var_class1 + var_class2);
end