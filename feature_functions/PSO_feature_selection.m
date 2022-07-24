function x_swarm = PSO_feature_selection(all_features, n_features)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% swarm intialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % each indice is the number of the target feature from 1 to 179
    n_particles = 20;
    x_swarm = zeros(n_particles, n_features);
    
    n_all_features = size(all_features, 3);
    
    for i = 1:n_particles
        x_swarm(i,:) = randperm(n_all_features, n_features);
    end
    
    % j score of the intialized swarm
    all_j_scores = {};
    current_j_score = zeros(1, n_particles);
    
    for i = 1:n_particles
        group_1 = squeeze(all_features(1,:,x_swarm(i,:)));
        group_2 = squeeze(all_features(2,:,x_swarm(i,:)));
        both_groups = cat(1, group_1, group_2);
        current_j_score(i) = j_score_cal_function(group_1, group_2, both_groups);
    end
    all_j_scores{end+1} = current_j_score;
    
    x_best_local = x_swarm;
    x_best_global = x_swarm(find(current_j_score == max(current_j_score)),:);
    max_j = max(current_j_score);
    local_max_j = current_j_score;
    
    % update particles loc
    alpha = 1;
    t = 1;
    b1 = rand(1); b2 = rand(1);
    v = zeros(n_particles, n_features, 2);
    J_thresh = 0.08;
    
    while (sum(all_j_scores{end} >= J_thresh) < n_particles || t < 100)
        for i = 1:n_particles
            % update x and v
            v(i,:,2) = alpha*v(i,:,1) + b1*(x_best_local(i,:) - x_swarm(i,:)) + ...
                b2*(x_best_global - x_swarm(i,:));
            x_swarm(i,:) = round(x_swarm(i,:) + v(i,:,2));
            
            if(length(unique(x_swarm(i,:)) ~= n_features))
                x_swarm(i,:) = round(x_swarm(i,:) - 0.5*rand());
            end
            
            % number of features bounderies [1, 179]
            n_all_features = size(all_features, 3);
            upper_bound = x_swarm(i,:) > n_all_features; x_swarm(i,upper_bound) = n_all_features;
            lower_bound = x_swarm(i,:) < 1; x_swarm(i,lower_bound) = 1;
            
            
            % cal the j score of the updated particle and update xlocal and
            % global
            group_1 = squeeze(all_features(1,:,x_swarm(i,:)));
            group_2 = squeeze(all_features(2,:,x_swarm(i,:)));
            both_groups = cat(1, group_1, group_2);
            prev_j_score = current_j_score(i);
            current_j_score(i) = j_score_cal_function(group_1, group_2, both_groups);
            % update x local
            if(current_j_score(i) >= local_max_j(i))
                x_best_local(i,:) = x_swarm(i,:);
                local_max_j(i) = current_j_score(i);
            end
            % update x global 
            if(max(current_j_score) >= max_j)
                max_j = max(current_j_score);
                x_best_global = x_swarm(find(current_j_score == max_j, 1),:);
            end
            t = t + 1;
            alpha = 1/t;
            b1 = rand(1);
            b2 = rand(1);
        end
        all_j_scores{end+1} = current_j_score;
    end
    all_j_scores{end}
end