function params = make_local_params(x0, neighbor_positions, u_previous, ...
                                    target, N, Ts, d_safe, ...
                                    gamma_cbf, rho_cbf, robot_radius, ...
                                    corredores, rotatorias)
    params.N = N;
    params.Ts = Ts;
    params.x0 = x0;
    params.neighbor_positions = neighbor_positions;
    params.u_previous = u_previous;
    params.target = target;
    params.d_safe = d_safe;
    params.gamma_cbf = gamma_cbf;
    params.rho_cbf = rho_cbf;
    
    % Dados Analíticos Injetados
    params.rotatorias = rotatorias;
    params.N_blocks = length(corredores);
    params.blocks_params = zeros(params.N_blocks*4, 1);
    for i = 1:params.N_blocks
        v_c = corredores{i}.v;
        params.blocks_params(4*(i-1) + 1) = min(v_c(:,1)) + robot_radius; 
        params.blocks_params(4*(i-1) + 2) = max(v_c(:,1)) - robot_radius;
        params.blocks_params(4*(i-1) + 3) = min(v_c(:,2)) + robot_radius; 
        params.blocks_params(4*(i-1) + 4) = max(v_c(:,2)) - robot_radius;
    end
    params.h_p2s = 0.05;
    params.eta_safe = 1.0e7;  params.gamma_safe = 0.5;
    params.eta_obs = 1.0e7;   params.gamma_obs = 0.5;

    distance_to_goal = norm(x0(1:2)-target);
    near_goal_weight = exp(-(distance_to_goal/0.25)^2);
    params.field_gain = 1.0 - near_goal_weight;
    params.W_track = 300.0;
    params.W_goal = 0.5 + 25.0*near_goal_weight;
    params.W_terminal_goal = 10.0 + 200.0*near_goal_weight;
    params.robot_radius = robot_radius;
    params.rho_geofence = 2.0e7;
    params.R_v = 0.01;
    params.R_w = 0.01;
    params.R_dv = 0.05;
    params.R_dw = 0.10;
end