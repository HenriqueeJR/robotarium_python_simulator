function [cost, grad, X_pred, diagnostics] = build_local_dcbf(U, params, field)
%BUILD_LOCAL_DCBF NMPC Multiagente com CBF Analítico de Geofence e DCBF Interagente.
    N = params.N;
    Ts = params.Ts;
    U = U(:);
    controls = reshape(U, 2, N);
    X_pred = zeros(3, N + 1);
    X_pred(:, 1) = params.x0(:);
    
    stage_x = zeros(3, N);
    stage_u = zeros(2, N);
    cbf_x = zeros(3, N);
    cbf_xnext = zeros(3, N);
    
    cost = 0;
    max_dcbf_violation = 0;
    max_geofence_violation = 0;
    min_distance = inf;
    
    if isempty(params.neighbor_positions)
        n_neighbors = 0;
    else
        n_neighbors = size(params.neighbor_positions, 3);
    end
    
    for n = 1:N
        x_n = X_pred(:, n);
        v_n = controls(1, n);
        omega_n = controls(2, n);
        theta_n = x_n(3);
        
        % 1. CAMPO VETORIAL
        [Vx, Vy, dVx_dx, dVx_dy, dVy_dx, dVy_dy] = interpolate_vector_field(field, x_n(1:2));
        Vx = params.field_gain * Vx;           Vy = params.field_gain * Vy;
        dVx_dx = params.field_gain * dVx_dx;   dVx_dy = params.field_gain * dVx_dy;
        dVy_dx = params.field_gain * dVy_dx;   dVy_dy = params.field_gain * dVy_dy;
        
        ex = v_n * cos(theta_n) - Vx;
        ey = v_n * sin(theta_n) - Vy;
        track_cost = params.W_track * (ex^2 + ey^2);
        
        goal_error = x_n(1:2) - params.target;
        goal_cost = params.W_goal * (goal_error.' * goal_error);
        
        x_next = unicycle_step(x_n, controls(:, n), Ts);
        X_pred(:, n + 1) = x_next;
        
        % -----------------------------------------------------------------
        % 2. GEOFENCE ANALITICO (CBF)
        % -----------------------------------------------------------------
        [P_geo_n, gP_geo_n] = calc_point_p2s_penalty_dyn(x_n(1:2), params.blocks_params, params.N_blocks, params.rotatorias, params.h_p2s, params.robot_radius);
        [P_geo_next, gP_geo_next] = calc_point_p2s_penalty_dyn(x_next(1:2), params.blocks_params, params.N_blocks, params.rotatorias, params.h_p2s, params.robot_radius);
        
        h_geo_n = -P_geo_n;
        h_geo_next = -P_geo_next;
        g_geo = (1 - params.gamma_safe) * h_geo_n - h_geo_next;
        
        geofence_cost = params.eta_safe * max(0, g_geo)^2;
        max_geofence_violation = max(max_geofence_violation, max(0, g_geo));
        
        if g_geo > 0
            multiplier_geo = 2 * params.eta_safe * g_geo;
            cbf_x(1:2, n)     = cbf_x(1:2, n)     + multiplier_geo * ((1 - params.gamma_safe) * -gP_geo_n);
            cbf_xnext(1:2, n) = cbf_xnext(1:2, n) + multiplier_geo * gP_geo_next;
        end
        
        % -----------------------------------------------------------------
        % 3. OBSTACULOS CIRCULARES ANALITICOS (CBF)
        % -----------------------------------------------------------------
        obs_cost = 0;
        N_rot = size(params.rotatorias, 1);
        for i_rot = 1:N_rot
            center = params.rotatorias(i_rot, 1:2)';
            R_safe = params.rotatorias(i_rot, 4) + params.robot_radius; % Raio interno (ilha) + margem
            
            dist_sq_n = norm(x_n(1:2) - center)^2;
            dist_sq_next = norm(x_next(1:2) - center)^2;
            
            v_circ_n = max(0, R_safe^2 - dist_sq_n);
            v_circ_next = max(0, R_safe^2 - dist_sq_next);
            
            h_circ_n = -(v_circ_n^2);
            h_circ_next = -(v_circ_next^2);
            
            g_circ = (1 - params.gamma_obs) * h_circ_n - h_circ_next;
            obs_cost = obs_cost + params.eta_obs * max(0, g_circ)^2;
            
            if g_circ > 0
                multiplier_obs = 2 * params.eta_obs * g_circ;
                g_h_circ_n = [0;0]; g_h_circ_next = [0;0];
                
                if v_circ_n > 0, g_h_circ_n = 4 * v_circ_n * (x_n(1:2) - center); end
                if v_circ_next > 0, g_h_circ_next = 4 * v_circ_next * (x_next(1:2) - center); end
                
                cbf_x(1:2, n)     = cbf_x(1:2, n)     + multiplier_obs * ((1 - params.gamma_obs) * g_h_circ_n);
                cbf_xnext(1:2, n) = cbf_xnext(1:2, n) - multiplier_obs * g_h_circ_next;
            end
        end
        
        input_cost = params.R_v * v_n^2 + params.R_w * omega_n^2;
        cost = cost + track_cost + goal_cost + geofence_cost + obs_cost + input_cost;
        
        % Gradientes dos termos de tracking e objetivo
        dex_dx = -dVx_dx; dex_dy = -dVx_dy; dex_dtheta = -v_n * sin(theta_n);
        dey_dx = -dVy_dx; dey_dy = -dVy_dy; dey_dtheta = v_n * cos(theta_n);
        
        stage_x(:, n) = 2 * params.W_track * [ex*dex_dx + ey*dey_dx; ex*dex_dy + ey*dey_dy; ex*dex_dtheta + ey*dey_dtheta];
        stage_x(1:2, n) = stage_x(1:2, n) + 2 * params.W_goal * goal_error;
        
        stage_u(:, n) = [2*params.R_v*v_n + 2*params.W_track*(ex*cos(theta_n) + ey*sin(theta_n)); 2*params.R_w*omega_n];
        
        % -----------------------------------------------------------------
        % 4. DCBF INTERAGENTE
        % -----------------------------------------------------------------
        for j = 1:n_neighbors
            p_j_n = params.neighbor_positions(:, n, j);
            p_j_next = params.neighbor_positions(:, n + 1, j);
            delta_n = x_n(1:2) - p_j_n;
            delta_next = x_next(1:2) - p_j_next;
            dist_n = norm(delta_n);
            dist_next = norm(delta_next);
            min_distance = min(min_distance, min(dist_n, dist_next));
            
            h_n = delta_n.' * delta_n - params.d_safe^2;
            h_next = delta_next.' * delta_next - params.d_safe^2;
            g = (1 - params.gamma_cbf) * h_n - h_next;
            positive_g = max(0, g);
            
            cost = cost + params.rho_cbf * positive_g^2;
            max_dcbf_violation = max(max_dcbf_violation, positive_g);
            
            if g > 0
                multiplier = 2 * params.rho_cbf * g;
                cbf_x(1:2, n)     = cbf_x(1:2, n)     + multiplier * (1 - params.gamma_cbf) * (2*delta_n);
                cbf_xnext(1:2, n) = cbf_xnext(1:2, n) - multiplier * (2*delta_next);
            end
        end
    end
    
    % Custo Terminal
    terminal_state = X_pred(:, N + 1);
    terminal_position_error = terminal_state(1:2) - params.target;
    [P_geo_term, gP_geo_term] = calc_point_p2s_penalty_dyn(terminal_state(1:2), params.blocks_params, params.N_blocks, params.rotatorias, params.h_p2s, params.robot_radius);
    
    terminal_cost = params.W_terminal_goal * (terminal_position_error.' * terminal_position_error) + params.rho_geofence * P_geo_term^2;
    cost = cost + terminal_cost;
    
    terminal_gradient = [2*params.W_terminal_goal*terminal_position_error + 2*params.rho_geofence*P_geo_term*gP_geo_term; 0];
    
    % Suavidade
    smooth_grad = zeros(2, N);
    previous_u = params.u_previous(:);
    for n = 1:N
        du = controls(:, n) - previous_u;
        cost = cost + params.R_dv * du(1)^2 + params.R_dw * du(2)^2;
        local_du_grad = [2*params.R_dv*du(1); 2*params.R_dw*du(2)];
        smooth_grad(:, n) = smooth_grad(:, n) + local_du_grad;
        if n > 1, smooth_grad(:, n - 1) = smooth_grad(:, n - 1) - local_du_grad; end
        previous_u = controls(:, n);
    end
    
    diagnostics.max_dcbf_violation = max_dcbf_violation;
    diagnostics.max_geofence_violation = max_geofence_violation;
    diagnostics.min_predicted_distance = min_distance;
    
    if nargout < 2, return; end
    
    % Backward Pass 
    grad_matrix = zeros(2, N);
    lambda_next = terminal_gradient;
    for n = N:-1:1
        x_n = X_pred(:, n); v_n = controls(1, n); theta_n = x_n(3);
        Fx = [1, 0, -Ts*v_n*sin(theta_n); 0, 1,  Ts*v_n*cos(theta_n); 0, 0,  1];
        Fu = [Ts*cos(theta_n), 0; Ts*sin(theta_n), 0; 0, Ts];
        
        transition_costate = lambda_next + cbf_xnext(:, n);
        grad_matrix(:, n) = stage_u(:, n) + Fu.' * transition_costate + smooth_grad(:, n);
        lambda_next = stage_x(:, n) + cbf_x(:, n) + Fx.' * transition_costate;
    end
    grad = grad_matrix(:);
end

% =========================================================================
% FUNÇÕES ANALÍTICAS DE GEOFENCE RESTAURADAS
% =========================================================================
function [val, grad] = phi_func(s, h)
    if s >= 0, val = (s^3)/(2*(s+h)); grad = (s^2*(2*s+3*h))/(2*(s+h)^2); else, val = 0; grad = 0; end
end

function [P, gP] = get_single_block_p2s(pt, blk, h)
    [v1, g1] = phi_func(blk(1) - pt(1), h); [v2, g2] = phi_func(pt(1) - blk(2), h);
    [v3, g3] = phi_func(blk(3) - pt(2), h); [v4, g4] = phi_func(pt(2) - blk(4), h);
    P = 0.25 * (v1 + v2 + v3 + v4); gP = [0.25 * (-g1 + g2); 0.25 * (-g3 + g4)];
end

function [M, gM] = smooth_min_func(a, b, ga, gb)
    diff = a - b; delta = sqrt(diff^2 + 1e-10); 
    M = 0.5 * (a + b - delta); 
    gM = 0.5 * (ga + gb - (diff / delta) * (ga - gb));
end

function [P, gP] = get_bojo_p2s(pt, center, R_bojo, r_rob)
    dx = pt(1) - center(1); dy = pt(2) - center(2); 
    v = max(0, (dx^2 + dy^2) - (R_bojo - r_rob)^2);
    P = v^2; if v > 0, gP = [4*v*dx; 4*v*dy]; else, gP = [0; 0]; end
end

function [P_total, grad_P] = calc_point_p2s_penalty_dyn(pt, blocks, N_blocks, rotatorias, h, r_rob)
    % Loop dinâmico pelas rotatórias e corredores
    N_rot = size(rotatorias, 1);
    P_total = 0; grad_P = [0;0];
    first = true;
    for i = 1:N_rot
        [P_b, gP_b] = get_bojo_p2s(pt, rotatorias(i, 1:2)', rotatorias(i, 3), r_rob);
        if first, P_total = P_b; grad_P = gP_b; first = false;
        else, [P_total, grad_P] = smooth_min_func(P_total, P_b, grad_P, gP_b); end
    end
    for i = 1:N_blocks
        [P_blk, gP_blk] = get_single_block_p2s(pt, blocks(4*(i-1)+1 : 4*i), h);
        if first, P_total = P_blk; grad_P = gP_blk; first = false;
        else, [P_total, grad_P] = smooth_min_func(P_total, P_blk, grad_P, gP_blk); end
    end
end

% =========================================================================
% FUNCOES AUXILIARES DE CINEMATICA E INTERPOLACAO LBM
% =========================================================================
function x_next = unicycle_step(x, u, Ts)
    theta = x(3);
    x_next = [x(1) + Ts*u(1)*cos(theta); ...
              x(2) + Ts*u(1)*sin(theta); ...
              x(3) + Ts*u(2)];
end

function [Vx, Vy, dVx_dx, dVx_dy, dVy_dx, dVy_dy] = ...
        interpolate_vector_field(field, position)
% Interpolacao bilinear e derivadas analiticas dentro da celula da malha.
    x_vec = field.x_vec;
    y_vec = field.y_vec;
    dx = x_vec(2) - x_vec(1);
    dy = y_vec(2) - y_vec(1);
    x = min(max(position(1), x_vec(1)), x_vec(end) - 1.01*dx);
    y = min(max(position(2), y_vec(1)), y_vec(end) - 1.01*dy);
    col = floor((x - x_vec(1))/dx) + 1;
    row = floor((y - y_vec(1))/dy) + 1;
    col = min(max(col, 1), numel(x_vec) - 1);
    row = min(max(row, 1), numel(y_vec) - 1);
    tx = (x - x_vec(col))/dx;
    ty = (y - y_vec(row))/dy;
    ux00 = field.ux(row, col);
    ux10 = field.ux(row, col + 1);
    ux01 = field.ux(row + 1, col);
    ux11 = field.ux(row + 1, col + 1);
    uy00 = field.uy(row, col);
    uy10 = field.uy(row, col + 1);
    uy01 = field.uy(row + 1, col);
    uy11 = field.uy(row + 1, col + 1);
    Vx = (1-tx)*(1-ty)*ux00 + tx*(1-ty)*ux10 + ...
         (1-tx)*ty*ux01 + tx*ty*ux11;
    Vy = (1-tx)*(1-ty)*uy00 + tx*(1-ty)*uy10 + ...
         (1-tx)*ty*uy01 + tx*ty*uy11;
    dVx_dx = ((1-ty)*(ux10-ux00) + ty*(ux11-ux01))/dx;
    dVx_dy = ((1-tx)*(ux01-ux00) + tx*(ux11-ux10))/dy;
    dVy_dx = ((1-ty)*(uy10-uy00) + ty*(uy11-uy01))/dx;
    dVy_dy = ((1-tx)*(uy01-uy00) + tx*(uy11-uy10))/dy;
end
