function [cost, grad] = build_local_dcbf_mexable(U, params)
    %#codegen
    
    % 1. DESEMPACOTAMENTO DO VETOR 1D DE PARÂMETROS
    x_k           = params(1:3);      
    target        = params(4:5);
    u_previous    = params(6:7);      
    N             = params(8);
    Ts            = params(9);           
    d_safe        = params(10);
    gamma_cbf     = params(11);        
    rho_cbf       = params(12);
    robot_radius  = params(13);     
    field_gain    = params(14);
    W_track       = params(15);          
    W_goal        = params(16);
    W_terminal_goal = params(17);  
    rho_geofence  = params(18);
    R_v           = params(19); 
    R_w           = params(20); 
    R_dv          = params(21); 
    R_dw          = params(22);
    
    % Parâmetros da sua CBF Analítica Original
    eta_safe      = params(23);       
    eta_obs       = params(24);       
    gamma_safe    = params(25);
    gamma_obs     = params(26);
    h_p2s         = params(27);            
    N_blocks      = params(28);
    N_rot         = params(29);            
    n_neighbors   = params(30);
    agent_id      = params(31);
    
    blocks_params = params(32:111);
    rotatorias    = params(112:131);
    neigh_flat    = params(132:317); 
    
    % U usa a mesma ordem [v1;w1;v2;w2;...], como reshape(2,N).
    controls = reshape(U, 2, N);
    X_pred = zeros(3, N + 1); 
    X_pred(:, 1) = x_k;
    
    stage_x   = zeros(3, N); 
    stage_u   = zeros(2, N);
    cbf_x     = zeros(3, N);   
    cbf_xnext = zeros(3, N);
    cost = 0;
    
    % =====================================================================
    % FORWARD PASS 
    % =====================================================================
    for n = 1:N
        x_n = X_pred(:, n); 
        v_n = controls(1, n); 
        w_n = controls(2, n); 
        theta_n = x_n(3);
        
        [Vx, Vy, dVx_dx, dVx_dy, dVy_dx, dVy_dy] = get_lbm_interpolated_data(x_n(1:2), agent_id);
        Vx = field_gain * Vx;           
        Vy = field_gain * Vy;
        dVx_dx = field_gain * dVx_dx;   
        dVx_dy = field_gain * dVx_dy;
        dVy_dx = field_gain * dVy_dx;   
        dVy_dy = field_gain * dVy_dy;
        
        ex = v_n * cos(theta_n) - Vx; 
        ey = v_n * sin(theta_n) - Vy;
        track_cost = W_track * (ex^2 + ey^2);
        
        goal_error = x_n(1:2) - target;
        goal_cost = W_goal * (goal_error.' * goal_error);
        
        x_next = [x_n(1) + Ts*v_n*cos(theta_n); x_n(2) + Ts*v_n*sin(theta_n); x_n(3) + Ts*w_n];
        X_pred(:, n + 1) = x_next;
        
        % -----------------------------------------------------------------
        % 2. GEOFENCE ANALITICO (CBF)
        % -----------------------------------------------------------------
        [P_geo_n, gP_geo_n]       = calc_point_p2s_penalty_dyn(x_n(1:2), blocks_params, N_blocks, rotatorias, N_rot, h_p2s, robot_radius);
        [P_geo_next, gP_geo_next] = calc_point_p2s_penalty_dyn(x_next(1:2), blocks_params, N_blocks, rotatorias, N_rot, h_p2s, robot_radius);
        
        h_geo_n = -P_geo_n;
        h_geo_next = -P_geo_next;
        g_geo = (1 - gamma_safe) * h_geo_n - h_geo_next;
        
        geofence_cost = eta_safe * max(0, g_geo)^2;
        
        if g_geo > 0
            multiplier_geo = 2 * eta_safe * g_geo;
            cbf_x(1:2, n)     = cbf_x(1:2, n)     + multiplier_geo * ((1 - gamma_safe) * -gP_geo_n);
            cbf_xnext(1:2, n) = cbf_xnext(1:2, n) + multiplier_geo * gP_geo_next;
        end
        
        % -----------------------------------------------------------------
        % 3. OBSTACULOS CIRCULARES ANALITICOS (CBF)
        % -----------------------------------------------------------------
        obs_cost = 0;
        for i_rot = 1:N_rot
            cx = rotatorias(4*(i_rot-1)+1); 
            cy = rotatorias(4*(i_rot-1)+2);
            R_safe = rotatorias(4*(i_rot-1)+4) + robot_radius; 
            
            dist_sq_n    = norm(x_n(1:2) - [cx; cy])^2;
            dist_sq_next = norm(x_next(1:2) - [cx; cy])^2;
            
            v_circ_n    = max(0, R_safe^2 - dist_sq_n);
            v_circ_next = max(0, R_safe^2 - dist_sq_next);
            
            h_circ_n    = -(v_circ_n^2);
            h_circ_next = -(v_circ_next^2);
            
            g_circ = (1 - gamma_obs) * h_circ_n - h_circ_next;
            obs_cost = obs_cost + eta_obs * max(0, g_circ)^2;
            
            if g_circ > 0
                multiplier_obs = 2 * eta_obs * g_circ;
                g_h_circ_n = [0;0]; g_h_circ_next = [0;0];
                
                if v_circ_n > 0, g_h_circ_n = 4 * v_circ_n * (x_n(1:2) - [cx; cy]); end
                if v_circ_next > 0, g_h_circ_next = 4 * v_circ_next * (x_next(1:2) - [cx; cy]); end
                
                cbf_x(1:2, n)     = cbf_x(1:2, n)     + multiplier_obs * ((1 - gamma_obs) * g_h_circ_n);
                cbf_xnext(1:2, n) = cbf_xnext(1:2, n) - multiplier_obs * g_h_circ_next;
            end
        end
        
        input_cost = R_v * v_n^2 + R_w * w_n^2;
        cost = cost + track_cost + goal_cost + geofence_cost + obs_cost + input_cost;
        
        % Gradientes dos termos de tracking e objetivo
        dex_dx = -dVx_dx; dex_dy = -dVx_dy; dex_dtheta = -v_n * sin(theta_n);
        dey_dx = -dVy_dx; dey_dy = -dVy_dy; dey_dtheta = v_n * cos(theta_n);
        
        stage_x(:, n) = 2 * W_track * [ex*dex_dx + ey*dey_dx; ex*dex_dy + ey*dey_dy; ex*dex_dtheta + ey*dey_dtheta];
        stage_x(1:2, n) = stage_x(1:2, n) + 2 * W_goal * goal_error;
        stage_u(:, n) = [2*R_v*v_n + 2*W_track*(ex*cos(theta_n) + ey*sin(theta_n)); 2*R_w*w_n];
        
        % -----------------------------------------------------------------
        % 4. DCBF INTERAGENTE
        % -----------------------------------------------------------------
        for j = 1:n_neighbors
            idx_n = (j-1)*(2*(N+1)) + (n-1)*2;
            p_j_n    = [neigh_flat(idx_n + 1); neigh_flat(idx_n + 2)];
            p_j_next = [neigh_flat(idx_n + 3); neigh_flat(idx_n + 4)];
            
            delta_n    = x_n(1:2) - p_j_n;
            delta_next = x_next(1:2) - p_j_next;
            
            h_n    = delta_n.' * delta_n - d_safe^2;
            h_next = delta_next.' * delta_next - d_safe^2;
            g = (1 - gamma_cbf) * h_n - h_next;
            
            if g > 0
                cost = cost + rho_cbf * g^2;
                multiplier = 2 * rho_cbf * g;
                cbf_x(1:2, n)     = cbf_x(1:2, n)     + multiplier * (1 - gamma_cbf) * (2*delta_n);
                cbf_xnext(1:2, n) = cbf_xnext(1:2, n) - multiplier * (2*delta_next);
            end
        end
    end
    
    % Custo terminal
    terminal_state = X_pred(:, N + 1);
    terminal_position_error = terminal_state(1:2) - target;
    [P_geo_term, gP_geo_term] = calc_point_p2s_penalty_dyn(terminal_state(1:2), blocks_params, N_blocks, rotatorias, N_rot, h_p2s, robot_radius);
    
    terminal_cost = W_terminal_goal * (terminal_position_error.' * terminal_position_error) + rho_geofence * P_geo_term^2;
    cost = cost + terminal_cost;
    
    terminal_gradient = [2*W_terminal_goal*terminal_position_error + 2*rho_geofence*P_geo_term*gP_geo_term; 0];
    
    % Penalidade de suavidade dos comandos
    smooth_grad = zeros(2, N);
    previous_u = u_previous(:);
    for n = 1:N
        du = controls(:, n) - previous_u;
        cost = cost + R_dv * du(1)^2 + R_dw * du(2)^2;
        local_du_grad = [2*R_dv*du(1); 2*R_dw*du(2)];
        smooth_grad(:, n) = smooth_grad(:, n) + local_du_grad;
        if n > 1, smooth_grad(:, n - 1) = smooth_grad(:, n - 1) - local_du_grad; end
        previous_u = controls(:, n);
    end
    
    if nargout == 1, return; end
    
    % =====================================================================
    % BACKWARD PASS (metodo adjunto)
    % =====================================================================
    grad_matrix = zeros(2, N);
    lambda_next = terminal_gradient;
    for n = N:-1:1
        x_n = X_pred(:, n); 
        v_n = controls(1, n); 
        theta_n = x_n(3);
        
        Fx = [1, 0, -Ts*v_n*sin(theta_n); 0, 1,  Ts*v_n*cos(theta_n); 0, 0,  1];
        Fu = [Ts*cos(theta_n), 0; Ts*sin(theta_n), 0; 0, Ts];
        
        transition_costate = lambda_next + cbf_xnext(:, n);
        grad_matrix(:, n) = stage_u(:, n) + Fu.' * transition_costate + smooth_grad(:, n);
        lambda_next = stage_x(:, n) + cbf_x(:, n) + Fx.' * transition_costate;
    end
    grad = grad_matrix(:);
end

% --- Funcoes analiticas compativeis com MATLAB Coder ---
function [val, grad] = phi_func(s, h)
    %#codegen
    if s >= 0, val = (s^3)/(2*(s+h)); grad = (s^2*(2*s+3*h))/(2*(s+h)^2); else, val = 0; grad = 0; end
end

function [P, gP] = get_single_block_p2s(pt, blk, h)
    %#codegen
    [v1, g1] = phi_func(blk(1) - pt(1), h); [v2, g2] = phi_func(pt(1) - blk(2), h);
    [v3, g3] = phi_func(blk(3) - pt(2), h); [v4, g4] = phi_func(pt(2) - blk(4), h);
    P = 0.25 * (v1 + v2 + v3 + v4); gP = [0.25 * (-g1 + g2); 0.25 * (-g3 + g4)];
end

function [M, gM] = smooth_min_func(a, b, ga, gb)
    %#codegen
    diff = a - b; delta = sqrt(diff^2 + 1e-10); 
    M = 0.5 * (a + b - delta); 
    gM = 0.5 * (ga + gb - (diff / delta) * (ga - gb));
end

function [P, gP] = get_bojo_p2s(pt, center, R_bojo, r_rob)
    %#codegen
    dx = pt(1) - center(1); dy = pt(2) - center(2); 
    v = max(0, (dx^2 + dy^2) - (R_bojo - r_rob)^2);
    P = v^2; if v > 0, gP = [4*v*dx; 4*v*dy]; else, gP = [0; 0]; end
end

function [P_total, grad_P] = calc_point_p2s_penalty_dyn(pt, blocks, N_blocks, rotatorias, N_rot, h, r_rob)
    %#codegen
    P_total = 0; grad_P = [0;0];
    first = true;
    for i = 1:N_rot
        [P_b, gP_b] = get_bojo_p2s(pt, rotatorias(4*(i-1)+1 : 4*(i-1)+2)', rotatorias(4*(i-1)+3), r_rob);
        if first, P_total = P_b; grad_P = gP_b; first = false;
        else, [P_total, grad_P] = smooth_min_func(P_total, P_b, grad_P, gP_b); end
    end
    for i = 1:N_blocks
        [P_blk, gP_blk] = get_single_block_p2s(pt, blocks(4*(i-1)+1 : 4*i), h);
        if first, P_total = P_blk; grad_P = gP_blk; first = false;
        else, [P_total, grad_P] = smooth_min_func(P_total, P_blk, grad_P, gP_blk); end
    end
end

function [Vx, Vy, dVx_dx, dVx_dy, dVy_dx, dVy_dy] = get_lbm_interpolated_data(pt, agent_id)
    %#codegen
    persistent LBM
    if isempty(LBM)
        LBM = coder.load('gvf_lbm_data_multi.mat');
    end

    x_vec = LBM.x_vec;
    y_vec = LBM.y_vec;
    dx = x_vec(2) - x_vec(1);
    dy = y_vec(2) - y_vec(1);
    nx = numel(x_vec);
    ny = numel(y_vec);

    % Clampa dentro da última célula válida (não no último nó da malha).
    x = min(max(pt(1), x_vec(1)), x_vec(end) - 1.01*dx);
    y = min(max(pt(2), y_vec(1)), y_vec(end) - 1.01*dy);
    col = floor((x - x_vec(1))/dx) + 1;
    row = floor((y - y_vec(1))/dy) + 1;
    col = min(max(col, 1), nx-1);
    row = min(max(row, 1), ny-1);
    tx = (x - x_vec(col))/dx;
    ty = (y - y_vec(row))/dy;

    % O MEX recebe agent_id 1-based, como os índices MATLAB.
    id = round(agent_id);
    id = min(max(id, 1), size(LBM.ux_all, 3));

    ux00 = LBM.ux_all(row,   col,   id);
    ux10 = LBM.ux_all(row,   col+1, id);
    ux01 = LBM.ux_all(row+1, col,   id);
    ux11 = LBM.ux_all(row+1, col+1, id);
    uy00 = LBM.uy_all(row,   col,   id);
    uy10 = LBM.uy_all(row,   col+1, id);
    uy01 = LBM.uy_all(row+1, col,   id);
    uy11 = LBM.uy_all(row+1, col+1, id);

    Vx = (1-tx)*(1-ty)*ux00 + tx*(1-ty)*ux10 + ...
         (1-tx)*ty*ux01 + tx*ty*ux11;
    Vy = (1-tx)*(1-ty)*uy00 + tx*(1-ty)*uy10 + ...
         (1-tx)*ty*uy01 + tx*ty*uy11;

    dVx_dx = ((1-ty)*(ux10-ux00) + ty*(ux11-ux01))/dx;
    dVx_dy = ((1-tx)*(ux01-ux00) + tx*(ux11-ux10))/dy;
    dVy_dx = ((1-ty)*(uy10-uy00) + ty*(uy11-uy01))/dx;
    dVy_dy = ((1-tx)*(uy01-uy00) + tx*(uy11-uy10))/dy;
end
