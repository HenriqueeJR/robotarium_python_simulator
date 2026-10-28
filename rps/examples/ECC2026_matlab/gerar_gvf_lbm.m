function [X_grid, Y_grid, ux, uy] = gerar_gvf_lbm(corredores, rotatorias, conexoes, pt_target, map_bounds, res)
    x_vec = map_bounds(1) : res : map_bounds(2);
    y_vec = map_bounds(3) : res : map_bounds(4);
    Nx = length(x_vec); Ny = length(y_vec);
    [X_grid, Y_grid] = meshgrid(x_vec, y_vec);
    
    obstacles = true(Ny, Nx); 
    F_accum_x = zeros(Ny, Nx); F_accum_y = zeros(Ny, Nx); F_count = zeros(Ny, Nx);
    D_x = zeros(Ny, Nx); D_y = zeros(Ny, Nx);
    
    % 1. Esculpir Corredores e Somar Forças nas Esquinas (C->C)
    for i = 1:length(corredores)
        mask = inpolygon(X_grid, Y_grid, corredores{i}.v(:,1), corredores{i}.v(:,2));
        obstacles(mask) = false;
        F_accum_x(mask) = F_accum_x(mask) + corredores{i}.dir(1);
        F_accum_y(mask) = F_accum_y(mask) + corredores{i}.dir(2);
        F_count(mask) = F_count(mask) + 1;
    end
    mask_valid = F_count > 0;
    D_x(mask_valid) = F_accum_x(mask_valid) ./ F_count(mask_valid);
    D_y(mask_valid) = F_accum_y(mask_valid) ./ F_count(mask_valid);
    
    % 2. Esculpir Rotatórias
    theta_poly = linspace(0, 2*pi, 32);
    for i = 1:size(rotatorias, 1)
        r = rotatorias(i,:);
        mask_ext = (X_grid - r(1)).^2 + (Y_grid - r(2)).^2 <= r(3)^2;
        mask_int = (X_grid - r(1)).^2 + (Y_grid - r(2)).^2 <= r(4)^2;
        obstacles(mask_ext) = false; obstacles(mask_int) = true; 
        D_x(mask_ext & ~obstacles) = -r(5) * (Y_grid(mask_ext & ~obstacles) - r(2));
        D_y(mask_ext & ~obstacles) =  r(5) * (X_grid(mask_ext & ~obstacles) - r(1));
    end
    
    % 3. Paredes Virtuais via Análise de Centroides (Infalível)
    for i = 1:size(conexoes, 1)
        C_in = corredores{conexoes(i,1)}; C_out = corredores{conexoes(i,2)}; rot = rotatorias(conexoes(i,3), :);
        
        poly_R = polyshape(rot(1) + rot(3)*cos(theta_poly), rot(2) + rot(3)*sin(theta_poly));
        poly_in = polyshape(C_in.v(:,1), C_in.v(:,2)); poly_out = polyshape(C_out.v(:,1), C_out.v(:,2));
        
        [cx_in, cy_in] = centroid(intersect(poly_R, poly_in));
        [cx_out, cy_out] = centroid(intersect(poly_R, poly_out));
        
        a_in = atan2(cy_in - rot(2), cx_in - rot(1));
        a_out = atan2(cy_out - rot(2), cx_out - rot(1));
        
        % Delta é a viagem CCW do fluido. O setor a bloquear é o complemento (2*pi - delta)
        delta_travel = mod(a_out - a_in, 2*pi);
        a_mid_block = mod(a_out + (2*pi - delta_travel)/2, 2*pi);
        half_width = max(0, (2*pi - delta_travel)/2 - 0.45); % 0.45 rad = ~25 graus de margem
        
        ang_grid = atan2(Y_grid - rot(2), X_grid - rot(1));
        ang_diff = mod(ang_grid - a_mid_block + pi, 2*pi) - pi;
        wedge = abs(ang_diff) <= half_width;
        
        mask_rot = (X_grid - rot(1)).^2 + (Y_grid - rot(2)).^2 <= rot(3)^2;
        obstacles(mask_rot & wedge) = true;
    end
    obstacles(1,:) = true; obstacles(end,:) = true; obstacles(:,1) = true; obstacles(:,end) = true;
    
    % 4. LBM Core
    norm_D = sqrt(D_x.^2 + D_y.^2) + 1e-10;
    Fx_base = 0.0015 * (D_x ./ norm_D); Fy_base = 0.0015 * (D_y ./ norm_D);
    Fx_base(obstacles) = 0; Fy_base(obstacles) = 0;
    mask_sink = (X_grid - pt_target(1)).^2 + (Y_grid - pt_target(2)).^2 <= 0.15^2 & ~obstacles;
    
    tau = 0.8; iteracoes = 3000; w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];
    cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]; cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    rho = ones(Ny, Nx); ux = zeros(Ny, Nx); uy = zeros(Ny, Nx); f = zeros(Ny, Nx, 9);
    for i = 1:9, f(:,:,i) = w(i) * rho; end
    
    for iter = 1:iteracoes
        rho = sum(f, 3); ux = zeros(Ny, Nx); uy = zeros(Ny, Nx);
        for i = 1:9, ux = ux + cx(i) * f(:,:,i); uy = uy + cy(i) * f(:,:,i); end
        fluid = ~obstacles; Fx = Fx_base - 0.02 * ux; Fy = Fy_base - 0.02 * uy;
        ux(fluid) = (ux(fluid) + 0.5 * Fx(fluid)) ./ rho(fluid); uy(fluid) = (uy(fluid) + 0.5 * Fy(fluid)) ./ rho(fluid);
        ux(obstacles) = 0; uy(obstacles) = 0;
        f_out = zeros(Ny, Nx, 9);
        for i = 1:9
            c_dot_u = cx(i)*ux + cy(i)*uy; c_dot_F = cx(i)*Fx + cy(i)*Fy; u_dot_F = ux.*Fx + uy.*Fy;
            f_eq = w(i) * rho .* (1 + 3*c_dot_u + 4.5*(c_dot_u.^2) - 1.5*(ux.^2 + uy.^2));
            f_out(:,:,i) = f(:,:,i) - (f(:,:,i) - f_eq) / tau + (1 - 0.5/tau) * w(i) * (3*c_dot_F + 9*(c_dot_u.*c_dot_F) - 3*u_dot_F); 
        end
        f_streamed = zeros(Ny, Nx, 9);
        for i = 1:9, f_streamed(:,:,i) = circshift(f_out(:,:,i), [cy(i), cx(i)]); end
        oposto = [1, 4, 5, 2, 3, 8, 9, 6, 7]; f = f_streamed;
        for i = 2:9
            f_temp = f(:,:,i); f_op = f_streamed(:,:,oposto(i)); f_temp(obstacles) = f_op(obstacles); f(:,:,i) = f_temp;
        end
        for i = 1:9, f_temp = f(:,:,i); f_temp(mask_sink) = w(i) * 0.15; f(:,:,i) = f_temp; end
    end
end