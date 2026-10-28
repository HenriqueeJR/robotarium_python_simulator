function report = test_equivalencia_codegen()
%TEST_EQUIVALENCIA_CODEGEN Valida o empacotamento e a formulacao mexable.
%
% Este teste nao requer o Robotarium nem um compilador C++. Primeiro ele
% compara a funcao MATLAB original com a fonte preparada para o Codegen.
% Se os MEX da plataforma atual existirem, compara tambem os binarios.

    N = 30;
    Ts = 0.033;
    d_safe = 0.27;
    gamma_cbf = 0.10;
    rho_cbf = 2.0e8;
    robot_radius = 0.05;

    data = load('campos_quatro_agentes.mat', 'fields');
    field = data.fields{1};
    [corredores, rotatorias] = local_map_geometry();

    x0 = [-1.30; -0.50; 0.0];
    target = [1.30; -0.50];
    u_previous = [0.04; 0.0];

    controls = [0.075 + 0.005*sin(1:N); 0.04*cos(1:N)];
    U = controls(:);
    X = rollout_unicycle_local(x0, U, Ts, N);

    % O primeiro vizinho se aproxima para ativar a DCBF. Os demais ficam
    % afastados. Assim, um deslocamento no vetor empacotado e detectado.
    neighbor_positions = zeros(2, N + 1, 3);
    approach_distance = linspace(0.34, 0.18, N + 1);
    neighbor_positions(:, :, 1) = X(1:2, :) + ...
        [approach_distance; 0.02*ones(1, N + 1)];
    neighbor_positions(:, :, 2) = repmat([1.20; 0.90], 1, N + 1);
    neighbor_positions(:, :, 3) = repmat([1.30; -0.90], 1, N + 1);

    params_matlab = make_local_params( ...
        x0, neighbor_positions, u_previous, target, N, Ts, d_safe, ...
        gamma_cbf, rho_cbf, robot_radius, corredores, rotatorias);
    params_1D = pack_params_1D( ...
        x0, neighbor_positions, u_previous, target, N, Ts, d_safe, ...
        gamma_cbf, rho_cbf, robot_radius, corredores, rotatorias, 1);

    decoded_neighbors = unpack_neighbors_local(params_1D, N, 3);
    report.neighbor_pack_max_error = max(abs( ...
        neighbor_positions(:) - decoded_neighbors(:)));

    [cost_matlab, grad_matlab] = build_local_dcbf(U, params_matlab, field);
    [cost_codegen_source, grad_codegen_source] = ...
        build_local_dcbf_mexable(U, params_1D);

    report.cost_matlab = cost_matlab;
    report.cost_codegen_source = cost_codegen_source;
    report.cost_relative_error = abs(cost_matlab - cost_codegen_source) / ...
        max([1, abs(cost_matlab), abs(cost_codegen_source)]);
    report.gradient_relative_error = norm( ...
        grad_matlab(:) - grad_codegen_source(:), inf) / ...
        max([1; abs(grad_matlab(:)); abs(grad_codegen_source(:))]);

    report.mex_tested = exist('wrapper_cost_mex', 'file') == 3 && ...
                        exist('wrapper_grad_mex', 'file') == 3;
    report.mex_cost_relative_error = NaN;
    report.mex_gradient_relative_error = NaN;
    if report.mex_tested
        cost_mex = wrapper_cost_mex(U, params_1D);
        grad_mex = wrapper_grad_mex(U, params_1D);
        report.mex_cost_relative_error = abs(cost_matlab - cost_mex) / ...
            max([1, abs(cost_matlab), abs(cost_mex)]);
        report.mex_gradient_relative_error = norm( ...
            grad_matlab(:) - grad_mex(:), inf) / ...
            max([1; abs(grad_matlab(:)); abs(grad_mex(:))]);
    end

    report.source_pass = report.neighbor_pack_max_error < 1e-14 && ...
                         report.cost_relative_error < 1e-12 && ...
                         report.gradient_relative_error < 1e-10;
    report.mex_pass = ~report.mex_tested || ...
                      (report.mex_cost_relative_error < 1e-12 && ...
                       report.mex_gradient_relative_error < 1e-10);
    report.passed = report.source_pass && report.mex_pass;

    fprintf('\n=== EQUIVALENCIA MATLAB x CODEGEN ===\n');
    fprintf('Erro do empacotamento dos vizinhos : %.4e\n', ...
        report.neighbor_pack_max_error);
    fprintf('Erro relativo do custo (fonte)     : %.4e\n', ...
        report.cost_relative_error);
    fprintf('Erro relativo do gradiente (fonte) : %.4e\n', ...
        report.gradient_relative_error);
    if report.mex_tested
        fprintf('Erro relativo do custo (MEX)       : %.4e\n', ...
            report.mex_cost_relative_error);
        fprintf('Erro relativo do gradiente (MEX)   : %.4e\n', ...
            report.mex_gradient_relative_error);
    else
        fprintf('MEX da plataforma %s nao encontrado; binario nao testado.\n', ...
            mexext);
    end
    fprintf('Resultado da fonte                 : %s\n', ...
        pass_text_local(report.source_pass));
    fprintf('Resultado final                    : %s\n\n', ...
        pass_text_local(report.passed));

    assert(report.source_pass, ...
        'A fonte preparada para Codegen nao equivale a formulacao MATLAB.');
    if report.mex_tested
        assert(report.mex_pass, ...
            'O binario MEX nao equivale a formulacao MATLAB atual. Recompile.');
    end
end

function decoded = unpack_neighbors_local(params, N, n_neighbors)
    decoded = zeros(2, N + 1, n_neighbors);
    flat = params(132:317);
    for j = 1:n_neighbors
        offset = (j - 1)*2*(N + 1);
        for k = 1:N + 1
            decoded(:, k, j) = flat(offset + 2*(k - 1) + (1:2));
        end
    end
end

function X = rollout_unicycle_local(x0, U, Ts, N)
    X = zeros(3, N + 1);
    X(:, 1) = x0;
    controls = reshape(U, 2, N);
    for k = 1:N
        theta = X(3, k);
        X(:, k + 1) = [X(1, k) + Ts*controls(1, k)*cos(theta); ...
                       X(2, k) + Ts*controls(1, k)*sin(theta); ...
                       X(3, k) + Ts*controls(2, k)];
    end
end

function [corredores, rotatorias] = local_map_geometry()
    overlap = 0.1;
    gap = 0.25;
    limites = [ ...
        -1.5, -0.8-gap,  0.3,         0.7; ...
        -0.8+gap, 0.0+overlap, 0.3,   0.7; ...
        0.0-overlap, 0.8-gap, 0.3,    0.7; ...
        0.8+gap, 1.5, 0.3,            0.7; ...
        -1.5, -0.8+overlap, -0.7,    -0.3; ...
        -0.8-overlap, 0.0-gap, -0.7, -0.3; ...
        0.0+gap, 0.8+overlap, -0.7,  -0.3; ...
        0.8-overlap, 1.5, -0.7,      -0.3; ...
        -1.0, -0.6, -1.0, -0.5+overlap; ...
        -1.0, -0.6, -0.5-overlap, 0.5-gap; ...
        -1.0, -0.6, 0.5+gap, 1.0; ...
        -0.2, 0.2, -1.0, -0.5-gap; ...
        -0.2, 0.2, -0.5+gap, 0.5+overlap; ...
        -0.2, 0.2, 0.5-overlap, 1.0; ...
        0.6, 1.0, -1.0, -0.5+overlap; ...
        0.6, 1.0, -0.5-overlap, 0.5-gap; ...
        0.6, 1.0, 0.5+gap, 1.0];

    corredores = cell(1, size(limites, 1));
    for i = 1:size(limites, 1)
        b = limites(i, :);
        corredores{i}.v = [b(1), b(3); b(2), b(3); ...
                            b(2), b(4); b(1), b(4)];
    end
    rotatorias = [-0.8, 0.5, 0.35, 0.08, 1; ...
                   0.8, 0.5, 0.35, 0.08, 1; ...
                   0.0,-0.5, 0.35, 0.08, 1];
end

function text = pass_text_local(value)
    if value
        text = 'PASSOU';
    else
        text = 'FALHOU';
    end
end
