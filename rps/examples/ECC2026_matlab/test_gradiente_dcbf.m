function report = test_gradiente_dcbf()
%TEST_GRADIENTE_DCBF Compara o gradiente analitico com diferencas finitas.

    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);
    rng(7);

    x_vec = -1:0.04:1;
    y_vec = -1:0.04:1;
    [X, Y] = meshgrid(x_vec, y_vec);

    % Campo com variacao espacial para testar tambem as derivadas da
    % interpolacao, e nao apenas a parte da DCBF.
    field.x_vec = x_vec;
    field.y_vec = y_vec;
    field.ux = 0.075 + 0.012*X - 0.006*Y;
    field.uy = 0.018 + 0.004*X + 0.010*Y;
    field.geofence_h = 0.07 - abs(Y);

    N = 9;
    params.N = N;
    params.Ts = 0.08;
    params.x0 = [-0.18; 0.09; 0.12];
    params.field_gain = 1.0;
    params.W_track = 40;
    params.W_goal = 0.8;
    params.W_terminal_goal = 15;
    params.target = [0.45; 0.02];
    params.robot_radius = 0.05;
    params.rho_geofence = 8.0e3;
    params.R_v = 0.05;
    params.R_w = 0.03;
    params.R_dv = 0.08;
    params.R_dw = 0.05;
    params.u_previous = [0.055; -0.02];
    params.d_safe = 0.22;
    params.gamma_cbf = 0.10;
    params.rho_cbf = 2.0e3;

    % Geofence analitica usada pela formulacao atual. O estado inicial foi
    % escolhido ligeiramente fora deste bloco para ativar a penalidade e
    % testar tambem sua contribuicao no backward pass.
    params.N_blocks = 1;
    params.blocks_params = [-0.15; 0.15; -0.05; 0.05];
    params.rotatorias = zeros(0, 4);
    params.h_p2s = 0.05;
    params.eta_safe = 8.0e3;
    params.gamma_safe = 0.5;
    params.eta_obs = 8.0e3;
    params.gamma_obs = 0.5;

    neighbor = zeros(2, N + 1, 1);
    neighbor(1, :, 1) = linspace(0.105, 0.045, N + 1);
    neighbor(2, :, 1) = linspace(0.015, 0.010, N + 1);
    params.neighbor_positions = neighbor;

    controls = zeros(2, N);
    controls(1, :) = 0.075 + 0.008*randn(1, N);
    controls(2, :) = 0.04*randn(1, N);
    U = controls(:);

    [cost, grad, ~, diagnostics] = build_local_dcbf(U, params, field);

    epsilon = 1e-6;
    grad_fd = zeros(size(U));
    for k = 1:numel(U)
        direction = zeros(size(U));
        direction(k) = 1;
        cost_plus = build_local_dcbf(U + epsilon*direction, params, field);
        cost_minus = build_local_dcbf(U - epsilon*direction, params, field);
        grad_fd(k) = (cost_plus - cost_minus)/(2*epsilon);
    end

    absolute_error = norm(grad - grad_fd, inf);
    relative_error = norm(grad - grad_fd)/max(1, norm(grad_fd));

    fprintf('Custo de teste                : %.10g\n', cost);
    fprintf('Violacao DCBF maxima ativa    : %.6e\n', ...
            diagnostics.max_dcbf_violation);
    fprintf('Violacao geofence maxima ativa: %.6e\n', ...
            diagnostics.max_geofence_violation);
    fprintf('Erro absoluto do gradiente    : %.6e\n', absolute_error);
    fprintf('Erro relativo do gradiente    : %.6e\n', relative_error);

    assert(diagnostics.max_dcbf_violation > 1e-8, ...
           'O teste nao ativou a penalidade DCBF.');
    assert(diagnostics.max_geofence_violation > 1e-8, ...
           'O teste nao ativou a penalidade de geofence.');
    assert(relative_error < 5e-5, ...
           'O gradiente analitico nao passou na verificacao numerica.');

    report.cost = cost;
    report.max_dcbf_violation = diagnostics.max_dcbf_violation;
    report.max_geofence_violation = diagnostics.max_geofence_violation;
    report.absolute_gradient_error = absolute_error;
    report.relative_gradient_error = relative_error;
    report.passed = true;

    disp('TESTE APROVADO: gradiente analitico consistente com diferencas finitas.');
end
