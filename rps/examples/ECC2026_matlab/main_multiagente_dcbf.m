function results = main_multiagente_dcbf(show_figure, simulation_time, ...
                                          enable_dcbf, solver_max_iterations, ...
                                          agent_scenario, d_safe_override, ...
                                          rho_cbf_override, ...
                                          skip_initialization)
%MAIN_MULTIAGENTE_DCBF Robos com campos individuais e DCBF interagente.
%
% Este e o primeiro experimento de integracao multiagente. Cada robo possui
% um NMPC/PANOC local e considera a previsao comunicada do outro robo como
% parametro fixo durante a otimizacao. Nao ha formacao de coalizoes nem ADMM
% nesta etapa.
%
% Exemplos:
%   results = main_multiagente_dcbf();          % simulacao grafica completa
%   results = main_multiagente_dcbf(false, 8);  % teste curto sem figura
%   baseline = main_multiagente_dcbf(false, 10, false); % sem DCBF
%   fisico = main_multiagente_dcbf(false, 45, true, 30); % limite PANOC
%   quatro = main_multiagente_dcbf(true, 60, true, 5, 'quatro');

    if nargin < 1
        show_figure = true;
    end
    if nargin < 2
        simulation_time = 45;
    end
    if nargin < 3
        enable_dcbf = true;
    end
    if nargin < 4
        solver_max_iterations = 200;
    end
    if nargin < 5
        agent_scenario = 'dois';
    end
    if nargin < 6
        d_safe_override = [];
    end
    if nargin < 7
        rho_cbf_override = [];
    end
    if nargin < 8
        skip_initialization = false;
    end

    close all;
    clc;

    this_dir = fileparts(mfilename('fullpath'));
    code_dir = fileparts(this_dir);
    robotarium_dir = fullfile(this_dir, 'Robotarium_simulator');
    if ~isfolder(robotarium_dir)
        robotarium_dir = fullfile(code_dir, 'Robotarium_simulator');
    end
    monoagent_dir = fullfile(code_dir, 'MPC Monoagente Robotarium', ...
                             'grafo_lbm_mpc_grande');

    addpath(this_dir);
    % No computador local, essas pastas fornecem o simulador e o PANOC. No
    % servidor fisico do Robotarium, a API ja esta instalada e os arquivos
    % auxiliares enviados ficam no mesmo diretorio do arquivo principal.
    if isfolder(robotarium_dir)
        addpath(genpath(robotarium_dir));
    end
    % Se o simulador nao estiver no pacote nem na pasta imediatamente
    % superior, procura-o no workspace apenas quando necessario.
    if exist('Robotarium', 'class') ~= 8 && exist('Robotarium', 'file') ~= 2
        robotarium_matches = dir(fullfile(code_dir, '**', 'Robotarium.m'));
        if ~isempty(robotarium_matches)
            robotarium_dir = robotarium_matches(1).folder;
            addpath(genpath(robotarium_dir));
            fprintf('Robotarium_simulator localizado em %s.\n', ...
                robotarium_dir);
        end
    end
    assert(exist('Robotarium', 'class') == 8 || ...
           exist('Robotarium', 'file') == 2, ...
        ['Nao foi possivel localizar Robotarium.m. Adicione a pasta ' ...
         'Robotarium_simulator ao path do MATLAB.']);
    if isfolder(monoagent_dir)
        addpath(monoagent_dir);
    end

    fprintf('=== EXPERIMENTO MULTIAGENTE COM DCBF ===\n');
    fprintf('Sem coalizoes; um PANOC local por agente.\n');
    fprintf('DCBF interagente: %s.\n\n', string_on_off(enable_dcbf));

    % ---------------------------------------------------------------------
    % Mapa grande utilizado pelo exemplo monoagente original. O ensaio usa
    % o cruzamento fisico entre a via horizontal inferior e a via vertical
    % esquerda. As rotas nao compartilham um no do grafo, mas suas regioes
    % geometricas se interceptam, que e exatamente o caso que a DCBF deve
    % tratar.
    % ---------------------------------------------------------------------
    overlap = 0.1;
    roundabout_gap = 0.25;
    corredores = cell(1, 17);

    corredores{1}.v = rectangle_vertices(-1.5, -0.8-roundabout_gap, 0.3, 0.7);
    corredores{1}.dir = [1, 0];
    corredores{2}.v = rectangle_vertices(-0.8+roundabout_gap, 0.0+overlap, 0.3, 0.7);
    corredores{2}.dir = [1, 0];
    corredores{3}.v = rectangle_vertices(0.0-overlap, 0.8-roundabout_gap, 0.3, 0.7);
    corredores{3}.dir = [1, 0];
    corredores{4}.v = rectangle_vertices(0.8+roundabout_gap, 1.5, 0.3, 0.7);
    corredores{4}.dir = [1, 0];

    corredores{5}.v = rectangle_vertices(-1.5, -0.8+overlap, -0.7, -0.3);
    corredores{5}.dir = [1, 0];
    corredores{6}.v = rectangle_vertices(-0.8-overlap, 0.0-roundabout_gap, -0.7, -0.3);
    corredores{6}.dir = [1, 0];
    corredores{7}.v = rectangle_vertices(0.0+roundabout_gap, 0.8+overlap, -0.7, -0.3);
    corredores{7}.dir = [1, 0];
    corredores{8}.v = rectangle_vertices(0.8-overlap, 1.5, -0.7, -0.3);
    corredores{8}.dir = [1, 0];

    corredores{9}.v = rectangle_vertices(-1.0, -0.6, -1.0, -0.5+overlap);
    corredores{9}.dir = [0, 1];
    corredores{10}.v = rectangle_vertices(-1.0, -0.6, -0.5-overlap, 0.5-roundabout_gap);
    corredores{10}.dir = [0, 1];
    corredores{11}.v = rectangle_vertices(-1.0, -0.6, 0.5+roundabout_gap, 1.0);
    corredores{11}.dir = [0, 1];

    corredores{12}.v = rectangle_vertices(-0.2, 0.2, -1.0, -0.5-roundabout_gap);
    corredores{12}.dir = [0, 1];
    corredores{13}.v = rectangle_vertices(-0.2, 0.2, -0.5+roundabout_gap, 0.5+overlap);
    corredores{13}.dir = [0, 1];
    corredores{14}.v = rectangle_vertices(-0.2, 0.2, 0.5-overlap, 1.0);
    corredores{14}.dir = [0, 1];

    corredores{15}.v = rectangle_vertices(0.6, 1.0, -1.0, -0.5+overlap);
    corredores{15}.dir = [0, 1];
    corredores{16}.v = rectangle_vertices(0.6, 1.0, -0.5-overlap, 0.5-roundabout_gap);
    corredores{16}.dir = [0, 1];
    corredores{17}.v = rectangle_vertices(0.6, 1.0, 0.5+roundabout_gap, 1.0);
    corredores{17}.dir = [0, 1];

    rotatorias = [ ...
        -0.8,  0.5, 0.35, 0.08, 1;
         0.8,  0.5, 0.35, 0.08, 1;
         0.0, -0.5, 0.35, 0.08, 1];

    switch lower(char(agent_scenario))
        case 'dois'
            Nr = 2;
            starts = [-1.30, -0.80; ...
                      -0.50, -0.90];
            targets = [1.30, -0.80; ...
                      -0.50,  0.90];
            initial_headings = [0, pi/2];
            cache_name = 'campos_cruzamento_dois_agentes.mat';
            cache_version = 4;
            scenario_d_safe = 0.215;
            scenario_rho_cbf = 2.0e7;

        case 'quatro'
            % Linha de base para o estudo de escalabilidade. Os quatro
            % agentes usam exatamente o mesmo controle local, sem
            % prioridade, custo de fluxo ou formacao de coalizoes.
            Nr = 4;
            starts = [-1.30, -0.80,  0.00, -1.30; ...
                      -0.50, -0.90, -0.90,  0.50];
            targets = [1.30, -0.80, 0.00, 1.30; ...
                      -0.50,  0.90, 0.90, 0.50];
            initial_headings = [0, pi/2, pi/2, 0];
            cache_name = 'campos_quatro_agentes.mat';
            cache_version = 1;
            % Valores validados para os conflitos simultaneos deste
            % cenario. A chamada direta da funcao geral tambem deve ser
            % segura, sem depender do wrapper de quatro agentes.
            scenario_d_safe = 0.270;
            scenario_rho_cbf = 2.0e8;

        otherwise
            error('main_multiagente_dcbf:UnknownScenario', ...
                  'Cenario desconhecido: %s.', char(agent_scenario));
    end
    initial_conditions = [starts; initial_headings];

    % Agente 1: C5 -> C6 -> R3 -> C7 -> C8 (movimento horizontal).
    % Agente 2: C9 -> C10 -> R1 -> C11 (movimento vertical).
    % O primeiro encontro ocorre no cruzamento em (-0.8,-0.5).
    bounds = [-1.6, 1.6, -1.0, 1.0];
    lbm_resolution = 0.02;
    robot_radius = 0.05;


    % =====================================================================
    % 1. GERAÇÃO E COMPACTAÇÃO DOS CAMPOS VETORIAIS (LBM 3D)
    % =====================================================================
    cache_file = fullfile(this_dir, cache_name);
    [fields, route_names] = load_or_generate_fields( ...
        cache_file, cache_version, corredores, rotatorias, starts, ...
        targets, bounds, lbm_resolution, robot_radius);
    
    % Compactar num único ficheiro 3D para o MEX conseguir carregar
    disp('>> Empacotando LBM 3D para compilação C++...');
    [n_rows, n_cols] = size(fields{1}.ux);
    ux_all = zeros(n_rows, n_cols, Nr);
    uy_all = zeros(n_rows, n_cols, Nr);
    for i = 1:Nr
        ux_all(:,:,i) = fields{i}.ux;
        uy_all(:,:,i) = fields{i}.uy;
    end
    x_vec = fields{1}.x_vec; y_vec = fields{1}.y_vec;
    for i = 1:Nr
        assert(isequal(size(fields{i}.ux), [numel(y_vec), numel(x_vec)]), ...
            'Campo %d tem dimensões incompatíveis com x_vec/y_vec.', i);
        assert(isequal(fields{i}.x_vec(:), x_vec(:)) && ...
               isequal(fields{i}.y_vec(:), y_vec(:)), ...
            'Todos os campos devem usar a mesma malha para o MEX 3D.', i);
    end
    % Salvar ao lado deste arquivo: coder.load usa esse arquivo durante a compilação.
    lbm_file = fullfile(this_dir, 'gvf_lbm_data_multi.mat');
    save(lbm_file, 'x_vec', 'y_vec', 'ux_all', 'uy_all');

    % =====================================================================
    % 2. SELECAO DA IMPLEMENTACAO DO CUSTO E DO GRADIENTE
    % =====================================================================
    N_mpc = 30;
    [cost_function, gradient_function, objective_mode] = ...
        select_objective_implementation(this_dir);
    using_native_mex = startsWith(objective_mode, 'MEX nativo');
    fprintf('>> Implementacao do objetivo: %s.\n', objective_mode);

    % =====================================================================
    % 3. SETUP ROBOTARIUM
    % =====================================================================
    r = Robotarium('NumberOfRobots', Nr, 'ShowFigure', logical(show_figure), 'InitialConditions', initial_conditions, 'SkipInitialization', logical(skip_initialization));
    Ts = r.time_step; n_steps = ceil(simulation_time/Ts);
    v_min = 0.0; v_max = 0.10; omega_min = -1.5; omega_max = 1.5;
    U_min = repmat([v_min; omega_min], N_mpc, 1); U_max = repmat([v_max; omega_max], N_mpc, 1);
    
    required_distance = 0.20;
    if isempty(d_safe_override), d_safe = scenario_d_safe; else, d_safe = d_safe_override; end
    gamma_cbf = 0.10;
    if isempty(rho_cbf_override), rho_cbf = scenario_rho_cbf * double(enable_dcbf); else, rho_cbf = rho_cbf_override * double(enable_dcbf); end
    
    reference_speed = 0.09; goal_tolerance = 0.06;
    
    solvers = cell(1, Nr); warm_starts = cell(1, Nr); previous_commands = zeros(2, Nr);
    
    % Respeita o limite recebido no argumento da função.
    
    for i = 1:Nr
        solvers{i} = PanocSolver(0.01, 1e-4, solver_max_iterations, 1e-3, U_min, U_max, 20);
        solvers{i}.cost_func = cost_function;
        solvers{i}.grad_func = gradient_function;
        warm_starts{i} = make_field_nominal_plan(initial_conditions(:, i), fields{i}, targets(:, i), N_mpc, Ts, reference_speed, omega_min, omega_max);
    end
    
    if show_figure, draw_map_on_robotarium(r.figure_handle, corredores, rotatorias, starts, targets, Nr); end
    
    % Matrizes de logging e variáveis de estado...
    X_history = zeros(3, Nr, n_steps); U_history = zeros(2, Nr, n_steps);
    pair_indices = nchoosek(1:Nr, 2); n_pairs = size(pair_indices, 1);
    distance_history = nan(n_pairs, n_steps); violation_history = zeros(Nr, n_steps); geofence_violation_history = zeros(Nr, n_steps);
    solver_time_history = zeros(Nr, n_steps); solver_iterations = zeros(Nr, n_steps);
    done = false(1, Nr); completion_times = nan(1, Nr); unsafe_distance_events = zeros(n_pairs, 1); unsafe_warning_issued = false;
    executed_steps = 0;

    % =====================================================================
    % 4. LOOP DE SIMULAÇÃO TEMPO REAL
    % =====================================================================
    for k = 1:n_steps
        X = r.get_poses(); X_history(:, :, k) = X;
        for i = 1:Nr
            if ~done(i) && norm(X(1:2, i) - targets(:, i)) < goal_tolerance
                done(i) = true; completion_times(i) = (k-1)*Ts; warm_starts{i} = zeros(2*N_mpc, 1);
            end
        end
        
        communicated_predictions = zeros(2, N_mpc + 1, Nr);
        for j = 1:Nr
            X_plan_j = rollout_unicycle(X(:, j), warm_starts{j}, Ts, N_mpc);
            communicated_predictions(:, :, j) = X_plan_j(1:2, :);
        end
        
        U_commands = zeros(2, Nr); new_plans = warm_starts;
        for i = 1:Nr
            if done(i), continue; end
            
            neighbor_ids = setdiff(1:Nr, i);
            neighbor_positions = communicated_predictions(:, :, neighbor_ids);
            
            % Empacotador do Vetor 1D para o C++
            params_1D = pack_params_1D(X(:, i), neighbor_positions, previous_commands(:, i), targets(:, i), ...
                                       N_mpc, Ts, d_safe, gamma_cbf, rho_cbf, robot_radius, corredores, rotatorias, i);
            % Verifica o empacotamento geometrico uma unica vez. Imprimir
            % dentro de todos os ciclos prejudica justamente a medicao de
            % desempenho que motivou o Codegen.
            if k == 1 && i == 1
                rot_original = rotatorias(:,1:4);
                rot_packed = reshape(params_1D(112:131),4,[]).';
                rot_reconstruidas = rot_packed(1:size(rot_original,1),:);
                erro_rot = max(abs(rot_original(:)-rot_reconstruidas(:)));
                fprintf('Erro maximo no empacotamento das rotatorias: %.12g\n', ...
                    erro_rot);
            end

          
            
            nominal_plan = make_field_nominal_plan(X(:, i), fields{i}, targets(:, i), N_mpc, Ts, reference_speed, omega_min, omega_max);
            initial_guess = warm_starts{i};
            initial_cost = cost_function(initial_guess, params_1D);
            nominal_cost = cost_function(nominal_plan, params_1D);
            if isfinite(nominal_cost) && nominal_cost < initial_cost
                initial_guess = nominal_plan;
                initial_cost = nominal_cost;
            end

            % ==========================================================
            % DIAGNOSTICO MATLAB x MEX: primeiro passo, primeiro agente
            % ==========================================================
            if k == 1 && i == 1 && using_native_mex
            
                fprintf('\nIniciando diagnostico da implementacao MATLAB x MEX...\n');
            
                % Estrutura de parametros da implementacao MATLAB original
                params_matlab = make_local_params( ...
                    X(:, i), neighbor_positions, previous_commands(:, i), ...
                    targets(:, i), N_mpc, Ts, d_safe, gamma_cbf, ...
                    rho_cbf, robot_radius, corredores, rotatorias);
            
                % Usa exatamente o mesmo vetor de controles nas duas versoes
                U_test = initial_guess;
            
                % Executa os testes
                report = diagnostico_comparacao_mex( ...
                    U_test, params_matlab, fields{i}, params_1D);
            
                % Salva os resultados
                save('diagnostico_primeiro_passo.mat', 'report');
            
                fprintf('Diagnostico concluido. Consulte o relatorio acima.\n');
            end
            % ==========================================================


            solve_timer = tic;
            [U_opt, ~, iterations] = solvers{i}.solve(initial_guess, params_1D);
            solver_time_history(i, k) = toc(solve_timer);
            solver_iterations(i, k) = iterations;

            % Evita aceitar uma saída inválida ou pior que o palpite viável.
            optimized_cost = cost_function(U_opt, params_1D);
            if ~isfinite(optimized_cost) || optimized_cost > initial_cost
                U_opt = initial_guess;
            end

            U_commands(:, i) = U_opt(1:2);
            new_plans{i} = shift_control_sequence(U_opt, N_mpc);
        end
        
        U_commands(1, :) = min(max(U_commands(1, :), v_min), v_max);
        U_commands(2, :) = min(max(U_commands(2, :), omega_min), omega_max);
        r.set_velocities(1:Nr, U_commands);
        r.step();
        
        U_history(:, :, k) = U_commands; previous_commands = U_commands; warm_starts = new_plans;
        for pair_index = 1:n_pairs
            i = pair_indices(pair_index, 1); j = pair_indices(pair_index, 2);
            distance_history(pair_index, k) = norm(X(1:2, i) - X(1:2, j));
            if distance_history(pair_index, k) < required_distance
                unsafe_distance_events(pair_index) = unsafe_distance_events(pair_index) + 1;
                if ~unsafe_warning_issued
                    fprintf('ATENCAO: distancia fisica violada em t = %.3f s pelo par A%d-A%d (%.4f m).\n', (k-1)*Ts, i, j, distance_history(pair_index, k));
                    unsafe_warning_issued = true;
                end
            end
        end
        executed_steps = k;
        if all(done), fprintf('Todos os %d agentes alcancaram seus destinos.\n', Nr); break; end
    end

    % Captura o ultimo estado sem quebrar a regra get_poses/step.
    X_history = X_history(:, :, 1:executed_steps);
    U_history = U_history(:, :, 1:executed_steps);
    distance_history = distance_history(:, 1:executed_steps);
    violation_history = violation_history(:, 1:executed_steps);
    geofence_violation_history = ...
        geofence_violation_history(:, 1:executed_steps);
    solver_time_history = solver_time_history(:, 1:executed_steps);
    solver_iterations = solver_iterations(:, 1:executed_steps);

    minimum_distance_per_pair = min(distance_history, [], 2, 'omitnan');
    minimum_distance = min(minimum_distance_per_pair, [], 'omitnan');
    maximum_violation = max(violation_history, [], 'all');
    measured_h = distance_history.^2 - d_safe^2;
    if size(measured_h, 2) > 1
        measured_g = (1-gamma_cbf)*measured_h(:, 1:end-1) - ...
                     measured_h(:, 2:end);
        maximum_measured_violation_per_pair = ...
            max(max(0, measured_g), [], 2);
        maximum_measured_violation = ...
            max(maximum_measured_violation_per_pair);
    else
        maximum_measured_violation_per_pair = nan(n_pairs, 1);
        maximum_measured_violation = NaN;
    end

    fprintf('\n=== RESULTADOS ===\n');
    fprintf('Distancia minima medida       : %.4f m\n', minimum_distance);
    fprintf('Distancia fisica requerida    : %.4f m\n', required_distance);
    fprintf('Distancia usada na DCBF       : %.4f m\n', d_safe);
    fprintf('Violacao DCBF maxima prevista : %.4e\n', maximum_violation);
    fprintf('Violacao DCBF maxima medida   : %.4e\n', ...
            maximum_measured_violation);
    fprintf('Saida maxima da geofence      : %.4e m\n', ...
            max(geofence_violation_history, [], 'all'));
    solved_times = solver_time_history(solver_time_history > 0);
    if isempty(solved_times)
        mean_panoc_time = NaN;
    else
        mean_panoc_time = mean(solved_times);
    end
    total_solver_time_per_cycle = sum(solver_time_history, 1);
    fprintf('Tempo medio por PANOC         : %.3f ms\n', ...
            1e3*mean_panoc_time);
    fprintf('Tempo medio total por ciclo   : %.3f ms\n', ...
            1e3*mean(total_solver_time_per_cycle));
    fprintf('Maximo de iteracoes PANOC     : %.0f\n', ...
            max(solver_iterations, [], 'all'));
    final_positions = X_history(1:2, :, end);
    final_errors = vecnorm(final_positions - targets);
    standstill_window_steps = min(executed_steps, ceil(3.0/Ts));
    standstill_window = executed_steps-standstill_window_steps+1:executed_steps;
    mean_speed_last_window = mean( ...
        abs(squeeze(U_history(1, :, standstill_window))), 2).';
    stalled_unfinished = ~done & ...
        (mean_speed_last_window < 0.005) & (final_errors > goal_tolerance);
    possible_safe_deadlock = any(stalled_unfinished);
    for i = 1:Nr
        fprintf('Erro final do agente %-2d       : %.4f m\n', ...
                i, final_errors(i));
    end
    fprintf('Missao concluida              : %s\n', mat2str(done));
    fprintf('Agentes no destino            : %d/%d\n', sum(done), Nr);
    fprintf('Amostras abaixo de %.2f m      : %d\n', ...
            required_distance, sum(unsafe_distance_events));
    fprintf('Possivel bloqueio seguro      : %s\n', ...
            string_yes_no(possible_safe_deadlock));

    for pair_index = 1:n_pairs
        fprintf('Distancia minima A%d-A%d       : %.4f m\n', ...
                pair_indices(pair_index, 1), pair_indices(pair_index, 2), ...
                minimum_distance_per_pair(pair_index));
    end

    r.debug();

    if show_figure && executed_steps > 0
        plot_results(X_history, U_history, distance_history, ...
                     violation_history, solver_time_history, Ts, d_safe, ...
                     starts, targets, pair_indices);
    end

    results.route_names = route_names;
    results.starts = starts;
    results.targets = targets;
    results.Ts = Ts;
    results.executed_steps = executed_steps;
    results.simulated_time = executed_steps*Ts;
    results.X = X_history;
    results.U = U_history;
    results.distance = distance_history;
    results.pair_indices = pair_indices;
    results.minimum_distance_per_pair = minimum_distance_per_pair;
    results.required_distance = required_distance;
    results.d_safe = d_safe;
    results.rho_cbf = rho_cbf;
    results.goal_tolerance = goal_tolerance;
    results.robot_radius = robot_radius;
    results.max_dcbf_violation = maximum_violation;
    results.max_measured_dcbf_violation = maximum_measured_violation;
    results.max_measured_dcbf_violation_per_pair = ...
        maximum_measured_violation_per_pair;
    results.max_geofence_violation = ...
        max(geofence_violation_history, [], 'all');
    results.minimum_distance = minimum_distance;
    results.unsafe_distance_events_per_pair = unsafe_distance_events;
    results.total_unsafe_distance_events = sum(unsafe_distance_events);
    results.solver_times = solver_time_history;
    results.total_solver_time_per_cycle = total_solver_time_per_cycle;
    results.mean_panoc_time = mean_panoc_time;
    results.solver_iterations = solver_iterations;
    results.final_errors = final_errors;
    results.completed = done;
    results.completion_times = completion_times;
    results.mean_speed_last_3s = mean_speed_last_window;
    results.stalled_unfinished = stalled_unfinished;
    results.possible_safe_deadlock = possible_safe_deadlock;
    results.number_of_agents = Nr;
    results.number_completed = sum(done);
    results.scenario = char(agent_scenario);
    results.dcbf_enabled = logical(enable_dcbf);
    results.solver_max_iterations = solver_max_iterations;
    results.skipped_robotarium_initialization = ...
        logical(skip_initialization);
end

function text = string_on_off(flag)
    if flag
        text = 'ATIVADA';
    else
        text = 'DESATIVADA';
    end
end

function text = string_yes_no(flag)
    if flag
        text = 'SIM';
    else
        text = 'NAO';
    end
end

function X = rollout_unicycle(x0, U, Ts, N)
    controls = reshape(U, 2, N);
    X = zeros(3, N + 1);
    X(:, 1) = x0;
    for n = 1:N
        theta = X(3, n);
        X(:, n + 1) = [ ...
            X(1, n) + Ts*controls(1, n)*cos(theta);
            X(2, n) + Ts*controls(1, n)*sin(theta);
            X(3, n) + Ts*controls(2, n)];
    end
end

function shifted = shift_control_sequence(U, N)
    controls = reshape(U, 2, N);
    controls = [controls(:, 2:end), controls(:, end)];
    shifted = controls(:);
end

function plan = make_field_nominal_plan(x0, field, target, N, Ts, ...
                                        speed_limit, omega_min, omega_max)
%MAKE_FIELD_NOMINAL_PLAN Palpite gerado somente pelo campo vetorial.

    controls = zeros(2, N);
    predicted_state = x0;
    heading_gain = 2.0;

    for n = 1:N
        field_x = interp2(field.X_grid, field.Y_grid, field.ux, ...
                          predicted_state(1), predicted_state(2), ...
                          'linear', 0);
        field_y = interp2(field.X_grid, field.Y_grid, field.uy, ...
                          predicted_state(1), predicted_state(2), ...
                          'linear', 0);
        target_delta = target - predicted_state(1:2);
        target_distance = norm(target_delta);
        goal_blend = exp(-(target_distance/0.25)^2);
        desired_vector = (1-goal_blend)*[field_x; field_y] + ...
                         goal_blend*0.8*target_delta;
        desired_speed = norm(desired_vector);
        if desired_speed > 1.0e-5
            desired_heading = atan2(desired_vector(2), desired_vector(1));
            speed = min(speed_limit, desired_speed);
        else
            desired_heading = predicted_state(3);
            speed = 0;
        end

        heading_error = wrap_angle(desired_heading - predicted_state(3));
        omega = heading_gain*heading_error;
        omega = min(max(omega, omega_min), omega_max);

        controls(:, n) = [speed; omega];
        predicted_state = [ ...
            predicted_state(1) + Ts*speed*cos(predicted_state(3)); ...
            predicted_state(2) + Ts*speed*sin(predicted_state(3)); ...
            predicted_state(3) + Ts*omega];
    end
    plan = controls(:);
end

function angle = wrap_angle(angle)
    angle = atan2(sin(angle), cos(angle));
end

function [fields, route_names] = load_or_generate_fields( ...
        cache_file, cache_version, corredores, rotatorias, starts, ...
        targets, bounds, resolution, robot_radius)

    cache_is_valid = false;
    if isfile(cache_file)
        cached = load(cache_file);
        cache_is_valid = isfield(cached, 'cache_version_saved') && ...
            cached.cache_version_saved == cache_version && ...
            isequal(cached.starts_saved, starts) && ...
            isequal(cached.targets_saved, targets) && ...
            isequal(cached.bounds_saved, bounds) && ...
            cached.resolution_saved == resolution && ...
            isfield(cached, 'robot_radius_saved') && ...
            cached.robot_radius_saved == robot_radius && ...
            all(cellfun(@(item) isfield(item, 'geofence_h'), ...
                        cached.fields));
    end

    if cache_is_valid
        fields = cached.fields;
        route_names = cached.route_names;
        fprintf('Campos vetoriais carregados do cache.\n');
        return;
    end

    Nr = size(starts, 2);
    fields = cell(1, Nr);
    route_names = cell(1, Nr);
    for i = 1:Nr
        fprintf('Gerando rota e campo do agente %d...\n', i);
        [C_opt, R_opt, Con_opt, names] = planejar_rota( ...
            corredores, rotatorias, starts(:, i).', targets(:, i).');
        [X_grid, Y_grid, ux, uy] = gerar_gvf_lbm( ...
            C_opt, R_opt, Con_opt, targets(:, i).', bounds, resolution);

        fields{i}.X_grid = X_grid;
        fields{i}.Y_grid = Y_grid;
        fields{i}.x_vec = X_grid(1, :);
        fields{i}.y_vec = Y_grid(:, 1).';
        fields{i}.ux = ux;
        fields{i}.uy = uy;
        fields{i}.route_boxes = zeros(4, numel(C_opt));
        for corridor_index = 1:numel(C_opt)
            vertices = C_opt{corridor_index}.v;
            fields{i}.route_boxes(:, corridor_index) = [ ...
                min(vertices(:, 1)); max(vertices(:, 1)); ...
                min(vertices(:, 2)); max(vertices(:, 2))];
        end
        fields{i}.route_roundabouts = R_opt;
        fields{i}.geofence_h = build_route_geofence_grid( ...
            X_grid, Y_grid, fields{i}.route_boxes, R_opt, robot_radius);
        route_names{i} = names;
    end

    cache_version_saved = cache_version;
    starts_saved = starts;
    targets_saved = targets;
    bounds_saved = bounds;
    resolution_saved = resolution;
    robot_radius_saved = robot_radius;
    save(cache_file, 'fields', 'route_names', 'cache_version_saved', ...
         'starts_saved', 'targets_saved', 'bounds_saved', ...
         'resolution_saved', 'robot_radius_saved', '-v7.3');
    fprintf('Campos vetoriais salvos em %s.\n', cache_file);
end

function geofence_h = build_route_geofence_grid( ...
        X_grid, Y_grid, route_boxes, route_roundabouts, robot_radius)
%BUILD_ROUTE_GEOFENCE_GRID Campo h>=0 na regiao navegavel da rota.
%
% A uniao e feita entre os corredores retangulares e os discos externos
% das rotatorias, ja descontando o raio do robo. Depois, os discos internos
% dilatados pelo raio do robo sao retirados. Este campo descreve uma area,
% nao uma linha ou caminho a seguir.

    geofence_h = -inf(size(X_grid));
    for region_index = 1:size(route_boxes, 2)
        bounds = route_boxes(:, region_index);
        h_box = min(cat(3, ...
            X_grid-(bounds(1)+robot_radius), ...
            (bounds(2)-robot_radius)-X_grid, ...
            Y_grid-(bounds(3)+robot_radius), ...
            (bounds(4)-robot_radius)-Y_grid), [], 3);
        geofence_h = max(geofence_h, h_box);
    end

    for region_index = 1:size(route_roundabouts, 1)
        roundabout = route_roundabouts(region_index, :);
        safe_outer_radius = roundabout(3) - robot_radius;
        radius_squared = (X_grid-roundabout(1)).^2 + ...
                         (Y_grid-roundabout(2)).^2;
        h_outer = (safe_outer_radius^2-radius_squared) / ...
                  (2*safe_outer_radius);
        geofence_h = max(geofence_h, h_outer);
    end

    for region_index = 1:size(route_roundabouts, 1)
        roundabout = route_roundabouts(region_index, :);
        safe_inner_radius = roundabout(4) + robot_radius;
        radius_squared = (X_grid-roundabout(1)).^2 + ...
                         (Y_grid-roundabout(2)).^2;
        h_inner = (radius_squared-safe_inner_radius^2) / ...
                  (2*safe_inner_radius);
        geofence_h = min(geofence_h, h_inner);
    end
end

function [cost_function, gradient_function, mode] = ...
        select_objective_implementation(this_dir)
%SELECT_OBJECTIVE_IMPLEMENTATION Usa MEX local ou a mesma fonte MATLAB.
%
% Um MEX so pode ser carregado na plataforma para a qual foi compilado.
% A alternativa interpretada mantem exatamente o mesmo funcional e
% permite desenvolver/testar o controlador antes da compilacao nativa.

    cost_mex_file = fullfile(this_dir, ['wrapper_cost_mex.' mexext]);
    grad_mex_file = fullfile(this_dir, ['wrapper_grad_mex.' mexext]);
    if isfile(cost_mex_file) && isfile(grad_mex_file)
        clear wrapper_cost_mex wrapper_grad_mex;
        rehash;
        cost_function = @wrapper_cost_mex;
        gradient_function = @wrapper_grad_mex;
        mode = ['MEX nativo (' mexext ')'];
        return;
    end

    cost_function = @wrapper_cost;
    gradient_function = @wrapper_grad;
    mode = 'MATLAB interpretado (fallback equivalente, mais lento)';
    warning('main_multiagente_dcbf:MexUnavailable', [ ...
        'Nao foram encontrados MEX para a plataforma %s. A simulacao ' ...
        'usara as mesmas funcoes em MATLAB e podera ser mais lenta. ' ...
        'Execute gerar_mex_local depois de configurar o compilador.'], ...
        mexext);
end

function vertices = rectangle_vertices(x_min, x_max, y_min, y_max)
    vertices = [x_min, y_min; x_max, y_min; ...
                x_max, y_max; x_min, y_max];
end

function draw_map_on_robotarium(figure_handle, corredores, rotatorias, ...
                                 starts, targets, number_of_agents)
    figure(figure_handle);
    ax = gca;
    hold(ax, 'on');
    colors = lines(size(starts, 2));
    angle = linspace(0, 2*pi, 100);

    for i = 1:numel(corredores)
        vertices = corredores{i}.v;
        fill(ax, vertices(:, 1), vertices(:, 2), [0.82, 0.84, 0.87], ...
             'FaceAlpha', 0.28, 'EdgeColor', [0.35, 0.35, 0.35]);

        center = mean(vertices, 1);
        direction = corredores{i}.dir;
        quiver(ax, center(1)-0.06*direction(1), ...
               center(2)-0.06*direction(2), ...
               0.12*direction(1), 0.12*direction(2), 0, ...
               'Color', [0.10, 0.35, 0.70], 'LineWidth', 1.3, ...
               'MaxHeadSize', 1.8, 'AutoScale', 'off');
    end
    for i = 1:size(rotatorias, 1)
        cx = rotatorias(i, 1);
        cy = rotatorias(i, 2);
        outer_radius = rotatorias(i, 3);
        inner_radius = rotatorias(i, 4);
        fill(ax, cx + outer_radius*cos(angle), ...
             cy + outer_radius*sin(angle), [0.82, 0.84, 0.87], ...
             'FaceAlpha', 0.28, 'EdgeColor', [0.35, 0.35, 0.35]);
        fill(ax, cx + inner_radius*cos(angle), ...
             cy + inner_radius*sin(angle), 'w', 'EdgeColor', 'k');

        direction = rotatorias(i, 5);
        arrow_angles = [0, pi/2, pi, 3*pi/2];
        arrow_radius = 0.22;
        for arrow_angle = arrow_angles
            px = cx + arrow_radius*cos(arrow_angle);
            py = cy + arrow_radius*sin(arrow_angle);
            tx = -direction*sin(arrow_angle);
            ty =  direction*cos(arrow_angle);
            quiver(ax, px-0.045*tx, py-0.045*ty, ...
                   0.09*tx, 0.09*ty, 0, ...
                   'Color', [0.10, 0.35, 0.70], 'LineWidth', 1.2, ...
                   'MaxHeadSize', 1.8, 'AutoScale', 'off');
        end
    end
    for i = 1:size(starts, 2)
        plot(ax, starts(1, i), starts(2, i), 'o', ...
             'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
             'MarkerSize', 7);
        plot(ax, targets(1, i), targets(2, i), 'p', ...
             'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
             'MarkerSize', 11);
    end
    title(ax, sprintf(['%d agentes com campos individuais e ' ...
                       'DCBF interagente'], number_of_agents));
end

function plot_results(X_history, U_history, distance_history, ...
                      violation_history, solver_time_history, Ts, d_safe, ...
                      starts, targets, pair_indices)
    steps = size(U_history, 3);
    time = (0:steps-1)*Ts;
    colors = lines(size(X_history, 2));

    figure('Name', 'Resultados DCBF multiagente', 'Color', 'w');
    tiledlayout(2, 2);

    nexttile;
    hold on;
    grid on;
    axis equal;
    for i = 1:size(X_history, 2)
        plot(squeeze(X_history(1, i, 1:steps)), ...
             squeeze(X_history(2, i, 1:steps)), ...
             'Color', colors(i, :), 'LineWidth', 1.6, ...
             'DisplayName', sprintf('Agente %d', i));
        plot(starts(1, i), starts(2, i), 'o', ...
             'Color', colors(i, :), 'HandleVisibility', 'off');
        plot(targets(1, i), targets(2, i), 'p', ...
             'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
             'HandleVisibility', 'off');
    end
    xlabel('x [m]'); ylabel('y [m]'); title('Trajetorias'); legend;

    nexttile;
    plot(time, distance_history(:, 1:steps).', 'LineWidth', 1.2);
    hold on; grid on;
    yline(d_safe, 'r--', 'd_{safe}');
    xlabel('tempo [s]'); ylabel('distancia [m]');
    title('Distancias entre pares de agentes');
    pair_labels = arrayfun(@(row) sprintf('A%d-A%d', ...
        pair_indices(row, 1), pair_indices(row, 2)), ...
        1:size(pair_indices, 1), 'UniformOutput', false);
    legend(pair_labels, 'Location', 'best');

    nexttile;
    semilogy(time, max(violation_history(:, 1:steps), [], 1) + eps, ...
             'b', 'LineWidth', 1.5);
    grid on;
    xlabel('tempo [s]'); ylabel('max(0,g_{ij})');
    title('Violacao prevista da DCBF');

    nexttile;
    plot(time, 1e3*solver_time_history(:, 1:steps).', 'LineWidth', 1.2);
    hold on; grid on; yline(1e3*Ts, 'r--', 'T_s');
    xlabel('tempo [s]'); ylabel('tempo do PANOC [ms]');
    title('Custo computacional');
end
