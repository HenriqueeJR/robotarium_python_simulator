function results = main_quatro_agentes_dcbf(show_figure, simulation_time, ...
                                             solver_max_iterations, ...
                                             d_safe, rho_cbf, ...
                                             skip_initialization)
%MAIN_QUATRO_AGENTES_DCBF Linha de base local com quatro agentes.
%
% Este cenario foi criado para avaliar o que acontece quando a densidade
% de trafego aumenta antes da introducao da proposta coalizional. Cada
% agente possui um MPC/PANOC local, um unico ponto de referencia, seu campo
% vetorial e sua geofence. A DCBF considera simultaneamente os outros tres
% agentes por meio das previsoes comunicadas e congeladas.
%
% Nao existem neste arquivo:
%   - prioridade de passagem;
%   - custo de fluxo;
%   - formacao de coalizoes;
%   - ADMM ou otimizacao conjunta;
%   - waypoints ou seguimento de caminho.
%
% Exemplos:
%   results = main_quatro_agentes_dcbf();
%   results = main_quatro_agentes_dcbf(false, 15, 5); % teste curto

    if nargin < 1
        show_figure = true;
    end
    if nargin < 2
        simulation_time = 60;
    end
    if nargin < 3
        % Cinco iteracoes foram validadas no caso de dois agentes e
        % constituem um ponto de partida conservador para a escalabilidade.
        solver_max_iterations = 5;
    end
    if nargin < 4
        % A margem maior compensa a natureza suave da penalidade e a
        % inconsistencia temporaria entre previsoes locais simultaneas.
        d_safe = 0.27;
    end
    if nargin < 5
        rho_cbf = 2.0e8;
    end
    if nargin < 6
        % Na simulacao local, evita a longa rotina que leva os robos de
        % poses aleatorias ate os inicios do experimento. O Robotarium
        % fisico e seu verificador sempre executam a inicializacao real.
        skip_initialization = true;
    end

    results = main_multiagente_dcbf( ...
        show_figure, simulation_time, true, solver_max_iterations, ...
        'quatro', d_safe, rho_cbf, skip_initialization);
end
