%% Simulacao local com quatro agentes, MPC/PANOC e DCBF
% Linha de base sem prioridade, coalizoes ou custo global de fluxo.

close all;
clc;

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

results4 = main_quatro_agentes_dcbf(true, 60, 5);

save(fullfile(this_dir, 'resultado_quatro_agentes_simulacao.mat'), ...
     'results4');

fprintf('\nResultado salvo em resultado_quatro_agentes_simulacao.mat\n');
fprintf('Agentes que concluiram: %d de %d\n', ...
        results4.number_completed, results4.number_of_agents);
fprintf('Distancia minima: %.4f m\n', results4.minimum_distance);
fprintf('Amostras abaixo da distancia requerida: %d\n', ...
        results4.total_unsafe_distance_events);

