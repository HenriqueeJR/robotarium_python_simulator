function report = diagnostico_comparacao_mex(U, params, field, params_mex, varargin)
%DIAGNOSTICO_COMPARACAO_MEX Compara a implementacao MATLAB com a MEX.
%
% USO MINIMO:
% report = diagnostico_comparacao_mex(U, params, field, params_mex);
%
% Entradas:
%   U          : vetor de controles (2*N x 1), mesma solucao de teste nas duas funcoes
%   params     : struct passado a build_local_dcbf(U, params, field)
%   field      : campo vetorial individual usado pela implementacao MATLAB
%   params_mex : vetor numerico empacotado passado a wrapper_cost_mex/grad
%
% Entradas opcionais em pares nome/valor:
%   'SolverMATLAB' : objeto PANOC configurado para build_local_dcbf
%   'SolverMEX'    : objeto PANOC configurado para wrapper_cost_mex/grad
%   'InitialGuess' : chute inicial para comparar os primeiros comandos otimizados
%   'FiniteDiffStep': passo para diferencas finitas (padrao 1e-6)
%   'GradientSamples': numero de componentes amostrados no teste FD (padrao todos)
%
% Requisitos:
%   build_local_dcbf.m, build_local_dcbf_mexable.m,
%   wrapper_cost_mex.mex* e wrapper_grad_mex.mex* no path.
%
% IMPORTANTE:
%   A comparacao de custo/gradiente usa exatamente o mesmo U. Para a comparacao
%   dos otimizadores, os dois solvers devem estar configurados com os mesmos
%   limites, tolerancias, iteracoes e parametros numericos equivalentes.
%
%   O script nao altera pesos nem parametros. Ele apenas diagnostica diferenças.

ip = inputParser;
ip.addParameter('SolverMATLAB', [], @(x) isempty(x) || isobject(x));
ip.addParameter('SolverMEX', [], @(x) isempty(x) || isobject(x));
ip.addParameter('InitialGuess', [], @(x) isempty(x) || isnumeric(x));
ip.addParameter('FiniteDiffStep', 1e-6, @(x) isnumeric(x) && isscalar(x) && x > 0);
ip.addParameter('GradientSamples', inf, @(x) isnumeric(x) && isscalar(x) && x >= 1);
ip.parse(varargin{:});
opt = ip.Results;

U = U(:);
params_mex = params_mex(:);
nU = numel(U);
assert(mod(nU,2)==0, 'U deve ter dimensao 2*N.');
assert(exist('build_local_dcbf','file')==2, 'Nao encontrei build_local_dcbf.m no path.');
assert(exist('build_local_dcbf_mexable','file')==2, 'Nao encontrei build_local_dcbf_mexable.m no path.');
assert(exist('wrapper_cost_mex','file')==3 || exist('wrapper_cost_mex','file')==2, ...
    'Nao encontrei wrapper_cost_mex. Compile o MEX primeiro.');
assert(exist('wrapper_grad_mex','file')==3 || exist('wrapper_grad_mex','file')==2, ...
    'Nao encontrei wrapper_grad_mex. Compile o MEX primeiro.');

fprintf('\n============================================================\n');
fprintf(' DIAGNOSTICO MATLAB x MEX\n');
fprintf('============================================================\n');
fprintf('Dimensao U: %d (%d etapas)\n', nU, nU/2);
fprintf('Dimensao params_mex: %d\n\n', numel(params_mex));

%% 1) Custo e gradiente analitico no mesmo U
[cost_mat, grad_mat, X_mat] = build_local_dcbf(U, params, field);
[cost_mex, grad_mex] = build_local_dcbf_mexable(U, params_mex);
grad_mat = grad_mat(:);
grad_mex = grad_mex(:);

report.cost.matlab = cost_mat;
report.cost.mex = cost_mex;
report.cost.abs_error = abs(cost_mat-cost_mex);
report.cost.rel_error = abs(cost_mat-cost_mex)/max([1,abs(cost_mat),abs(cost_mex)]);
report.gradient.matlab = grad_mat;
report.gradient.mex = grad_mex;

if numel(grad_mat) ~= numel(grad_mex)
    report.gradient.dimension_match = false;
    report.gradient.abs_inf_error = inf;
    report.gradient.rel_inf_error = inf;
else
    report.gradient.dimension_match = true;
    report.gradient.abs_inf_error = norm(grad_mat-grad_mex, inf);
    report.gradient.rel_inf_error = report.gradient.abs_inf_error / ...
        max([1; abs(grad_mat); abs(grad_mex)]);
end

fprintf('[1] CUSTO no mesmo U\n');
fprintf('    MATLAB : %.12g\n    MEX     : %.12g\n    erro abs: %.4e | relativo: %.4e\n', ...
    cost_mat, cost_mex, report.cost.abs_error, report.cost.rel_error);
fprintf('[2] GRADIENTE analitico no mesmo U\n');
fprintf('    dimensoes iguais: %s\n', yesno(report.gradient.dimension_match));
fprintf('    erro infinito abs: %.4e | relativo: %.4e\n', ...
    report.gradient.abs_inf_error, report.gradient.rel_inf_error);

%% 2) Finite differences para cada implementacao
eps_fd = opt.FiniteDiffStep;
idx = 1:nU;
if isfinite(opt.GradientSamples) && opt.GradientSamples < nU
    % Amostra distribuida por todo o vetor, em vez de verificar apenas o inicio.
    idx = unique(round(linspace(1,nU,opt.GradientSamples)));
end
gfd_mat = nan(nU,1);
gfd_mex = nan(nU,1);
for ii = idx
    d = zeros(nU,1); d(ii) = eps_fd;
    cp = build_local_dcbf(U+d, params, field);
    cm = build_local_dcbf(U-d, params, field);
    gfd_mat(ii) = (cp-cm)/(2*eps_fd);
    cp = wrapper_cost_mex(U+d, params_mex);
    cm = wrapper_cost_mex(U-d, params_mex);
    gfd_mex(ii) = (cp-cm)/(2*eps_fd);
end
mask = false(nU,1); mask(idx) = true;
report.finite_difference.indices = idx;
report.finite_difference.step = eps_fd;
report.finite_difference.matlab_max_abs_error = norm(grad_mat(mask)-gfd_mat(mask),inf);
report.finite_difference.mex_max_abs_error = norm(grad_mex(mask)-gfd_mex(mask),inf);
den_mat = max([1; abs(grad_mat(mask)); abs(gfd_mat(mask))]);
den_mex = max([1; abs(grad_mex(mask)); abs(gfd_mex(mask))]);
report.finite_difference.matlab_rel_error = report.finite_difference.matlab_max_abs_error/den_mat;
report.finite_difference.mex_rel_error = report.finite_difference.mex_max_abs_error/den_mex;
report.finite_difference.matlab_gradient = gfd_mat;
report.finite_difference.mex_gradient = gfd_mex;

fprintf('[3] GRADIENTE vs diferencas finitas (%d/%d componentes; h=%.1e)\n', numel(idx), nU, eps_fd);
fprintf('    MATLAB erro abs %.4e | relativo %.4e\n', ...
    report.finite_difference.matlab_max_abs_error, report.finite_difference.matlab_rel_error);
fprintf('    MEX    erro abs %.4e | relativo %.4e\n', ...
    report.finite_difference.mex_max_abs_error, report.finite_difference.mex_rel_error);

%% 3) Estados previstos (dinamica deve ser igual)
N = nU/2;
if isfield(params,'N'), N_struct = params.N; else, N_struct = N; end
report.dimensions.horizon_match = (N_struct == N);
if isfield(params,'x0')
    x0 = params.x0(:);
elseif isfield(params,'x_k')
    x0 = params.x_k(:);
else
    x0 = [];
end
if ~isempty(x0) && numel(x0)==3 && numel(params_mex)>=3
    x0_mex = params_mex(1:3);
    report.parameters.initial_state_error = norm(x0-x0_mex,inf);
else
    report.parameters.initial_state_error = NaN;
end
report.prediction.X_matlab = X_mat;
report.prediction.X_mex = rollout_unicycle(params_mex(1:3), U, params_mex(9), N);
report.prediction.max_abs_error = norm(X_mat(:)-report.prediction.X_mex(:),inf);
fprintf('[4] DINAMICA / ESTADOS PREDITOS\n');
fprintf('    horizonte consistente: %s\n', yesno(report.dimensions.horizon_match));
fprintf('    erro inicial MATLAB x vetor MEX: %.4e\n', report.parameters.initial_state_error);
fprintf('    erro maximo entre estados previstos: %.4e\n', report.prediction.max_abs_error);

%% 4) Amostragem de campo vetorial nos estados previstos
% A funcao MATLAB de interpolacao e subfuncao de build_local_dcbf; reproduzimos
% aqui a interpolacao bilinear para comparar com a matriz 3D salva para o MEX.
try
    data = load('gvf_lbm_data_multi.mat','x_vec','y_vec','ux_all','uy_all');
    agent_id = round(params_mex(31));
    report.field.agent_id = agent_id;
    report.field.samples = zeros(N,4); % Vx MATLAB, Vy MATLAB, Vx MEX, Vy MEX
    for k=1:N
        [vx1,vy1] = interp_field(field, X_mat(1:2,k));
        [vx2,vy2] = interp_field_3d(data, X_mat(1:2,k), agent_id);
        report.field.samples(k,:) = [vx1,vy1,vx2,vy2];
    end
    report.field.max_abs_error = max(abs(report.field.samples(:,1:2)-report.field.samples(:,3:4)),[],'all');
    fprintf('[5] CAMPO VETORIAL (amostras nos estados MATLAB)\n');
    fprintf('    agente MEX: %d | erro maximo de interpolacao: %.4e\n', ...
        agent_id, report.field.max_abs_error);
catch ME
    report.field.max_abs_error = NaN;
    report.field.error_message = ME.message;
    fprintf('[5] CAMPO VETORIAL: NAO FOI POSSIVEL COMPARAR: %s\n', ME.message);
end

%% 5) Verificacoes basicas do vetor de parametros
report.parameters.vector_length = numel(params_mex);
report.parameters.all_finite = all(isfinite(params_mex));
report.parameters.agent_id = NaN;
if numel(params_mex)>=31, report.parameters.agent_id = params_mex(31); end
report.parameters.N = NaN;
if numel(params_mex)>=8, report.parameters.N = params_mex(8); end
report.parameters.Ts = NaN;
if numel(params_mex)>=9, report.parameters.Ts = params_mex(9); end
report.parameters.neighbor_count = NaN;
if numel(params_mex)>=30, report.parameters.neighbor_count = params_mex(30); end
report.parameters.block_count = NaN;
if numel(params_mex)>=28, report.parameters.block_count = params_mex(28); end
report.parameters.roundabout_count = NaN;
if numel(params_mex)>=29, report.parameters.roundabout_count = params_mex(29); end
fprintf('[6] PARAMETROS EMPACOTADOS\n');
fprintf('    comprimento: %d | todos finitos: %s\n', ...
    report.parameters.vector_length, yesno(report.parameters.all_finite));
fprintf('    N=%g | Ts=%g | agente=%g | vizinhos=%g | blocos=%g | rotatorias=%g\n', ...
    report.parameters.N, report.parameters.Ts, report.parameters.agent_id, ...
    report.parameters.neighbor_count, report.parameters.block_count, ...
    report.parameters.roundabout_count);

%% 6) Opcional: comparar primeiro comando otimizado dos dois solvers
report.solver.compared = false;
if ~isempty(opt.SolverMATLAB) && ~isempty(opt.SolverMEX)
    U0 = opt.InitialGuess;
    if isempty(U0), U0 = U; end
    U0 = U0(:);
    [Uopt_mat, ~, iter_mat] = opt.SolverMATLAB.solve(U0, params);
    [Uopt_mex, ~, iter_mex] = opt.SolverMEX.solve(U0, params_mex);
    Uopt_mat = Uopt_mat(:); Uopt_mex = Uopt_mex(:);
    report.solver.compared = true;
    report.solver.U_matlab = Uopt_mat;
    report.solver.U_mex = Uopt_mex;
    report.solver.iterations_matlab = iter_mat;
    report.solver.iterations_mex = iter_mex;
    report.solver.first_control_matlab = Uopt_mat(1:2);
    report.solver.first_control_mex = Uopt_mex(1:2);
    report.solver.first_control_abs_error = norm(Uopt_mat(1:2)-Uopt_mex(1:2),inf);
    report.solver.solution_abs_error = norm(Uopt_mat-Uopt_mex,inf);
    report.solver.cost_matlab = build_local_dcbf(Uopt_mat,params,field);
    report.solver.cost_mex = wrapper_cost_mex(Uopt_mex,params_mex);
    fprintf('[7] SOLUCAO PANOC (opcional)\n');
    fprintf('    u0 MATLAB = [%.8g %.8g]\n', Uopt_mat(1), Uopt_mat(2));
    fprintf('    u0 MEX    = [%.8g %.8g]\n', Uopt_mex(1), Uopt_mex(2));
    fprintf('    erro u0   = %.4e | erro U completo = %.4e\n', ...
        report.solver.first_control_abs_error, report.solver.solution_abs_error);
    fprintf('    custo final MATLAB=%.8g | MEX=%.8g\n', ...
        report.solver.cost_matlab, report.solver.cost_mex);
end

%% 7) Conclusao automatica
report.summary.cost_pass = report.cost.rel_error < 1e-6;
report.summary.gradient_match_pass = report.gradient.dimension_match && ...
    report.gradient.rel_inf_error < 1e-4;
report.summary.matlab_fd_pass = report.finite_difference.matlab_rel_error < 5e-3;
report.summary.mex_fd_pass = report.finite_difference.mex_rel_error < 5e-3;
report.summary.prediction_pass = report.prediction.max_abs_error < 1e-10;
report.summary.field_pass = isfield(report.field,'max_abs_error') && ...
    isfinite(report.field.max_abs_error) && report.field.max_abs_error < 1e-8;
fprintf('\n---------------- RESUMO AUTOMATICO ----------------\n');
fprintf('Custo MATLAB x MEX       : %s\n', yesno(report.summary.cost_pass));
fprintf('Gradiente MATLAB x MEX   : %s\n', yesno(report.summary.gradient_match_pass));
fprintf('Gradiente MATLAB vs FD   : %s\n', yesno(report.summary.matlab_fd_pass));
fprintf('Gradiente MEX vs FD      : %s\n', yesno(report.summary.mex_fd_pass));
fprintf('Dinamica/estados         : %s\n', yesno(report.summary.prediction_pass));
fprintf('Campo vetorial           : %s\n', yesno(report.summary.field_pass));
if report.solver.compared
    report.summary.first_control_pass = report.solver.first_control_abs_error < 1e-3;
    fprintf('Primeiro comando PANOC   : %s\n', yesno(report.summary.first_control_pass));
end
fprintf('---------------------------------------------------\n\n');

try
    save('diagnostico_comparacao_mex_resultado.mat','report');
    writetable(report_table(report), 'diagnostico_comparacao_mex_resultado.csv');
    fprintf('Resultados salvos em diagnostico_comparacao_mex_resultado.mat/.csv\n');
catch ME
    warning('Nao foi possivel salvar CSV/MAT: %s', ME.message);
end
end

function X = rollout_unicycle(x0,U,Ts,N)
X=zeros(3,N+1); X(:,1)=x0(:); C=reshape(U,2,N);
for k=1:N
    th=X(3,k); v=C(1,k); w=C(2,k);
    X(:,k+1)=[X(1,k)+Ts*v*cos(th); X(2,k)+Ts*v*sin(th); X(3,k)+Ts*w];
end
end

function [vx,vy] = interp_field(field,p)
xv=field.x_vec(:).'; yv=field.y_vec(:).';
dx=xv(2)-xv(1); dy=yv(2)-yv(1);
x=min(max(p(1),xv(1)),xv(end)-1.01*dx);
y=min(max(p(2),yv(1)),yv(end)-1.01*dy);
c=min(max(floor((x-xv(1))/dx)+1,1),numel(xv)-1);
r=min(max(floor((y-yv(1))/dy)+1,1),numel(yv)-1);
tx=(x-xv(c))/dx; ty=(y-yv(r))/dy;
vx=bilin(field.ux(r,c),field.ux(r,c+1),field.ux(r+1,c),field.ux(r+1,c+1),tx,ty);
vy=bilin(field.uy(r,c),field.uy(r,c+1),field.uy(r+1,c),field.uy(r+1,c+1),tx,ty);
end

function [vx,vy] = interp_field_3d(data,p,agent)
assert(agent>=1 && agent<=size(data.ux_all,3),'agent_id fora da faixa do campo 3D.');
xv=data.x_vec(:).'; yv=data.y_vec(:).';
dx=xv(2)-xv(1); dy=yv(2)-yv(1);
x=min(max(p(1),xv(1)),xv(end)-1.01*dx);
y=min(max(p(2),yv(1)),yv(end)-1.01*dy);
c=min(max(floor((x-xv(1))/dx)+1,1),numel(xv)-1);
r=min(max(floor((y-yv(1))/dy)+1,1),numel(yv)-1);
tx=(x-xv(c))/dx; ty=(y-yv(r))/dy;
ux=data.ux_all(:,:,agent); uy=data.uy_all(:,:,agent);
vx=bilin(ux(r,c),ux(r,c+1),ux(r+1,c),ux(r+1,c+1),tx,ty);
vy=bilin(uy(r,c),uy(r,c+1),uy(r+1,c),uy(r+1,c+1),tx,ty);
end

function z=bilin(z00,z10,z01,z11,tx,ty)
z=(1-tx)*(1-ty)*z00+tx*(1-ty)*z10+(1-tx)*ty*z01+tx*ty*z11;
end

function s=yesno(tf)
if tf, s='PASSOU'; else, s='FALHOU'; end
end

function T=report_table(r)
names = {'Custo relativo';'Gradiente relativo MATLAB x MEX'; ...
    'Erro FD MATLAB';'Erro FD MEX';'Erro estados previstos';'Erro campo vetorial'};
vals = [r.cost.rel_error; r.gradient.rel_inf_error; ...
    r.finite_difference.matlab_rel_error; r.finite_difference.mex_rel_error; ...
    r.prediction.max_abs_error; r.field.max_abs_error];
T=table(names,vals,'VariableNames',{'Teste','Erro'});
end
