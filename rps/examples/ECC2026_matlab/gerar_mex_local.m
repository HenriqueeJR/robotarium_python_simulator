function info = gerar_mex_local()
%GERAR_MEX_LOCAL Compila e valida custo/gradiente na plataforma atual.
%
% Windows gera arquivos mexw64; Macs Apple silicon geram mexmaca64. Os
% binarios nao sao portaveis entre sistemas operacionais ou arquiteturas.

    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);

    assert(license('test', 'MATLAB_Coder') == 1, ...
        'O MATLAB Coder nao esta licenciado nesta instalacao.');

    compiler = mex.getCompilerConfigurations('C++', 'Selected');
    if isempty(compiler)
        error('gerar_mex_local:NoCompiler', [ ...
            'Nenhum compilador C++ foi reconhecido pelo MATLAB. ' ...
            'No macOS, instale uma versao compativel do Xcode, abra o ' ...
            'Xcode para aceitar a licenca e execute "mex -setup C++" ' ...
            'no MATLAB antes de repetir este comando.']);
    end

    required_file = fullfile(this_dir, 'gvf_lbm_data_multi.mat');
    assert(isfile(required_file), ...
        'Nao encontrei gvf_lbm_data_multi.mat ao lado das fontes.');

    fprintf('Compilador selecionado: %s\n', compiler.Name);
    fprintf('Plataforma MEX esperada: %s\n', mexext);

    N_mpc = 30;
    args_w = coder.typeof(0, [2*N_mpc, 1]);
    args_p = coder.typeof(0, [350, 1]);

    previous_dir = pwd;
    cleanup = onCleanup(@() cd(previous_dir));
    cd(this_dir);
    clear wrapper_cost_mex wrapper_grad_mex;
    codegen wrapper_cost -args {args_w, args_p} -config:mex
    codegen wrapper_grad -args {args_w, args_p} -config:mex
    clear cleanup;
    cd(previous_dir);

    rehash;
    report = test_equivalencia_codegen();
    assert(report.mex_tested && report.mex_pass, ...
        'Os MEX foram gerados, mas nao passaram no teste de equivalencia.');

    info.compiler = compiler.Name;
    info.mex_extension = mexext;
    info.cost_file = fullfile(this_dir, ['wrapper_cost_mex.' mexext]);
    info.gradient_file = fullfile(this_dir, ['wrapper_grad_mex.' mexext]);
    info.validation = report;
    info.passed = true;
    save(fullfile(this_dir, 'resultado_validacao_mex_local.mat'), 'info');

    fprintf('MEX locais gerados e validados com sucesso.\n');
end
