function report = test_mex_gradient(U, params, relative_tolerance)
%TEST_MEX_GRADIENT Compara gradiente analítico MEX com diferenças finitas.
% Requer wrapper_cost_mex e wrapper_grad_mex já compilados.
    if nargin < 3, relative_tolerance = 5e-3; end
    U = U(:); n = numel(U); g = wrapper_grad_mex(U, params);
    g_fd = zeros(n,1); eps0 = 1e-6;
    for k=1:n
        d=zeros(n,1); d(k)=eps0;
        fp=wrapper_cost_mex(U+d,params);
        fm=wrapper_cost_mex(U-d,params);
        g_fd(k)=(fp-fm)/(2*eps0);
    end
    denom=max([1;abs(g);abs(g_fd)]);
    relerr=norm(g-g_fd,inf)/denom;
    report.max_abs_error=norm(g-g_fd,inf);
    report.relative_error=relerr;
    report.pass=isfinite(relerr) && relerr <= relative_tolerance;
    fprintf('Erro absoluto infinito: %.4e\nErro relativo: %.4e\nTeste: %s\n', ...
        report.max_abs_error, relerr, string_on_off_local(report.pass));
end
function s=string_on_off_local(tf)
    if tf, s='PASSOU'; else, s='FALHOU'; end
end
