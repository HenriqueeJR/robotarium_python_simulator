function grad = wrapper_grad(U, params)
%#codegen
% Wrapper de gradiente com a mesma função objetivo de wrapper_cost.
    [~, grad] = build_local_dcbf_mexable(U, params);
end
