function cost = wrapper_cost(U, params)
%#codegen
% Wrapper de custo escalar usado pelo PANOC e pelo MATLAB Coder.
    cost = build_local_dcbf_mexable(U, params);
end
