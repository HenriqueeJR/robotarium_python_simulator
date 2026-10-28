function p = pack_params_1D(x0, neighbor_positions, u_previous, target, N, Ts, d_safe, gamma_cbf, rho_cbf, robot_radius, corredores, rotatorias, agent_id)
%PACK_PARAMS_1D Empacota parâmetros na ordem consumida pelo MEX.
% O formato é fixo em 350x1 para MATLAB Coder. Suporta até 4 agentes.
    p = zeros(350,1);
    p(1:3) = x0(:); p(4:5) = target(:); p(6:7) = u_previous(:);
    p(8:13) = [N; Ts; d_safe; gamma_cbf; rho_cbf; robot_radius];
    distance_to_goal = norm(x0(1:2)-target(:));
    near_goal = exp(-(distance_to_goal/0.25)^2);
    p(14:22) = [1-near_goal; 300; 0.5+25*near_goal; 10+200*near_goal; ...
                2e7; 0.01; 0.01; 0.05; 0.10];
    p(23:27) = [1e7; 1e7; 0.5; 0.5; 0.05];
    nb = numel(corredores); nr = size(rotatorias,1);
    p(28:31) = [nb; nr; size(neighbor_positions,3); agent_id];
    for i=1:nb
        vertices=corredores{i}.v;
        p(32+4*(i-1):35+4*(i-1)) = [min(vertices(:,1))+robot_radius; ...
            max(vertices(:,1))-robot_radius; min(vertices(:,2))+robot_radius; ...
            max(vertices(:,2))-robot_radius];
    end
    for i=1:nr
        p(112+4*(i-1):115+4*(i-1)) = rotatorias(i,1:4).';
    end
    nn=size(neighbor_positions,3);
    assert(nn<=3,'No máximo 3 vizinhos são suportados.');
    for j=1:nn
        % O leitor MEX usa params(132:317). Como os indices abaixo
        % acrescentam +1 e +2, a base deve ser 131 para que a primeira
        % coordenada seja gravada exatamente em p(132). A base 132
        % deixava p(132) vazio e deslocava todas as previsoes dos vizinhos.
        base=131+(j-1)*2*(N+1);
        for t=1:N+1
            p(base+2*(t-1)+1)=neighbor_positions(1,t,j);
            p(base+2*(t-1)+2)=neighbor_positions(2,t,j);
        end
    end
end
