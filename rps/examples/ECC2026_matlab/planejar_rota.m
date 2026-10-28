function [C_opt, R_opt, Con_opt, path_names, G, nodes_centers, path_nodes] = planejar_rota(corredores, rotatorias, pt_start, pt_target)
    [G, nodes_polys, nodes_centers] = construir_grafo_topologico(corredores, rotatorias);
    
    start_node = []; target_node = [];
    for i = 1:numnodes(G)
        xv = nodes_polys(i).Vertices(:,1); yv = nodes_polys(i).Vertices(:,2);
        if inpolygon(pt_start(1), pt_start(2), xv, yv), start_node = i; end
        if inpolygon(pt_target(1), pt_target(2), xv, yv), target_node = i; end
    end
    
    if isempty(start_node) || isempty(target_node), error('Pontos fora das vias!'); end
    
    [path_nodes, dist] = shortestpath(G, start_node, target_node);
    if isempty(path_nodes), error('Nenhum caminho viável encontrado!'); end
    
    path_names = G.Nodes.Name(path_nodes);
    
    orig_C = []; orig_R = [];
    for i = 1:length(path_nodes)
        nid = path_nodes(i);
        if G.Nodes.Type(nid) == 'C', orig_C = [orig_C, G.Nodes.OrigIndex(nid)];
        else, orig_R = [orig_R, G.Nodes.OrigIndex(nid)]; end
    end
    
    C_opt = cell(1, length(orig_C));
    for i = 1:length(orig_C), C_opt{i} = corredores{orig_C(i)}; end
    
    R_opt = zeros(length(orig_R), 5);
    for i = 1:length(orig_R), R_opt(i,:) = rotatorias(orig_R(i), :); end
    
    Con_opt = [];
    for i = 1:(length(path_nodes) - 2)
        n1 = path_nodes(i); n2 = path_nodes(i+1); n3 = path_nodes(i+2);
        if G.Nodes.Type(n1) == 'C' && G.Nodes.Type(n2) == 'R' && G.Nodes.Type(n3) == 'C'
            c1_idx = find(orig_C == G.Nodes.OrigIndex(n1));
            r_idx  = find(orig_R == G.Nodes.OrigIndex(n2));
            c2_idx = find(orig_C == G.Nodes.OrigIndex(n3));
            Con_opt = [Con_opt; c1_idx, c2_idx, r_idx];
        end
    end
    disp(['Rota Planeada: ', strjoin(path_names, ' -> ')]);
end