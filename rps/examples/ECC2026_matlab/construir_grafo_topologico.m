function [G, nodes_polys, nodes_centers] = construir_grafo_topologico(corredores, rotatorias)
    Nc = length(corredores); Nr = size(rotatorias, 1); N_nodes = Nc + Nr;
    
    nodes_centers = zeros(N_nodes, 2);
    nodes_polys = repmat(polyshape(), N_nodes, 1);
    node_names = cell(N_nodes, 1);
    node_type = char(zeros(N_nodes, 1));
    orig_index = zeros(N_nodes, 1);
    dirs = zeros(N_nodes, 2);
    
    for i = 1:Nc
        p = polyshape(corredores{i}.v(:,1), corredores{i}.v(:,2));
        nodes_polys(i) = p; [cx, cy] = centroid(p);
        nodes_centers(i,:) = [cx, cy];
        node_names{i} = sprintf('C%d', i);
        node_type(i) = 'C'; orig_index(i) = i; dirs(i,:) = corredores{i}.dir;
    end
    
    theta = linspace(0, 2*pi, 32);
    for i = 1:Nr
        idx = Nc + i; cx = rotatorias(i, 1); cy = rotatorias(i, 2); R_ext = rotatorias(i, 3);
        p = polyshape(cx + R_ext*cos(theta), cy + R_ext*sin(theta));
        nodes_polys(idx) = p; nodes_centers(idx,:) = [cx, cy];
        node_names{idx} = sprintf('R%d', i);
        node_type(idx) = 'R'; orig_index(idx) = i;
    end
    
    s = []; t = []; weights = [];
    for i = 1:N_nodes
        for j = 1:N_nodes
            if i == j, continue; end
            poly_int = intersect(nodes_polys(i), nodes_polys(j));
            if area(poly_int) > 1e-4 
                [cx_int, cy_int] = centroid(poly_int); p_int = [cx_int, cy_int];
                valid_edge = true;
                if node_type(i) == 'C'
                    if dot(p_int - nodes_centers(i,:), dirs(i,:)) < -1e-3, valid_edge = false; end
                end
                if node_type(j) == 'C'
                    if dot(nodes_centers(j,:) - p_int, dirs(j,:)) < -1e-3, valid_edge = false; end
                end
                if valid_edge
                    s = [s; i]; t = [t; j];
                    weights = [weights; norm(nodes_centers(j,:) - nodes_centers(i,:))];
                end
            end
        end
    end
    
    G = digraph(s, t, weights, node_names);
    G.Nodes.Type = node_type; G.Nodes.OrigIndex = orig_index;
end