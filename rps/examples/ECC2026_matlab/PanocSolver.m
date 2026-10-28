classdef PanocSolver < handle
    properties
        gamma       % Tamanho do passo inicial
        sigma       % Parâmetro do FBE
        max_iter    % Limite de iterações
        tol         % Tolerância (norma do resíduo)
        u_min       % Limite inferior do motor
        u_max       % Limite superior do motor
        lbfgs_size  % Memória do L-BFGS
        cost_func   % Função de custo genérica
        grad_func   % Função de gradiente genérica
    end
    
    methods
        function obj = PanocSolver(gamma, sigma, max_iter, tol, u_min, u_max, lbfgs_size)
            obj.gamma = gamma;
            obj.sigma = sigma;
            obj.max_iter = max_iter;
            obj.tol = tol;
            obj.u_min = u_min;
            obj.u_max = u_max;
            obj.lbfgs_size = lbfgs_size;
        end
        
        function [u_opt, res_norm, iter] = solve(obj, u_init, params)
            u_k = u_init;
            
            % Memória L-BFGS
            S = zeros(length(u_k), obj.lbfgs_size);
            Y = zeros(length(u_k), obj.lbfgs_size);
            lbfgs_idx = 1;
            lbfgs_count = 0;
            
            grad_k = obj.grad_func(u_k, params);
            
            for iter = 1:obj.max_iter
                cost_k = obj.cost_func(u_k, params);
                
                u_bar = obj.project_U(u_k - obj.gamma * grad_k);
                diff_u = u_bar - u_k;
                
                % Backtracking do Gamma (Estimativa de Lipschitz)
                gamma_changed = false;
                while obj.cost_func(u_bar, params) > cost_k + grad_k' * diff_u + (1 / (2 * obj.gamma)) * norm(diff_u)^2
                    obj.gamma = obj.gamma / 2;
                    obj.sigma = obj.sigma / 2;
                    u_bar = obj.project_U(u_k - obj.gamma * grad_k);
                    diff_u = u_bar - u_k;
                    gamma_changed = true;
                    if obj.gamma < 1e-16, break; end
                end
                
                % Se o operador residual mudou, a memória L-BFGS anterior perde a validade
                if gamma_changed
                    lbfgs_count = 0;
                end
                
                r_k = -diff_u / obj.gamma;
                res_norm = norm(r_k, inf);
                
                if res_norm < obj.tol
                    break;
                end
                
                d_k = obj.compute_lbfgs_direction(-r_k, S, Y, lbfgs_count);
                
                fbe_k = cost_k - (obj.gamma / 2) * norm(grad_k)^2 + ...
                        (1 / (2 * obj.gamma)) * norm(u_k - obj.gamma * grad_k - u_bar)^2;
                        
                tau = 1.0;
                while tau > 1e-6
                    u_next = u_k - (1 - tau) * obj.gamma * r_k + tau * d_k;
                    
                    cost_next = obj.cost_func(u_next, params);
                    grad_next = obj.grad_func(u_next, params);
                    u_bar_next = obj.project_U(u_next - obj.gamma * grad_next);
                    
                    fbe_next = cost_next - (obj.gamma / 2) * norm(grad_next)^2 + ...
                               (1 / (2 * obj.gamma)) * norm(u_next - obj.gamma * grad_next - u_bar_next)^2;
                               
                    if fbe_next <= fbe_k - obj.sigma * norm(r_k)^2
                        break;
                    end
                    tau = tau / 2;
                end
                
                if tau <= 1e-6
                    u_next = u_k - obj.gamma * r_k;
                    grad_next = obj.grad_func(u_next, params);
                    u_bar_next = obj.project_U(u_next - obj.gamma * grad_next);
                end
                
                r_next = (u_next - u_bar_next) / obj.gamma;
                
                s_k = u_next - u_k;
                y_k = r_next - r_k; % Era grad_next - grad_k!
                
                % Atualiza a memória se houver curvatura estrita
                if s_k' * y_k > 1e-10
                    S(:, lbfgs_idx) = s_k;
                    Y(:, lbfgs_idx) = y_k;
                    lbfgs_idx = mod(lbfgs_idx, obj.lbfgs_size) + 1;
                    lbfgs_count = min(lbfgs_count + 1, obj.lbfgs_size);
                end
                
                u_k = u_next;
                grad_k = grad_next;
            end
            
            u_opt = u_k;
        end
        
        function d = compute_lbfgs_direction(obj, q, S, Y, lbfgs_count)
            if lbfgs_count == 0
                d = q;
                return;
            end
            
            alpha = zeros(lbfgs_count, 1);
            rho = zeros(lbfgs_count, 1);
            
            for i = lbfgs_count:-1:1
                rho(i) = 1.0 / (Y(:, i)' * S(:, i));
                alpha(i) = rho(i) * S(:, i)' * q;
                q = q - alpha(i) * Y(:, i);
            end
            
            H0 = (S(:, lbfgs_count)' * Y(:, lbfgs_count)) / (Y(:, lbfgs_count)' * Y(:, lbfgs_count));
            d = H0 * q;
            
            for i = 1:lbfgs_count
                beta = rho(i) * Y(:, i)' * d;
                d = d + S(:, i) * (alpha(i) - beta);
            end
        end
        
        function u_prox = project_U(obj, u_in)
            u_prox = max(obj.u_min, min(obj.u_max, u_in));
        end
    end
end