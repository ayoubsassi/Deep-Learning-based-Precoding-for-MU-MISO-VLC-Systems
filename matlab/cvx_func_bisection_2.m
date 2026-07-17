function [W, d, status] = cvx_func_bisection_2(t, M, K, sigma, p, H)
    % Ensure all matrices and variables are of type double
    V = double(kron(eye(M), ones(1, K)));  % Convert V to double
    ll = M * K;
    c = double(10^10);  % Ensure constant is double

    % Start CVX optimization
    cvx_precision high
    cvx_begin quiet
        variables w(ll) d(ll)
        minimize(t);
        subject to
            w <= d;
            w >= -d;
            V * d <= p;
            
            for k = 1:K
                h_k = double(H(:, k));  % Convert h_k to double
                I_k = eye(K);
                I_k(k, k) = 0;
                A_k = double([kron(h_k', I_k); zeros(1, ll)]);  % Convert A_k to double
                sigma_k = double([zeros(K, 1); sqrt(sigma(k))]);  % Convert sigma_k to double
                e_k = zeros(K, 1);
                e_k(k) = 1;
                T_k = double(kron(eye(M), e_k'));  % Convert T_k to double
                
                % CVX constraint
                c * (t * norm(A_k * w + sigma_k)) <= (h_k' * T_k * w) * c;
            end
    cvx_end

    % Set default status to "Failed"
    status = 'Failed';
    aa = zeros(K, 1);
    bb = zeros(K, 1);

    % Evaluate the optimization results
    for k = 1:K
        h_k = double(H(:, k));  % Convert h_k to double
        I_k = eye(K);
        I_k(k, k) = 0;
        A_k = double([kron(h_k', I_k); zeros(1, ll)]);  % Convert A_k to double
        sigma_k = double([zeros(K, 1); sqrt(sigma(k))]);  % Convert sigma_k to double
        e_k = zeros(K, 1);
        e_k(k) = 1;
        T_k = double(kron(eye(M), e_k'));  % Convert T_k to double
        
        % Calculate the values of aa and bb
        aa(k) = t * norm(A_k * w + sigma_k);
        bb(k) = h_k' * T_k * w;
    end
    
    % Check if the solution is valid
    if (sum(aa > bb) == 0) && (sum(p < V * d) == 0) && (sum(isnan([aa; bb; w])) == 0)
        status = 'Solved';
    end

    % Reshape w into the desired form if it's valid
    W = reshape(w, K, M)';
end
