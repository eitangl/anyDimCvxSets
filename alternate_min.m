function [A,B,lambda] = alternate_min(X, dim_arr, f_vals, N_U, N_W,...
    A_basis, B_basis, phi, psi_W, psi_U,...
    num_inits, num_alts, lambda_min, lambda_init_max, B_bnd,...
    err_rel_tol, err_consec_iter_bnd, ops)
% Perform alternating minimization to fit a description (defined by A, B,
% lambda) to data. 
% Inputs:
%  - X:        cell array of data, represented as vectors of different lengths
%  - dim_arr:  vector containing the degree/dimension index of each data point in X
%  - f_vals:   vector of function values to fit for each data point
%  - N_U, N_W: vectors containing size of symmetric matrices in description space for each data dimension
% 
%  - A_basis, B_basis:  matrices whose columns are vectorized bases for linear maps A & B appearing in descriptions.
%  - phi, psi_W, psi_U: cell arrays containing embedding matrices for each of the three consistent sequences involved.
%  - num_inits:         number of random initializations to try
%  - num_alts:          max number of alternation steps to perform
%  - lambda_min:        lower bound on regularization parameter
%  - lambda_init_max:   max initial value for regularization parameter
%  - B_bnd:             Frobenius norm bound on the B linear map, used for stability
% 
%  - err_rel_tol:         threshold on relative change in error, used for termination 
%  - err_consec_iter_bnd: number of consecutive iterations during which relative change is below threshold needed to terminate
%  - ops:                 YALMIP sdpsettings struct
% 
% Outputs:
%  - A, B:   matrices representing the linear maps in a conic description
%  - lambda: regularization parameter
% 
% Eitan Levin, March '23

% Set up optimization variables:
alpha_var = sdpvar(size(A_basis,2),1);            % coefficients for A
A_var = reshape(A_basis*alpha_var,max(N_U)^2,[]); % A matrix

beta_var = sdpvar(size(B_basis,2),1);             % coefficients for B
B_var = reshape(B_basis*beta_var,max(N_U)^2,[]);  % B matrix

% Form restrictions of A and B to each lower dimension
n = length(N_U);
A_var_cell = cell(n,1); B_var_cell = cell(n,1);
for ii = 1:n
    A_var_cell{ii} = psi_U{ii}'*A_var*phi{ii};
    B_var_cell{ii} = psi_U{ii}'*B_var*psi_W{ii};
end

N = length(X);
err = sdpvar(N,1);        % errors
lambda_var = sdpvar(1,1); % regularization parameter
t = sdpvar(N,1);          % primal scalar vars

Y_vars = cell(N,1); Z_vars = cell(N,1); % primal and dual matrix vars
for ii = 1:N
    Y_vars{ii} = sdpvar(N_W(dim_arr(ii)));
    Z_vars{ii} = sdpvar(N_U(dim_arr(ii)));
end

% initialize arrays
errs_per_init = zeros(num_inits, 1);           % error for each init
alpha_arr = zeros(size(A_basis,2), num_inits); % coefficients in A_basis init.
beta_arr = zeros(size(B_basis,2), num_inits);  % coefficients in B_basis for each init
lambda_arr = zeros(num_inits,1);               % lambda for each init

A_curr_cell = cell(n,1); B_curr_cell = cell(n,1); % cell arrays for restrictions of A and B at each iter. to each dim.

for init = 1:num_inits % for each init
    % randomly initialize A,B,lambda:
    A = reshape(A_basis*randn(size(A_basis,2),1),max(N_U)^2,[]);
    B = reshape(B_basis*randn(size(B_basis,2),1),max(N_U)^2,[]);
    lambda = (lambda_init_max - lambda_min)*rand() + lambda_min; 

    % Alternate
    err_curr = zeros(num_alts,1); % error at each iter
    for iter = 1:num_alts
        for ii = 1:n % restrict A,B to each dim.
            A_curr_cell{ii} = psi_U{ii}'*A*phi{ii};
            B_curr_cell{ii} = psi_U{ii}'*B*psi_W{ii};
        end
        
        % optimize over t, y, z, err:
        F = [err >= 0, t >= 0]; % nonnegativity on error and t-var's for primal
        for ii = 1:N
            y = Y_vars{ii}; % primal matrix var
            z = Z_vars{ii}; % dual matrix var
            A_curr = A_curr_cell{dim_arr(ii)}; % A restricted to (ii)th dim.
            B_curr = B_curr_cell{dim_arr(ii)}; % B restricted to (ii)th dim.
            
            % Impose primal feasibility and cost bound
            F = [F, reshape(A_curr*X{ii} + B_curr*y(:),N_U(dim_arr(ii)),[]) + t(ii)*eye(N_U(dim_arr(ii))) >= 0];
            F = [F, t(ii) + lambda*norm(y(:)) <= f_vals(ii) + err(ii)];
            
            % Impose dual feasibility and cost bound
            F = [F, norm(B_curr'*z(:)) <= lambda, trace(z) <= 1, z >= 0];
            F = [F, -dot(z(:), A_curr*X{ii}) >= f_vals(ii) - err(ii)];
        end
        diags = optimize(F, norm(err), ops); 
        
        if diags.problem == 4    % numerical problems (rare)
            warning('Init. caused numerical problems')
            err_bnd = inf;
            break
        elseif diags.problem ~= 0 % other problems
            error(['Unknown error with YALMIP code ' num2str(diags.problem)])
        end
        
        % optimize over A, B, lambda, err:
        F = [err >= 0, t >= 0, norm(B_var(:)) <= B_bnd, lambda_var >= lambda_min]; % nonnegativity, norm bound on B, lower bound on lambda
        for ii = 1:N
            y_curr = value(Y_vars{ii});
            z_curr = value(Z_vars{ii});

            A_curr = A_var_cell{dim_arr(ii)};
            B_curr = B_var_cell{dim_arr(ii)};

            % Impose primal feasibility and cost bound
            F = [F, reshape(A_curr*X{ii} + B_curr*y_curr(:), N_U(dim_arr(ii)), []) + t(ii)*eye(N_U(dim_arr(ii))) >= 0];
            F = [F, t(ii) + lambda_var*norm(y_curr(:)) <= f_vals(ii) + err(ii)];
            
            % Impose dual feasibility and cost bound
            F = [F, norm(B_curr'*z_curr(:)) <= lambda_var];
            F = [F, -dot(z_curr(:), A_curr*X{ii}) >= f_vals(ii) - err(ii)];
        end
        diags = optimize(F, norm(err), ops);
        
        if diags.problem == 4    % numerical problems (rare)
            warning('Init. caused numerical problems')
            err_bnd = inf;
            break
        elseif diags.problem ~= 0 % other problems
            error(['Unknown error with YALMIP code ' num2str(diags.problem)])
        end

        % update A, B, lambda:
        A = value(A_var); 
        B = value(B_var);
        lambda = value(lambda_var);

        err_bnd = norm(value(err))/norm(f_vals) % current error
        
        % terminate alt. min. if relative change in error has been below threshold for several consecutive iterations:
        err_curr(iter) = err_bnd;
        if iter > err_consec_iter_bnd
            err_rel_diff = 0;             % max relative change in error among the last few iterations
            for jj = 1:err_consec_iter_bnd
                err_rel_diff = max(err_rel_diff, (err_curr(iter-jj)-err_curr(iter-jj+1))/err_curr(iter-jj));
            end

            if err_rel_diff < err_rel_tol % if below threshold, terminate
                break
            end
        end
    end

    % save error, A_basis and B_basis coefficients, regularization parameter:
    errs_per_init(init) = err_bnd; 
    alpha_arr(:,init) = value(alpha_var); 
    beta_arr(:,init) = value(beta_var);   
    lambda_arr(init) = lambda;            
end
% Find and print best error, output corresponding A, B, lambda:
init_best = find(errs_per_init(1:init) == min(errs_per_init(1:init)));
disp(['Lowest training error = ' num2str(errs_per_init(init_best))])

A = reshape(A_basis*alpha_arr(:,init_best), max(N_U)^2,[]);
B = reshape(B_basis*beta_arr(:,init_best), max(N_U)^2,[]);
lambda = lambda_arr(init_best);