%% Init
clear all, close all, clc
rng(2023)

%% Generate data
n = 4;   % dim of description
N = 50;  % number of data points
p = pi;  % learn p-norm for this p

X = cell(N,1);        % matrix whose col's are data points
dim_arr = zeros(N,1); % dimension / degree of each data point
for ii = 1:N
   dim = (rand() > .1) + 1; % random dimension <= 2
   dim_arr(ii)=dim;

   m = randn(dim, 1); % random vector
   
   f_curr = norm(m,p); % evaluate function to be estimated:
   m = m./f_curr; % normalize data 
   
   X{ii} = m;
end
f_vals = ones(N,1);  % function values on (normalized) data

imposeExt = 0; % whether or not to impose extendability conditions

%% Set up description spaces
% We use description spaces U = W = Sym^2( Sym^{\leq k}( R^{2n+1} ) ) 
% with corresponding PSD cones. 
k = 1; 

d_V = 1; d_W = 2*k; d_U = 2*k; % generation degrees

% get generators for B_n:
Pi = zeros(n,n,3);
Pi(:,:,1) = eye(n); Pi(:,[1,2],1) = Pi(:,[2,1],1);
Pi(:,:,2) = eye(n); Pi(:,:,2) = Pi(:, [n,1:n-1],2);
Pi(:,:,3) = eye(n); Pi(1,1,3) = -1;

% get action of generators on R^{2n + 1}, viewed as two vectors and a scalar
Pib = zeros(2*n+1,2*n+1,3);
Pib(:,:,1) = blkdiag(Pi(:,:,1),Pi(:,:,1),1);
Pib(:,:,2) = blkdiag(Pi(:,:,2),Pi(:,:,2),1);
Pib(:,:,3) = eye(2*n+1); Pib(:,[1,n+1],3) = Pib(:,[n+1,1],3);

% get induced action of generators on polynomials in 2n+1 variables
x_ext = sdpvar(2*n+1,1); 
m_U = monolist(x_ext, k);            % monomials of degree <= k in 2n+1 variables
deg_list = get_deg_list(m_U, x_ext); % multi-degrees for each monomial
N_U = size(deg_list,1);              % size of matrices in U and W

Pi_U = gen_algebra_map(Pib, x_ext, deg_list); % get action of generators

%% Get bases for linear maps
% generate matrices whose kernels are spaces of extendable, equivariant linear maps
K_A = []; K_B = [];
for ii = 1:size(Pi,3)
    % action of generators on symmetric matrices index by monomials:
    G = kron(sparse(Pi_U(:,:,ii)), sparse(Pi_U(:,:,ii)));

    % append equations for equivariance:
    K_A = [K_A; kron(sparse(Pi(:,:,ii))', speye(N_U^2)) - kron(speye(n), G)];
    K_B = [K_B; kron(speye(N_U^2),G) - kron(G',speye(N_U^2))];
end

% Since we represent symmetric matrices as full matrices, require linear
% maps to map symmetric matrices to symmetric matrices and act by zero on
% skew-symmetric matrices.
Tperm_U = gen_transpose_perm_mtx(N_U); % permutation matrices sending vec(X) to vec(X')
K_A = [K_A; kron(speye(n),Tperm_U) - speye(n*N_U^2)];
K_B = [K_B; kron(speye(N_U^2),Tperm_U) - speye(N_U^4)];
K_B = [K_B; kron(Tperm_U',speye(N_U^2)) - speye(N_U^4)];

% generate embeddings
phi = cell(n,1); psi_U = cell(n,1); 
N_U = zeros(n, 1); % number of monomials for each dim.
for ii = 1:n
    [phi{ii}, psi_U{ii}] = get_embeddings(n, ii, k);
    N_U(ii) = round(sqrt(size(psi_U{ii},2)));
end

% Add extendability conditions:
if imposeExt
    for ii = 1:d_V
        K_A = [K_A; kron(phi{ii}', speye(N_U(end)^2) - psi_U{ii}*psi_U{ii}')];   % Ensure A extends to a morphism
    end
    for ii = 1:d_U
        K_A = [K_A; kron(speye(n) - phi{ii}*phi{ii}', psi_U{ii}')];              % Ensure A' extends to a morphism
    end

    for ii = 1:d_U
        K_B = [K_B; kron(psi_U{ii}', speye(N_U(end)^2) - psi_U{ii}*psi_U{ii}')]; % Ensure B extends to a morphism
    end

    for ii = 1:d_U
        K_B = [K_B; kron(speye(N_U(end)^2) - psi_U{ii}*psi_U{ii}', psi_U{ii}')]; % Ensure B' extends to a morphism
    end
end

% Find bases for kernels:
[~,SpRight] = spspaces(K_A,2); A_basis = SpRight{1}(:, SpRight{3});
[~,SpRight] = spspaces(K_B,2); B_basis = SpRight{1}(:, SpRight{3});

%% Fit description to data
num_alts = 500; % max number of alternations
num_inits = 1;  % number of initializations

% terminate alternation when relative change in error is below threshold
% for a number of consecutive iterations:
err_rel_tol = 1e-3;      % threshold on relative change in error
err_consec_iter_bnd = 5; % number of consecutive iterations

B_bnd = 1e4;         % norm bound on B, needed for stability during alternation
lambda_min = 1e-3;   % min value for lambda
lambda_init_max = 2; % max value for initial lambda 

ops = sdpsettings('solver','mosek','verbose',0); % make non-verbose

% Run alternating min:
[A,B,lambda] = alternate_min(X, dim_arr, f_vals, N_U, N_U,...
    A_basis, B_basis, phi, psi_U, psi_U,...
    num_inits, num_alts, lambda_min, lambda_init_max, B_bnd,...
    err_rel_tol, err_consec_iter_bnd, ops);

%% Extend description to higher dimension
m = 20; % dimension to which to extend

% Set up higher-dim. description spaces:

% Group action on R^m
Pi = zeros(m,m,3);
Pi(:,:,1) = eye(m); Pi(:,[1,2],1) = Pi(:,[2,1],1);
Pi(:,:,2) = eye(m); Pi(:,:,2) = Pi(:, [m,1:m-1],2);
Pi(:,:,3) = eye(m); Pi(1,1,3) = -1;
% Group action on R^{2m+1}
Pib = zeros(2*m+1,2*m+1,3);
Pib(:,:,1) = blkdiag(Pi(:,:,1),Pi(:,:,1),1);
Pib(:,:,2) = blkdiag(Pi(:,:,2),Pi(:,:,2),1);
Pib(:,:,3) = eye(2*m+1); Pib(:,[1,m+1],3) = Pib(:,[m+1,1],3);
% group action on monomials
x_ext_b = sdpvar(2*m+1,1);
m_U_b = monolist(x_ext_b, k);
deg_list_b = get_deg_list(m_U_b, x_ext_b);
N_U_b = size(deg_list_b,1); % number of monomials

Pi_U = gen_algebra_map(Pib, x_ext_b, deg_list_b);

% form coefficient matrices for linear system used to extend
K_A = []; K_B = [];
for ii = 1:size(Pi,3)
    G = kron(sparse(Pi_U(:,:,ii)), sparse(Pi_U(:,:,ii))); % group actions on symmetric matrices indexed by monomials:
    
    % add equivariance equations
    K_A = [K_A; kron(sparse(Pi(:,:,ii))', speye(N_U_b^2)) - kron(speye(m), G)];
    K_B = [K_B; kron(speye(N_U_b^2),G) - kron(G',speye(N_U_b^2))];
end

% Require linear maps to map symmetric matrices to symmetric matrices and act by zero on skew-symmetric matrices.
Tperm_U = gen_transpose_perm_mtx(N_U_b); 
K_A = [K_A; kron(speye(m),Tperm_U) - speye(m*N_U_b^2)];
K_B = [K_B; kron(speye(N_U_b^2),Tperm_U) - speye(N_U_b^4)];
K_B = [K_B; kron(Tperm_U',speye(N_U_b^2)) - speye(N_U_b^4)];

% Form embeddings
[phi_b, psi_U_b] = get_embeddings(m, n, k);

% Extend A, B by solving linear systems
A_big = lsqr([K_A; kron(sparse(phi_b)', psi_U_b')],sparse([zeros(size(K_A,1),1); vec(A)]), 1e-16, 1e4);
A_big = reshape(A_big, N_U_b^2, []);

B_big = lsqr([K_B; kron(psi_U_b', psi_U_b')],sparse([zeros(size(K_B,1),1); vec(B)]), 1e-16, 1e4); 
B_big = reshape(B_big, N_U_b^2,N_U_b^2);

%% Compute error in each dimension
M = 1000; % number of random unit-norm points in each dimension
err_arr = zeros(m,1); 
for n_small = 1:m
    f_true_test = zeros(M,1); % true function values on test points
    f_pred_test = zeros(M,1); % predicted function values

    % form optimization problem defining our estimated function
    [phi, psi_U] = get_embeddings(m, n_small, k);
    
    N_U = round(sqrt(size(psi_U,2)));

    % restrict extended description to current dimension
    A_small = psi_U'*A_big*phi;
    B_small = psi_U'*B_big*psi_U;
    
    % define variables for primal optimization problem
    x_b_small = sdpvar(n_small,1);
    y_b_small = sdpvar(N_U);
    t = sdpvar(1,1);
    
    % form optim. problem defining the function
    ext_prob_small = optimizer([t>=0, reshape(A_small*x_b_small + B_small*y_b_small(:),N_U,[]) + t*eye(N_U) >= 0],... 
        t + lambda*norm(y_b_small(:)), ops, x_b_small, [t;y_b_small(:)]);
    
    % comptue error for each test point
    for ii = 1:M
        x_test = randn(n_small,1); x_test = x_test / norm(x_test); % random unit-norm test vector

        sln = ext_prob_small(x_test(:));                    % solve problem defining the function
        f_pred_test(ii) = sln(1) + lambda*norm(sln(2:end)); % save function value

        f_true_test(ii) = norm(x_test, p); % true value
    end
    err_arr(n_small) = mean(abs(f_true_test - f_pred_test)./f_true_test); % save mean relative error
end

if imposeExt
    save('lPi_err_plot.mat','err_arr')
else
    save('lPi_err_plot_noExt.mat','err_arr')
end

%% Plot error vs. dim with and without compatibility
load('lPi_err_plot_noExt.mat','err_arr') % load errors without extendability
figure, plot(err_arr, 'linewidth', 4) 

hold on
load('lPi_err_plot.mat','err_arr') % load errors with extendability
plot(err_arr, 'linewidth', 4) 

xline(max(dim_arr), '--k', 'linewidth', 3)
xlabel('n'), ylabel('error'), legend({'Free', 'Free + Compatible'},'box','off', 'location', 'southeast')
set(gca,'fontsize',18)
set(gca, 'yscale', 'log')
% % exportgraphics(gcf,'...'); 

%% Auxiliary function

function [phi, psi] = get_embeddings(n, n_0, k) % get embeddings V_{n_0} to V_n and U_{n_0} to U_n
phi = eye(n); phi = sparse(phi(:,1:n_0)); % embed R^{n_0} to R^n
phi_ext = eye(2*n+1); phi_ext = phi_ext(:, [1:n_0, n+1:n+n_0, 2*n+1]); % embed R^{2n_0+1} to R^{2n+1}

x_small_ext = sdpvar(2*n_0+1, 1);
deg_list_small = get_deg_list(monolist(x_small_ext, k), x_small_ext); % list of degrees for monomials in 2n_0+1 variables of degree <= k

psi = gen_algebra_map(phi_ext, x_small_ext, deg_list_small); % action of embeddings on monomials
psi = sparse(psi); psi = kron(psi,psi); % form embeddings acting on matrices
end
