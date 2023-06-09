%% Init
clear all, close all, clc
rng(2023)

%% Generate data
n = 4;   % max degree of data
N = 400; % number of points

X = cell(N,1);        % matrix whose col's are data points
dim_arr = zeros(N,1); % dimension / degree of each data point
for ii = 1:N
   dim = randi(n); % random dimension
   dim_arr(ii)=dim;

   M = randn(dim); M = M*M'; M = M./norm(M(:)); % random unit-F-norm PSD matrix
    
   % evaluate function to be estimated:
   f_curr = trace((M+trace(M)*eye(dim))*logm((M./trace(M)+eye(dim)))); 
   M = M./f_curr; % normalize data 
   
   X{ii} = M(:);
end
f_vals = ones(N,1);  % function values on (normalized) data

imposeExt = 0; % whether or not to impose extendability conditions

%% Set up description spaces
% We use description spaces U = Sym^2(Sym^{\leq k_U}(R^n)) and 
% W = Sym^2(Sym^{\leq k_W}(R^n)), with corresponding PSD cones.
k_U = 2; 
k_W = 1; 

d_V = 2; d_U = 2*k_U; d_W = 2*k_W; % generation degrees

% get discrete and continuous generators for O(n):
Pi = zeros(n,n,nchoosek(n,2)+1);
idx_cont = nchoosek(n,2); % dimension of Lie algebra Skew(n)
kk = 1;
for ii = 1:n-1
    for jj = ii+1:n
        Pi(ii,jj,kk) = 1;
        Pi(jj,ii,kk) = -1;
        kk = kk + 1;
    end
end
Pi(:,:,end) = eye(n); Pi(1,1,end) = -1; % discrete generator

% generate matrix representations for action on polynomials
x = sdpvar(n,1);
m_U = monolist(x, k_U); m_W = monolist(x,k_W);                       % monomials in x of degree <= k
N_U = length(m_U); N_W = length(m_W);                                % size of matrices in U and W
deg_list_U = get_deg_list(m_U, x); deg_list_W = get_deg_list(m_W,x); % multi-degrees for each monomial

Pi_U = zeros(N_U, N_U, size(Pi,3));
Pi_U(:,:,1:idx_cont) = gen_deriv_map(Pi(:,:,1:idx_cont), x, m_U, deg_list_U);      % get action of Lie algebra
Pi_U(:,:,idx_cont+1:end) = gen_algebra_map(Pi(:,:,idx_cont+1:end), x, deg_list_U); % get action of discrete generators

Pi_W = zeros(N_W, N_W, size(Pi,3));
Pi_W(:,:,1:idx_cont) = gen_deriv_map(Pi(:,:,1:idx_cont), x, m_W, deg_list_W);
Pi_W(:,:,1+idx_cont:end) = gen_algebra_map(Pi(:,:,1+idx_cont:end), x, deg_list_W);

%% Get bases for linear maps
% generate matrices whose kernels are spaces of extendable, equivariant linear maps
K_A = []; K_B = []; 
for ii = 1:size(Pi,3)
    if ii <= idx_cont % action of Lie algebra on symmetric matrices indexed by monomials
        G_V = kron(speye(n), sparse(Pi(:,:,ii))) - kron(sparse(Pi(:,:,ii))', speye(n));
        G_W = kron(speye(N_W), sparse(Pi_W(:,:,ii))) - kron(sparse(Pi_W(:,:,ii))', speye(N_W));
        G_U = kron(speye(N_U), sparse(Pi_U(:,:,ii))) - kron(sparse(Pi_U(:,:,ii))', speye(N_U));
    else              % action of discrete generators on such matrices
        G_V = kron(sparse(inv(Pi(:,:,ii))'),sparse(Pi(:,:,ii)));
        G_W = kron(sparse(inv(Pi_W(:,:,ii))'), sparse(Pi_W(:,:,ii)));
        G_U = kron(sparse(inv(Pi_U(:,:,ii))'), sparse(Pi_U(:,:,ii)));
    end
    
    % append equations for equivariance
    K_A = [K_A; kron(G_V', speye(N_U^2)) - kron(speye(n^2), G_U)];
    K_B = [K_B; kron(G_W', speye(N_U^2)) - kron(speye(N_W^2),G_U)];
end

% Since we represent symmetric matrices as full matrices, require linear
% maps to map symmetric matrices to symmetric matrices and act by zero on
% skew-symmetric matrices.
Tperm_V = gen_transpose_perm_mtx(n); 
Tperm_W = gen_transpose_perm_mtx(N_W); % permutation matrices sending vec(X) to vec(X')
Tperm_U = gen_transpose_perm_mtx(N_U);

K_A = [K_A; kron(speye(n^2),Tperm_U) - speye(n^2*N_U^2)];
K_A = [K_A; kron(Tperm_V',speye(N_U^2)) - speye(n^2*N_U^2)];

K_B = [K_B; kron(speye(N_W^2),Tperm_U) - speye(N_U^2*N_W^2)];
K_B = [K_B; kron(Tperm_W',speye(N_U^2)) - speye(N_U^2*N_W^2)];

% generate embeddings
phi = cell(n,1); psi_U = cell(n,1); psi_W = cell(n,1);
N_U = zeros(n, 1); N_W = zeros(n, 1); % number of monomials for each dim.
for ii = 1:n
    [phi{ii}, psi_U{ii}] = get_embeddings(n, ii, k_U);
    [~, psi_W{ii}] = get_embeddings(n, ii, k_W);
    
    N_U(ii) = round(sqrt(size(psi_U{ii},2)));
    N_W(ii) = round(sqrt(size(psi_W{ii},2)));
end

% Add extendability conditions:
if imposeExt
    for ii = 1:d_V
        K_A = [K_A; kron(phi{ii}', speye(N_U(end)^2) - psi_U{ii}*psi_U{ii}')];   % Ensure A extends to a morphism
    end

    for ii = 1:d_W
        K_B = [K_B; kron(psi_W{ii}', speye(N_U(end)^2) - psi_U{ii}*psi_U{ii}')]; % Ensure B extends to a morphism
    end

    for ii = 1:d_U
        K_B = [K_B; kron(speye(N_W(end)^2) - psi_W{ii}*psi_W{ii}', psi_U{ii}')]; % Ensure B' extends to a morphism
    end
end

% Find bases for kernels:
[~,SpRight] = spspaces(K_A,2); A_basis = SpRight{1}(:, SpRight{3});
[~,SpRight] = spspaces(K_B,2); B_basis = SpRight{1}(:, SpRight{3});

%% Fit description to data
num_alts = 500;  % max number of alternations
num_inits = 100; % number of initializations

% terminate alternation when relative change in error is below threshold
% for a number of consecutive iterations:
err_rel_tol = 1e-3;      % threshold on relative change in error
err_consec_iter_bnd = 5; % number of consecutive iterations

B_bnd = 1e4;         % norm bound on B, needed for stability during alternation
lambda_min = 1e-3;   % min value for lambda
lambda_init_max = 2; % max value for initial lambda 

ops = sdpsettings('solver','mosek','verbose',0); % make non-verbose

% Run alternating min:
[A,B,lambda] = alternate_min(X, dim_arr, f_vals, N_U, N_W,...
    A_basis, B_basis, phi, psi_W, psi_U,...
    num_inits, num_alts, lambda_min, lambda_init_max, B_bnd,...
    err_rel_tol, err_consec_iter_bnd, ops);

%% Extend description to higher dimension
m = 20; % dimension to which to extend

% Set up higher-dim. description spaces:

% Group action on description space
% (suffices to consider only permutations to guarantee unique extension)
Pi = zeros(m,m,2);
Pi(:,:,1) = eye(m); Pi(:,[1,2],1) = Pi(:,[2,1],1);
Pi(:,:,2) = eye(m); Pi(:,:,2) = Pi(:, [m,1:m-1],2);

% group action on monomials
x_b = sdpvar(m, 1); 
m_U_b = monolist(x_b, k_U); m_W_b = monolist(x_b, k_W);
N_U_b = length(m_U_b); N_W_b = length(m_W_b);
deg_list_big_U = get_deg_list(m_U_b, x_b);
deg_list_big_W = get_deg_list(m_W_b, x_b);

Pi_U = gen_algebra_map(Pi, x_b, deg_list_big_U);
Pi_W = gen_algebra_map(Pi, x_b, deg_list_big_W);

% form coefficient matrices for linear system used to extend
K_A = []; K_B = []; 
for ii = 1:size(Pi,3)
    % group actions on symmetric matrices indexed by monomials:
    G_U = kron(sparse(Pi_U(:,:,ii)), sparse(Pi_U(:,:,ii)));
    G_V = kron(sparse(Pi(:,:,ii)),sparse(Pi(:,:,ii)));
    G_W = kron(sparse(Pi_W(:,:,ii)), sparse(Pi_W(:,:,ii)));
    
    % add equivariance equations
    K_A = [K_A; kron(G_V', speye(N_U_b^2)) - kron(speye(m^2), G_U)];
    K_B = [K_B; kron(G_W', speye(N_U_b^2)) - kron(speye(N_W_b^2),G_U)];
end

% Require linear maps to map symmetric matrices to symmetric matrices and act by zero on skew-symmetric matrices.
Tperm_V = gen_transpose_perm_mtx(m);
Tperm_U = gen_transpose_perm_mtx(N_U_b);
Tperm_W = gen_transpose_perm_mtx(N_W_b);

K_A = [K_A; kron(speye(m^2),Tperm_U) - speye(m^2*N_U_b^2)];
K_A = [K_A; kron(Tperm_V',speye(N_U_b^2)) - speye(m^2*N_U_b^2)];

K_B = [K_B; kron(speye(N_W_b^2),Tperm_U) - speye(N_U_b^2*N_W_b^2)];
K_B = [K_B; kron(Tperm_W',speye(N_U_b^2)) - speye(N_U_b^2*N_W_b^2)];

% Form embeddings
[phi_b, psi_U_b] = get_embeddings(m, n, k_U);
[~, psi_W_b] = get_embeddings(m, n, k_W);

% Extend A, B by solving linear systems
A_big = lsqr([K_A; kron(sparse(phi_b)', psi_U_b')],sparse([zeros(size(K_A,1),1); vec(A)]), 1e-16, 1e4);
A_big = reshape(A_big, N_U_b^2, m^2);

B_big = lsqr([K_B; kron(psi_W_b', psi_U_b')],sparse([zeros(size(K_B,1),1); vec(B)]), 1e-16, 1e4); 
B_big = reshape(B_big, N_U_b^2,N_W_b^2);

%% Compute error in each dimension
M = 1000; % number of random unit-norm points in each dimension
err_arr = zeros(m,1); 
for n_small = 1:m
    f_true_test = zeros(M,1); % true function values on test points
    f_pred_test = zeros(M,1); % predicted function values

    % form optimization problem defining our estimated function
    [phi, psi_U] = get_embeddings(m, n_small, k_U);
    [~, psi_W] = get_embeddings(m, n_small, k_W);
    
    N_U = round(sqrt(size(psi_U,2)));
    N_W = round(sqrt(size(psi_W,2)));

    % restrict extended description to current dimension
    A_small = psi_U'*A_big*phi;
    B_small = psi_U'*B_big*psi_W;
    
    % define variables for primal optimization problem
    x_b_small = sdpvar(n_small);
    y_b_small = sdpvar(N_W);
    t = sdpvar(1,1);
    
    % form optim. problem defining the function
    ext_prob_small = optimizer([t>=0, reshape(A_small*x_b_small(:) + B_small*y_b_small(:),N_U,[]) + t*eye(N_U) >= 0],... 
        t + lambda*norm(y_b_small(:)), ops, x_b_small(:), [t;y_b_small(:)]);
    
    % comptue error for each test point
    for ii = 1:M
        M_test = randn(n_small); M_test = M_test*M_test'/norm(M_test,'fro')^2; % random unit-norm PSD matrix

        sln = ext_prob_small(M_test(:));                    % solve problem defining the function
        f_pred_test(ii) = sln(1) + lambda*norm(sln(2:end)); % save function value

        f_true_test(ii) = trace((M_test+trace(M_test)*eye(n_small))*logm(M_test./trace(M_test)+eye(n_small))); % true value
    end
    err_arr(n_small) = mean(abs(f_true_test - f_pred_test)./f_true_test); % save mean relative error
end
if imposeExt
    save('quantEnt_err_plot.mat','err_arr')
else
    save('quantEnt_err_plot_noExt.mat','err_arr')
end

%% Plot error vs. dim with and without compatibility
load('quantEnt_err_plot_noExt.mat','err_arr') % load errors without extendability
figure, plot(err_arr, 'linewidth', 4) 

hold on
load('quantEnt_err_plot.mat','err_arr') % load errors with extendability
plot(err_arr, 'linewidth', 4) 

xline(n, '--k', 'linewidth', 3)
xlabel('n'), ylabel('error'), legend({'Free', 'Free + Compatible'},'box','off', 'location', 'southeast')
set(gca,'fontsize',18)
set(gca, 'yscale', 'log')
% % exportgraphics(gcf,'...'); 

%% Auxiliary function

function [phi, psi] = get_embeddings(n, n_0, k) % get embeddings V_{n_0} to V_n and U_{n_0} to U_n
phi = eye(n); phi = phi(:,1:n_0); % embed R^{n_0} to R^n

x_small = sdpvar(n_0,1);
deg_list_small = get_deg_list(monolist(x_small,k),x_small); % list of degrees for monomials in n_0 variables of degree <= k

psi = gen_algebra_map(phi, x_small, deg_list_small); % action of embeddings on monomials

phi = sparse(phi); psi = sparse(psi);
phi = kron(phi,phi); psi = kron(psi,psi); % form embeddings acting on matrices
end