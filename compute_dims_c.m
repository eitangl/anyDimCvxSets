%% Compute dimensions for V = {R^n}, W = U = Sym^2(Sym^{\leq k} R^{2n+1}) with G_n = B_n
clear all, close all, clc
k = 1; 

imposeMorph1 = 1; % whether or not to impose (Morph1) from paper 
if imposeMorph1
    imposeMorph2 = 0; % whether or not to also impose (Morph2)
end

d_V = 1; d_W = 2*k; d_U = 2*k; % generation degrees (= presentation degs)

% Dimension n needs to exceed relevant presentation degrees
if imposeMorph1 
    n = max([d_V, d_W, d_U]);             
else
    n = max([d_V + d_U, d_W + d_U, d_U]); % max of pres. deg's of V ⊗ U, W ⊗ U, and U
end

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

% get induced action of generators on polynomials in 2n + 1 variables
x = sdpvar(2*n+1,1); 
m_U = monolist(x, k);            % monomials of degree <= k in n variables
deg_list = get_deg_list(m_U, x); % multi-degrees for each monomial
N_U = size(deg_list,1);          % size of matrices in U and W

Pi_U = gen_algebra_map(Pib, x, deg_list); % get action of generators

% generate matrices whose kernels are spaces of (extendable) equivariant linear maps and vectors
K_A = []; K_B = []; K_u = [];
for ii = 1:size(Pi,3)
    % action of generators on symmetric matrices index by monomials:
    G_U = kron(sparse(Pi_U(:,:,ii)), sparse(Pi_U(:,:,ii)));

    % append equations for equivariance:
    K_A = [K_A; kron(Pi(:,:,ii)', speye(N_U^2)) - kron(speye(n), G_U)];
    K_B = [K_B; kron(speye(N_U^2),G_U) - kron(G_U',speye(N_U^2))];
    
    K_u = [K_u; G_U - speye(N_U^2)]; % impose invariance on vectors
end

% Since we represent symmetric matrices as full matrices, require linear
% maps to map symmetric matrices to symmetric matrices and act by zero on
% skew-symmetric matrices.
Tperm_U = gen_transpose_perm_mtx(N_U); % permutation matrices sending vec(X) to vec(X')
K_A = [K_A; kron(speye(n),Tperm_U) - speye(n*N_U^2)];
K_B = [K_B; kron(speye(N_U^2),Tperm_U) - speye(N_U^4)];
K_B = [K_B; kron(Tperm_U',speye(N_U^2)) - speye(N_U^4)];

K_u = [K_u; Tperm_U - speye(N_U^2)]; % impose symmetry on invariants

% Add extendability conditions:
if imposeMorph1
    % generate embeddings
    phi = cell(n,1); psi_U = cell(n,1);
    N_U = zeros(n, 1); % number of monomials for each dim.
    for ii = 1:n
        [phi{ii}, psi_U{ii}] = get_embeddings(n, ii, k);
        N_U(ii) = round(sqrt(size(psi_U{ii},2)));
    end

    for ii = 1:d_V
        K_A = [K_A; kron(phi{ii}', speye(N_U(end)^2) - psi_U{ii}*psi_U{ii}')];   % Ensure A extends to a morphism
    end

    if imposeMorph2
        for ii = 1:d_U
            K_A = [K_A; kron(speye(n) - phi{ii}*phi{ii}', psi_U{ii}')];        % Ensure A' extends to a morphism
        end
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
[~,SpRight] = spspaces(K_u,2); u_basis = SpRight{1}(:, SpRight{3});

disp(['Dim of A''s = ' num2str(size(A_basis,2))])
disp(['Dim of B''s = ' num2str(size(B_basis,2))])
disp(['Dim of u''s = ' num2str(size(u_basis,2))])

%% auxiliary function

function [phi, psi] = get_embeddings(n, n_0, k) % get embeddings V_{n_0} to V_n and U_{n_0} to U_n
phi = eye(n); phi = sparse(phi(:,1:n_0)); % embed R^{n_0} to R^n
phi_ext = eye(2*n+1); phi_ext = phi_ext(:, [1:n_0, n+1:n+n_0, 2*n+1]); % embed R^{2n_0+1} to R^{2n+1}

x_small_ext = sdpvar(2*n_0+1, 1);
deg_list_small = get_deg_list(monolist(x_small_ext, k), x_small_ext); % list of degrees for monomials in 2n_0+1 variables of degree <= k

psi = gen_algebra_map(phi_ext, x_small_ext, deg_list_small); % action of embeddings on monomials
psi = sparse(psi); psi = kron(psi,psi); % form embeddings acting on matrices
end
