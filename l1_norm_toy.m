%% Initialization
clear all, close all, clc
rng(2023)
vec = @(x) x(:);

n = 4; % dim of representation
n_d = 2; % dim of data
N = 50; % # of data points
X = zeros(n_d,N);
r=pi;
for ii = 1:N
%     if ii == 1, s = 1; else s = 1+randi(n-1); end
%     X(:,ii) = [randn(s, 1); zeros(n-s,1)];
%     X(:,ii) = randn(n,1); 
    X(:,ii) = randn(n_d,1);
    X(:,ii) = X(:,ii)/norm(X(:,ii),r); 
end
% f_vals = sum(abs(X),1); % function values to regress
f_vals = ones(N,1);

% ops = sdpsettings('solver','mosek','verbose',0,'debug',1, 'CACHESOLVERS',1);%,'scs.rho_x',1e-12,'scs.eps',1e-12,'scs.max_iters',1e5);
% ops = sdpsettings(ops,'mosek.MSK_DPAR_INTPNT_CO_TOL_DFEAS', delta, 'mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS', delta, 'mosek.MSK_DPAR_INTPNT_CO_TOL_MU_RED', delta, 'mosek.MSK_DPAR_INTPNT_TOL_DFEAS',delta, 'mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP', delta);
ops = sdpsettings('solver','mosek','verbose',0,'debug',1);
% cvx_precision high
%% Get group generators in R^{n+1} and Sym^4(R^{n+1})
Pi = zeros(n,n,3);
Pi(:,:,1) = eye(n); Pi(:,[1,2],1) = Pi(:,[2,1],1);
Pi(:,:,2) = eye(n); Pi(:,:,2) = Pi(:, [n,1:n-1],2);
Pi(:,:,3) = eye(n); Pi(1,1,3) = -1;

Pib = zeros(2*n+1,2*n+1,3);
Pib(:,:,1) = blkdiag(Pi(:,:,1),Pi(:,:,1),1);
Pib(:,:,2) = blkdiag(Pi(:,:,2),Pi(:,:,2),1);
Pib(:,:,3) = eye(2*n+1); Pib(:,[1,n+1],3) = Pib(:,[n+1,1],3);

k = 1;
x_ext = sdpvar(2*n+1,1);
v = monolist(x_ext, k);
deg_list = get_deg_list(v, x_ext);
num_mons = size(deg_list,1); % number of monomials

K1 = []; K2 = []; K3 = [];

disc_gens_p = gen_algebra_map(Pib, x_ext, deg_list);

for ii = 1:size(Pi,3)
    G = kron(sparse(disc_gens_p(:,:,ii)), sparse(disc_gens_p(:,:,ii)));
    K1 = [K1; kron(sparse(Pi(:,:,ii))', speye(num_mons^2)) - kron(speye(n), G)];
    K2 = [K2; kron(speye(num_mons^2),G) - kron(G',speye(num_mons^2))];
    K3 = [K3; G - speye(num_mons^2)];
end

% Enforce symmetry:
vecperm = reshape(1:num_mons^2,num_mons,num_mons)'; vecperm = vecperm(:);
Pi = speye(num_mons^2); Pi = Pi(:,vecperm);
K1 = [K1; kron(speye(n),Pi) - speye(n*num_mons^2)];
K2 = [K2; kron(speye(num_mons^2),Pi) - speye(num_mons^4)];
K2 = [K2; kron(Pi',speye(num_mons^2)) - speye(num_mons^4)];
K3 = [K3; Pi - speye(num_mons^2)];

% Enforce ext.
phi = eye(n); phi = sparse(phi(:,1:n-1));
phi_ext = eye(2*n+1); phi_ext = phi_ext(:, [1:n-1, n+1:n+n-1, 2*n+1]);
x_small_ext = sdpvar(2*(n-1)+1, 1);
deg_list_small = get_deg_list(monolist(x_small_ext, k), x_small_ext);
psi = gen_algebra_map(phi_ext, x_small_ext, deg_list_small);
psi = sparse(psi); psi = kron(psi,psi);

K1 = [K1; kron(phi', speye(size(psi,1)) - psi*psi'); kron(speye(size(phi,1)) - phi*phi', psi')];
K2 = [K2; kron(psi', speye(size(psi,1)) - psi*psi'); kron(speye(size(psi,1)) - psi*psi', psi')];

% Get basis for kernel
[~,SpRight] = spspaces(K1,2); A_basis = SpRight{1}(:, SpRight{3});
[~,SpRight] = spspaces(K2,2); B_basis = SpRight{1}(:, SpRight{3});

%% Optimize

T = 500; % number of alternations
num_inits = 500; % number of initializations

phi = eye(n); phi = sparse(phi(:,1:n_d));
phi_ext = eye(2*n+1); phi_ext = phi_ext(:, [1:n_d, n+1:n+n_d, 2*n+1]);
x_small_ext = sdpvar(2*n_d+1, 1);
deg_list_small = get_deg_list(monolist(x_small_ext, k), x_small_ext);
psi = gen_algebra_map(phi_ext, x_small_ext, deg_list_small);
psi = sparse(psi); psi = kron(psi,psi);

num_mons = size(deg_list_small,1);

alpha_var = sdpvar(size(A_basis,2),1); A_var = psi'*reshape(A_basis*alpha_var,length(v)^2,[])*phi;
beta_var = sdpvar(size(B_basis,2),1); B_var = psi'*reshape(B_basis*beta_var,length(v)^2,[])*psi;
err = sdpvar(N,1);

t = sdpvar(N,1); 
t1 = sdpvar(1,1);
lambda_var = sdpvar(1,1); 

x_in = sdpvar(n_d,1);
f_pred_vals = zeros(size(f_vals));

err_rel_tol = 1e-3;
err_consec_iter_bnd = 5;

B_bnd = 1e4; lambda_min = 1e-3; lambda_init_max = 2;

p = 2; q = p/(p-1);

errs_per_init = zeros(num_inits, 1);
alpha_arr = zeros(size(A_basis,2), num_inits);
beta_arr = zeros(size(B_basis,2), num_inits);
lambda_arr = zeros(num_inits,1);

Y_vars = []; Z_vars = [];
for ii = 1:N
    Y_vars = [Y_vars, vec(sdpvar(num_mons))];
    Z_vars = [Z_vars, vec(sdpvar(num_mons))];
end
Y_arr = zeros(num_mons^2, N);
Z_arr = zeros(num_mons^2, N);

for init = 1:num_inits

    A = psi'*reshape(A_basis*randn(size(A_basis,2),1),length(v)^2,[])*phi;
    B = psi'*reshape(B_basis*randn(size(B_basis,2),1),length(v)^2,[])*psi;
    lambda_z = (lambda_init_max - lambda_min)*rand() + lambda_min;


    % % Alternate
    err_curr = zeros(T,1);
    for iter = 1:T
        F = [err >= 0, t >= 0];
        for ii = 1:N
            y = reshape(Y_vars(:,ii), num_mons, num_mons);
            z = reshape(Z_vars(:,ii), num_mons, num_mons);
            F = [F, t(ii) + lambda_z*norm(z(:),p) <= f_vals(ii) + err(ii)];
            F = [F, reshape(A*X(:,ii) + B*z(:), num_mons,[]) + t(ii)*eye(num_mons) >= 0];
            F = [F, -dot(y(:), A*X(:,ii)) >= f_vals(ii) - err(ii)];
            F = [F, norm(B'*y(:),q) <= lambda_z];
            F = [F, trace(y) <= 1, y >= 0];
        end
        diags = optimize(F, norm(err), ops);
        assert(diags.problem == 0)

        Y_arr = value(Y_vars);
        Z_arr = value(Z_vars);

        F = [err >= 0, t >= 0, norm(B_var(:)) <= B_bnd, lambda_var >= lambda_min];
        for ii = 1:N
            y_curr = reshape(Y_arr(:,ii), num_mons, num_mons);
            z_curr = reshape(Z_arr(:,ii), num_mons, num_mons);
            F = [F, t(ii) + lambda_var*norm(z_curr(:),p) <= f_vals(ii) + err(ii)];
            F = [F, reshape(A_var*X(:,ii) + B_var*z_curr(:), num_mons, num_mons) + t(ii)*eye(num_mons) >= 0];
            F = [F, -dot(y_curr(:), A_var*X(:,ii)) >= f_vals(ii) - err(ii)];
            F = [F, norm(B_var'*y_curr(:), q) <= lambda_var];
        end
        diags = optimize(F, norm(err) + 1e-4*norm(beta_var,1), ops);
        assert(diags.problem == 0)

        A = value(A_var);
        B = value(B_var);
        lambda_z = value(lambda_var);

        prob_curr = optimizer([t1>=0, reshape(A*x_in(:) + B*z(:),num_mons,[]) + t1*eye(num_mons) >= 0],... 
            t1 + lambda_z*norm(z(:), p), ops, x_in(:), [t1;z(:)]);

        for ii = 1:N
            sln = prob_curr(X(:,ii));
            f_pred_vals(ii) = sln(1) + lambda_z*norm(sln(2:end), p);
        end
        err_actual = norm(f_pred_vals - f_vals)/norm(f_vals)
        err_bnd = norm(value(err))/norm(f_vals)

        err_curr(iter) = err_bnd;
        if iter > err_consec_iter_bnd
            err_rel_diff = 0;
            for jj = 1:err_consec_iter_bnd
                err_rel_diff = max(err_rel_diff, (err_curr(iter-jj)-err_curr(iter-jj+1))/err_curr(iter-jj));
            end
            if err_rel_diff < err_rel_tol
                break
            end
        end
    end
    errs_per_init(init) = norm(value(err))/norm(f_vals);
    alpha_arr(:,init) = value(alpha_var);
    beta_arr(:,init) = value(beta_var);
    lambda_arr(init) = lambda_z;
end
init_best = find(errs_per_init(1:init) == min(errs_per_init(1:init)));
disp(['Lowest training error = ' num2str(errs_per_init(init_best))])
A = reshape(A_basis*alpha_arr(:,init_best), num_mons^2,n);
B = reshape(B_basis*beta_arr(:,init_best), num_mons^2,num_mons^2);
lambda_z = lambda_arr(init_best);

%% Generalize
m = 20; M = 1000;

Pi = zeros(m,m,3);
Pi(:,:,1) = eye(m); Pi(:,[1,2],1) = Pi(:,[2,1],1);
Pi(:,:,2) = eye(m); Pi(:,:,2) = Pi(:, [m,1:m-1],2);
Pi(:,:,3) = eye(m); Pi(1,1,3) = -1;

Pib = zeros(2*m+1,2*m+1,3);
Pib(:,:,1) = blkdiag(Pi(:,:,1),Pi(:,:,1),1);
Pib(:,:,2) = blkdiag(Pi(:,:,2),Pi(:,:,2),1);
Pib(:,:,3) = eye(2*m+1); Pib(:,[1,m+1],3) = Pib(:,[m+1,1],3);

x_ext_b = sdpvar(2*m+1,1);
v_b = monolist(x_ext_b, k);
deg_list_b = get_deg_list(v_b, x_ext_b);
num_mons_b = size(deg_list_b,1); % number of monomials

K1 = []; K2 = [];

disc_gens_p = gen_algebra_map(Pib, x_ext_b, deg_list_b);

for ii = 1:size(Pi,3)
    G = kron(sparse(disc_gens_p(:,:,ii)), sparse(disc_gens_p(:,:,ii)));
    K1 = [K1; kron(sparse(Pi(:,:,ii))', speye(num_mons_b^2)) - kron(speye(m), G)];
    K2 = [K2; kron(speye(num_mons_b^2),G) - kron(G',speye(num_mons_b^2))];
end

% Enforce symmetry:
vecperm = reshape(1:num_mons_b^2,num_mons_b,num_mons_b)'; vecperm = vecperm(:);
Pi = speye(num_mons_b^2); Pi = Pi(:,vecperm);
K1 = [K1; kron(speye(m),Pi) - speye(m*num_mons_b^2)];
K2 = [K2; kron(speye(num_mons_b^2),Pi) - speye(num_mons_b^4)];
K2 = [K2; kron(Pi',speye(num_mons_b^2)) - speye(num_mons_b^4)];

% Add freeness
phi = eye(m); phi = sparse(phi(:,1:n));
phi_ext = eye(2*m+1); phi_ext = phi_ext(:, [1:n, m+1:m+n, 2*m+1]);
psi = gen_algebra_map(phi_ext, x_ext, deg_list);
psi = sparse(psi); psi = kron(psi,psi);

% Extend
A_big = lsqr([K1; kron(phi',psi')],sparse([zeros(size(K1,1),1); vec(A)]), 1e-16, 1e4);
B_big = lsqr([K2; kron(psi',psi')],sparse([zeros(size(K2,1),1); vec(B)]), 1e-16, 1e4);
A_big = reshape(A_big, num_mons_b^2, m);
B_big = reshape(B_big, num_mons_b^2, num_mons_b^2);

% Test
x_in_b = sdpvar(m,1); 
z_b = sdpvar(num_mons_b);
t1_b = sdpvar(1,1);
prob_big = optimizer([t1_b>=0, reshape(A_big*x_in_b + B_big*z_b(:),num_mons_b,[]) + t1_b*eye(num_mons_b) >= 0],... 
            t1_b + lambda_z*norm(z_b(:), p), ops, x_in_b, [t1_b;z_b(:)]);

X_test = randn(m,M);
f_test = zeros(M,1);
f_test_true = zeros(M,1);
for ii = 1:M
    sln = prob_big(X_test(:,ii));
    f_test(ii) = sln(1) + lambda_z*norm(sln(2:end), p);
    f_test_true(ii) = norm(X_test(:,ii),r);
end
err_test = norm(f_test - f_test_true)/norm(f_test_true)

%% Compute gen. error

err_arr = zeros(m,1);
for n_small = 1:m
    f_test = zeros(M,1);
    f_test_true = zeros(M,1);

    x_ext_small = sdpvar(2*n_small+1,1);
    deg_list_small = get_deg_list(monolist(x_ext_small,k),x_ext_small);
    phi = eye(m); phi = sparse(phi(:,1:n_small));
    phi_ext = eye(2*m+1); phi_ext = phi_ext(:, [1:n_small, m+1:m+n_small, 2*m+1]);
    psi = gen_algebra_map(phi_ext, x_ext_small, deg_list_small);
    psi = sparse(psi); psi = kron(psi,psi);

    A_small = psi'*A_big*phi;
    B_small = psi'*B_big*psi;
    x_b_small = sdpvar(n_small,1);
    z_b_small = sdpvar(size(deg_list_small,1));
    ext_prob_small = optimizer([t1>=0, reshape(A_small*x_b_small(:) + B_small*z_b_small(:),size(deg_list_small,1),[]) + t1*eye(size(deg_list_small,1)) >= 0],... 
        t1 + lambda_z*norm(z_b_small(:), p), ops, x_b_small(:), [t1;z_b_small(:)]);
    
    X_test = randn(n_small,M);
    for ii = 1:M
        sln = ext_prob_small(X_test(:,ii));
        f_test(ii) = sln(1) + lambda_z*norm(sln(2:end), p);
        f_test_true(ii) = norm(X_test(:,ii),r);
    end
    err_arr(n_small) = mean(abs(f_test - f_test_true)./f_test_true);
end
save('lPi_err_plot.mat','err_arr')
%% Plot gen. error w/ and w/o ext.
load('lPi_err_plot_noExt.mat','err_arr')
figure, plot(err_arr, 'linewidth', 4) 
hold on
load('lPi_err_plot.mat','err_arr')
plot(err_arr, 'linewidth', 4) 
xline(n_d, '--k', 'linewidth', 3)
xlabel('n'), ylabel('error'), legend({'Free', 'Free + Compatible', 'data'},'box','off', 'location', 'southeast')
set(gca,'fontsize',18)
set(gca, 'yscale', 'log')
exportgraphics(gcf,'lPi_learn.eps'); 

%% Plot 3D ball
n_small = 3;
x_ext_small = sdpvar(2*n_small+1,1);
deg_list_small = get_deg_list(monolist(x_ext_small,k),x_ext_small);
phi = eye(m); phi = sparse(phi(:,1:n_small));
phi_ext = eye(2*m+1); phi_ext = phi_ext(:, [1:n_small, m+1:m+n_small, 2*m+1]);
psi = gen_algebra_map(phi_ext, x_ext_small, deg_list_small);
psi = sparse(psi); psi = kron(psi,psi);

A_small = psi'*A_big*phi;
B_small = psi'*B_big*psi;
x_b_small = sdpvar(n_small,1);
z_b_small = sdpvar(size(deg_list_small,1));
ext_prob_small = optimizer([t1>=0, reshape(A_small*x_b_small(:) + B_small*z_b_small(:),size(deg_list_small,1),[]) + t1*eye(size(deg_list_small,1)) >= 0],...
    t1 + lambda_z*norm(z_b_small(:), p), ops, x_b_small(:), [t1;z_b_small(:)]);

[X,Y,Z] = meshgrid(-1.1:.1:1.1);

% lvl_set = 1; V = abs(X) + abs(Y) + abs(Z) - lvl_set; V2d = abs(x2d) + abs(y2d);
% lvl_set = .8; V = sqrt(X.^2 + Y.^2 + Z.^2) - lvl_set;  V2d = sqrt(x2d.^2 + y2d.^2);
lvl_set = 1; 
V = zeros(size(X));
for ii = 1:numel(X)
    sln = ext_prob_small([X(ii);Y(ii);Z(ii)]);
    V(ii) = sln(1) + lambda_z*norm(sln(2:end));
end
U = (abs(X).^r + abs(Y).^r + abs(Z).^r).^(1/r);
p = patch(isosurface(X, Y, Z, V, 1));
isonormals(X,Y,Z,V,p)
p.FaceColor = [0 0.4470 0.7410];
p.EdgeColor = 'none';
p.FaceAlpha = 0.75;
view(3)
% camlight 
% light
lightangle(-20,0)
lighting gouraud


