%% Init
clear all, close all, clc
rng(2023)
n = 4;
N = 200;
X = zeros(n^2,N);
Y = zeros(n^2,N);
f_vals = zeros(N,1);
f_approx_vals = zeros(N,1);
ops = sdpsettings('solver','mosek','verbose',0,'debug',1);
sprsty_arr = zeros(N,1);
n_min = 1; n_max = n;
for ii = 1:N
%    s = n;
   sprsty = n_min + randi(n_max-1);
   if ii == 1, sprsty = 1; end
   sprsty_arr(ii)=sprsty;
   M = randn(sprsty); M = M*M'; M = M./norm(M(:)); 
   M = [M, zeros(sprsty, n-sprsty); zeros(n-sprsty,sprsty), zeros(n-sprsty)];
%    X(:,ii) = M(:);
    
   f_vals(ii) = trace((M+trace(M)*eye(n))*logm((M./trace(M)+eye(n))));
   M = M./f_vals(ii);
   f_vals(ii) = 1;
   X(:,ii) = M(:);
%    f_approx_vals(ii) = -quantum_entr((M + 1e-8*eye(n))./trace(M))*trace(M);
end
% norm(f_vals - f_approx_vals)/norm(f_vals)

%% Set up description spaces
x = sdpvar(n,1);
k = 2; %description space = Sym^2(Sym^k(R^n)) = Sym^{2k}(R^n)
s = 1; % domain of B
v = monolist(x, k); u = monolist(x,s);
num_mons = length(v); num_mons_s = length(u);
deg_list = get_deg_list(v, x); deg_list_s = get_deg_list(u,x);

Pi = zeros(n,n,nchoosek(n,2)+1);
idx_cont = nchoosek(n,2);
kk = 1;
for ii = 1:n-1
    for jj = ii+1:n
        Pi(ii,jj,kk) = 1;
        Pi(jj,ii,kk) = -1;
        kk = kk + 1;
    end
end
Pi(:,:,end) = eye(n); Pi(1,1,end) = -1;

Pi_p = zeros(num_mons, num_mons, size(Pi,3));
Pi_p(:,:,1:idx_cont) = gen_deriv_map(Pi(:,:,1:idx_cont), x, v, deg_list);
Pi_p(:,:,idx_cont+1:end) = gen_algebra_map(Pi(:,:,idx_cont+1:end), x, deg_list);
Pi_s = zeros(num_mons_s, num_mons_s, size(Pi,3));
Pi_s(:,:,1:idx_cont) = gen_deriv_map(Pi(:,:,1:idx_cont), x, u, deg_list_s);
Pi_s(:,:,1+idx_cont:end) = gen_algebra_map(Pi(:,:,1+idx_cont:end), x, deg_list_s);

K1 = []; K2 = []; K3 = []; 
for ii = 1:size(Pi,3)
    if ii <= idx_cont
        G = kron(speye(num_mons), sparse(Pi_p(:,:,ii))) - kron(sparse(Pi_p(:,:,ii))', speye(num_mons));
        H = kron(speye(n), sparse(Pi(:,:,ii))) - kron(sparse(Pi(:,:,ii))', speye(n));
        G_s = kron(speye(num_mons_s), sparse(Pi_s(:,:,ii))) - kron(sparse(Pi_s(:,:,ii))', speye(num_mons_s));

        K3 = [K3; G_s];
    else
        G = kron(sparse(inv(Pi_p(:,:,ii))'), sparse(Pi_p(:,:,ii)));
        H = kron(sparse(inv(Pi(:,:,ii))'),sparse(Pi(:,:,ii)));
        G_s = kron(sparse(inv(Pi_s(:,:,ii))'), sparse(Pi_s(:,:,ii)));

        K3 = [K3; G_s - speye(num_mons_s^2)];
    end
    K1 = [K1; kron(H', speye(num_mons^2)) - kron(speye(n^2), G)];
    K2 = [K2; kron(G_s', speye(num_mons^2)) - kron(speye(num_mons_s^2),G)];
end
Tperm = gen_transpose_perm_mtx(n);
Tperm_big = gen_transpose_perm_mtx(num_mons);
Tperm_big_s = gen_transpose_perm_mtx(num_mons_s);
% K1 = [K1; kron(speye(n^2),Tperm_big) - kron(Tperm',speye(num_mons^2))];
K1 = [K1; kron(speye(n^2),Tperm_big) - speye(n^2*num_mons^2)];
K1 = [K1; kron(Tperm',speye(num_mons^2)) - speye(n^2*num_mons^2)];
% K2 = [K2; kron(speye(num_mons^2),Tperm_big) - kron(Tperm_big',speye(num_mons^2))];
K2 = [K2; kron(speye(num_mons_s^2),Tperm_big) - speye(num_mons^2*num_mons_s^2)];
K2 = [K2; kron(Tperm_big_s',speye(num_mons^2)) - speye(num_mons^2*num_mons_s^2)];
K3 = [K3; Tperm_big_s - speye(num_mons_s^2)];

n_small = 3;
phi = eye(n); phi = phi(:,1:n_small);
x_small = sdpvar(n_small,1);
deg_list_small = get_deg_list(monolist(x_small,k),x_small);
deg_list_small_s = get_deg_list(monolist(x_small,s),x_small);

psi = gen_algebra_map(phi, x_small, deg_list_small);
psi_s = gen_algebra_map(phi, x_small, deg_list_small_s);
phi = kron(sparse(phi),sparse(phi)); psi = kron(sparse(psi),sparse(psi)); psi_s = kron(sparse(psi_s),sparse(psi_s));

% K1 = [K1; kron(phi', speye(length(v)^2) - psi*psi')];
% K2 = [K2; kron(psi_s', speye(length(v)^2) - psi*psi'); kron(speye(size(psi_s,1)) - psi_s*psi_s', psi')];
% K2 = [K1; kron(speye(size(phi,1)) - phi*phi', psi')];

% DD = sparse(func_to_mat(@(X) diag(diag(X)), n, n, n, n));
% K1 = [K1; kron(DD', speye(length(v)^2))];

[~,SpRight] = spspaces(K1,2); A_basis = SpRight{1}(:, SpRight{3});
[~,SpRight] = spspaces(K2,2); B_basis = SpRight{1}(:, SpRight{3});
[~,SpRight] = spspaces(K3,2); C_basis = SpRight{1}(:, SpRight{3});

%% Alternate
T = 500; % number of alternations
num_inits = 500; % number of initializations

alpha_var = sdpvar(size(A_basis,2),1); A_var = reshape(A_basis*alpha_var,length(v)^2,[]);
beta_var = sdpvar(size(B_basis,2),1); B_var = reshape(B_basis*beta_var,length(v)^2,[]);
err = sdpvar(N,1);

t = sdpvar(N,1); 
t1 = sdpvar(1,1);
lambda_var = sdpvar(1,1); 

x_in = sdpvar(n);
f_pred_vals = zeros(size(f_vals));

err_rel_tol = 1e-3;
err_consec_iter_bnd = 5;

B_bnd = 1e4; lambda_min = 1e-3; lambda_init_max = 2;

p = 2; q = 1/(1-1/p);

phi = cell(n_max-n_min+1,1); psi = cell(n_max-n_min+1,1); psi_s = cell(n_max-n_min+1,1);
A_var_cell = cell(n_max-n_min+1,1); B_var_cell = cell(n_max-n_min+1,1);
num_mons = zeros(n_max-n_min+1, 1); num_mons_s = zeros(n_max-n_min+1, 1);
for ii = 1:n_max - n_min + 1
    n_small = n_min + ii - 1;
    phi_curr = eye(n); phi_curr = phi_curr(:,1:n_small);
    x_small = sdpvar(n_small,1);
    deg_list_small = get_deg_list(monolist(x_small,k),x_small);
    deg_list_small_s = get_deg_list(monolist(x_small,s),x_small);
    
    num_mons(ii) = size(deg_list_small,1);
    num_mons_s(ii) = size(deg_list_small_s,1);

    psi_curr = gen_algebra_map(phi_curr, x_small, deg_list_small);
    psi_s_curr = gen_algebra_map(phi_curr, x_small, deg_list_small_s);
    phi{ii} = kron(sparse(phi_curr),sparse(phi_curr)); psi{ii} = kron(sparse(psi_curr),sparse(psi_curr)); psi_s{ii} = kron(sparse(psi_s_curr),sparse(psi_s_curr));

    A_var_cell{ii} = psi{ii}'*A_var*phi{ii};
    B_var_cell{ii} = psi{ii}'*B_var*psi_s{ii};
end

%% Initialize (debugging)

errs_per_init = zeros(num_inits, 1);
alpha_arr = zeros(size(A_basis,2), num_inits);
beta_arr = zeros(size(B_basis,2), num_inits);
lambda_arr = zeros(num_inits,1);

Y_vars = {N,1}; Z_vars = {N,1};
for ii = 1:N
    Y_vars{ii} = sdpvar(num_mons(sprsty_arr(ii)-n_min+1));
    Z_vars{ii} = sdpvar(num_mons_s(sprsty_arr(ii)-n_min+1));
end

A_curr_cell = cell(n_max-n_min+1,1); B_curr_cell = cell(n_max-n_min+1,1);

for init = 1:num_inits

    A = reshape(A_basis*randn(size(A_basis,2),1),length(v)^2,[]);
    B = reshape(B_basis*randn(size(B_basis,2),1),length(v)^2,[]);
    lambda_z = (lambda_init_max - lambda_min)*rand() + lambda_min;

    % % Alternate
    err_curr = zeros(T,1);
    for iter = 1:T
        for ii = 1:n_max - n_min + 1
            A_curr_cell{ii} = psi{ii}'*A*phi{ii};
            B_curr_cell{ii} = psi{ii}'*B*psi_s{ii};
        end
        F = [err >= 0, t >= 0];
        for ii = 1:N
            idx = sprsty_arr(ii)-n_min+1;
            y = Y_vars{ii};
            z = Z_vars{ii};
            A_curr = A_curr_cell{idx};
            B_curr = B_curr_cell{idx};
            X_curr = X(:,ii); X_curr = reshape(X_curr, n,n); X_curr = X_curr(1:sprsty_arr(ii), 1:sprsty_arr(ii));

            F = [F, t(ii) + lambda_z*norm(z(:),p) <= f_vals(ii) + err(ii)];
            F = [F, reshape(A_curr*X_curr(:) + B_curr*z(:),num_mons(idx),[]) + t(ii)*eye(num_mons(idx)) >= 0];
            F = [F, -dot(y(:), A_curr*X_curr(:)) >= f_vals(ii) - err(ii)];
            F = [F, norm(B_curr'*y(:),q) <= lambda_z];
            F = [F, trace(y) <= 1, y >= 0];
        end
        diags = optimize(F, norm(err), ops);
        assert(diags.problem == 0)

        F = [err >= 0, t >= 0, norm(B_var(:)) <= B_bnd, lambda_var >= lambda_min];
        for ii = 1:N
            idx = sprsty_arr(ii)-n_min+1;
            y_curr = value(Y_vars{ii});
            z_curr = value(Z_vars{ii});
            A_curr = A_var_cell{idx};
            B_curr = B_var_cell{idx};
            X_curr = X(:,ii); X_curr = reshape(X_curr, n,n); X_curr = X_curr(1:sprsty_arr(ii), 1:sprsty_arr(ii));

            F = [F, t(ii) + lambda_var*norm(z_curr(:),p) <= f_vals(ii) + err(ii)];
            F = [F, reshape(A_curr*X_curr(:) + B_curr*z_curr(:), num_mons(idx), num_mons(idx)) + t(ii)*eye(num_mons(idx)) >= 0];
            F = [F, -dot(y_curr(:), A_curr*X_curr(:)) >= f_vals(ii) - err(ii)];
            F = [F, norm(B_curr'*y_curr(:), q) <= lambda_var];
        end
        diags = optimize(F, norm(err), ops);
        assert(diags.problem == 0)

        A = value(A_var);
        B = value(B_var);
        lambda_z = value(lambda_var);

%         prob_curr = optimizer([t1>=0, reshape(A*x_in(:) + B*z(:),length(v),[]) + t1*eye(length(v)) >= 0], t1 + lambda_z*norm(z(:), p), ops, x_in(:), [t1;z(:)]);
% 
%         for ii = 1:N
%             sln = prob_curr(X(:,ii));
%             f_pred_vals(ii) = sln(1) + lambda_z*norm(sln(2:end), p);
%             %         optimize([t>=0, reshape(A*X(:,ii) + B*z(:),length(v),[]) + t*eye(length(v)) >= 0, lambda_z*norm(z(:)) <= t], t, ops);
%             %         f_pred_vals(ii) = value(t);
%         end
%         err_actual = norm(f_pred_vals - f_vals)/norm(f_vals)
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
A = reshape(A_basis*alpha_arr(:,init_best), length(v)^2,n^2);
B = reshape(B_basis*beta_arr(:,init_best), length(v)^2,[]);
lambda_z = lambda_arr(init_best);

%% Check generalization
m = 20; M = 1000;
x_b = sdpvar(m, 1); 
v_b = monolist(x_b, k); u_b = monolist(x_b, s);
num_mons_b = length(v_b); num_mons_sb = length(u_b);
deg_list_big = get_deg_list(v_b, x_b);
deg_list_big_s = get_deg_list(u_b, x_b);

% suffices to consider perm. invar.
Pi = zeros(m,m,3);
Pi(:,:,1) = eye(m); Pi(:,[1,2],1) = Pi(:,[2,1],1);
Pi(:,:,2) = eye(m); Pi(:,:,2) = Pi(:, [m,1:m-1],2);
Pi(:,:,3) = eye(m); Pi(1,1,3) = -1;

Pi_p = gen_algebra_map(Pi, x_b, deg_list_big);
Pi_s = gen_algebra_map(Pi, x_b, deg_list_big_s);
K1 = []; K2 = []; 
for ii = 1:size(Pi,3)
   G = kron(sparse(Pi_p(:,:,ii)), sparse(Pi_p(:,:,ii)));
   H = kron(sparse(Pi(:,:,ii)),sparse(Pi(:,:,ii)));
   G_s = kron(sparse(Pi_s(:,:,ii)), sparse(Pi_s(:,:,ii)));
   K1 = [K1; kron(H', speye(num_mons_b^2)) - kron(speye(m^2), G)];
   K2 = [K2; kron(G_s', speye(num_mons_b^2)) - kron(speye(num_mons_sb^2),G)];
end
% Pi = zeros(m,m,nchoosek(m,2)+1);
% idx_cont = nchoosek(m,2);
% kk = 1;
% for ii = 1:m-1
%     for jj = ii+1:m
%         Pi(ii,jj,kk) = 1;
%         Pi(jj,ii,kk) = -1;
%         kk = kk + 1;
%     end
% end
% Pi(:,:,end) = eye(m); Pi(1,1,end) = -1;
% 
% Pi_p = zeros(num_mons_b, num_mons_b, size(Pi,3));
% Pi_p(:,:,1:idx_cont) = gen_deriv_map(Pi(:,:,1:idx_cont), x_b, v_b, deg_list_big);
% Pi_p(:,:,idx_cont+1:end) = gen_algebra_map(Pi(:,:,idx_cont+1:end), x_b, deg_list_big);
% Pi_s = zeros(num_mons_sb, num_mons_sb, size(Pi,3));
% Pi_s(:,:,1:idx_cont) = gen_deriv_map(Pi(:,:,1:idx_cont), x_b, u_b, deg_list_big_s);
% Pi_s(:,:,1+idx_cont:end) = gen_algebra_map(Pi(:,:,1+idx_cont:end), x_b, deg_list_big_s);
% 
% K1 = []; K2 = []; 
% for ii = 1:size(Pi,3)
%     if ii <= idx_cont
%         G = kron(speye(num_mons_b), sparse(Pi_p(:,:,ii))) - kron(sparse(Pi_p(:,:,ii))', speye(num_mons_b));
%         H = kron(speye(m), sparse(Pi(:,:,ii))) - kron(sparse(Pi(:,:,ii))', speye(m));
%         G_s = kron(speye(num_mons_sb), sparse(Pi_s(:,:,ii))) - kron(sparse(Pi_s(:,:,ii))', speye(num_mons_sb));
%     else
%         G = kron(sparse(inv(Pi_p(:,:,ii))'), sparse(Pi_p(:,:,ii)));
%         H = kron(sparse(inv(Pi(:,:,ii))'),sparse(Pi(:,:,ii)));
%         G_s = kron(sparse(inv(Pi_s(:,:,ii))'), sparse(Pi_s(:,:,ii)));
%     end
%     K1 = [K1; kron(H', speye(num_mons_b^2)) - kron(speye(m^2), G)];
%     K2 = [K2; kron(G_s', speye(num_mons_b^2)) - kron(speye(num_mons_sb^2),G)];
% end
Tperm = gen_transpose_perm_mtx(m);
Tperm_big = gen_transpose_perm_mtx(num_mons_b);
Tperm_big_s = gen_transpose_perm_mtx(num_mons_sb);

K1 = [K1; kron(speye(m^2),Tperm_big) - speye(m^2*num_mons_b^2)];
K1 = [K1; kron(Tperm',speye(num_mons_b^2)) - speye(m^2*num_mons_b^2)];

K2 = [K2; kron(speye(num_mons_sb^2),Tperm_big) - speye(num_mons_b^2*num_mons_sb^2)];
K2 = [K2; kron(Tperm_big_s',speye(num_mons_b^2)) - speye(num_mons_b^2*num_mons_sb^2)];

phi_b = eye(m); phi_b = phi_b(:,1:n);
psi_b = gen_algebra_map(phi_b, x, deg_list); psi_sb = gen_algebra_map(phi_b, x, deg_list_s); 
phi_b = kron(sparse(phi_b),sparse(phi_b)); psi_b = kron(sparse(psi_b),sparse(psi_b)); psi_sb = kron(sparse(psi_sb),sparse(psi_sb));

% A_big = [K1; kron(sparse(phi_b)', speye(length(v_b)^2))]\sparse([zeros(size(K1,1),1); vec(psi_b*A)]); 
A_big = lsqr([K1; kron(sparse(phi_b)', psi_b')],sparse([zeros(size(K1,1),1); vec(A)]), 1e-16, 1e4);
A_big = reshape(A_big, num_mons_b^2, m^2);
% B_big = [K2; kron(speye(length(u_b)^2), psi_b'); kron(sparse(psi_sb)', speye(length(v_b)^2))]\sparse([zeros(size(K2,1),1); vec(B*psi_sb'); vec(psi_b*B)]); 
B_big = lsqr([K2; kron(psi_sb', psi_b')],sparse([zeros(size(K2,1),1); vec(B)]), 1e-16, 1e4); 
B_big = reshape(B_big, length(v_b)^2,length(u_b)^2);
% B_big = [K1; kron(speye(m^2), psi_b'); kron(sparse(phi_b)', speye(length(v_b)^2))]\sparse([zeros(size(K1,1),1); vec(B*phi_b'); vec(psi_b*B)]);
% B_big = reshape(B_big, num_mons_b^2, []);

x_b_in = sdpvar(m);
z_b = sdpvar(length(u_b));
ext_prob = optimizer([t1>=0, reshape(A_big*x_b_in(:) + B_big*z_b(:),length(v_b),[]) + t1*eye(length(v_b)) >= 0], t1 + lambda_z*norm(z_b(:), p), ops, x_b_in(:), [t1;z_b(:)]);
X_test = zeros(m^2,M); 
f_test = zeros(M,1);
f_pred_test = zeros(M,1);
for ii = 1:M
%      sprsty = 3;
%      M_test = randn(sprsty); M_test = M_test*M_test'/norm(M_test,'fro')^2; M_test = [M_test, zeros(sprsty, m-sprsty); zeros(m-sprsty,sprsty), zeros(m-sprsty)];
    M_test = randn(m); M_test = M_test*M_test'; M_test = M_test./norm(M_test(:));
%     M_test = reshape(X(:,ii),n,n); M_test = [M_test, zeros(n,m-n);zeros(m-n,n), zeros(m-n)];
    X_test(:,ii) = M_test(:);

    sln = ext_prob(X_test(:,ii));
    f_pred_test(ii) = sln(1) + lambda_z*norm(sln(2:end), p);
   
    f_test(ii) = trace((M_test+trace(M_test)*eye(m))*logm(M_test./trace(M_test)+eye(m)));
end
test_err = norm(f_test - f_pred_test)/norm(f_test)

%% Compute gen. error

err_arr = zeros(m,1);
for n_small = 1:m
    f_test = zeros(M,1);
    f_pred_test = zeros(M,1);

    phi = eye(m); phi = phi(:,1:n_small);
    x_small = sdpvar(n_small,1);
    deg_list_small = get_deg_list(monolist(x_small,k),x_small);
    deg_list_small_s = get_deg_list(monolist(x_small,s),x_small);

    psi = gen_algebra_map(phi, x_small, deg_list_small);
    psi_s = gen_algebra_map(phi, x_small, deg_list_small_s);
    phi = kron(sparse(phi),sparse(phi)); psi = kron(sparse(psi),sparse(psi)); psi_s = kron(sparse(psi_s),sparse(psi_s));

    A_small = psi'*A_big*phi;
    B_small = psi'*B_big*psi_s;
    x_b_small = sdpvar(n_small);
    z_b_small = sdpvar(size(deg_list_small_s,1));
    ext_prob_small = optimizer([t1>=0, reshape(A_small*x_b_small(:) + B_small*z_b_small(:),size(deg_list_small,1),[]) + t1*eye(size(deg_list_small,1)) >= 0],... 
        t1 + lambda_z*norm(z_b_small(:), p), ops, x_b_small(:), [t1;z_b_small(:)]);
    for ii = 1:M
        M_test = randn(n_small); M_test = M_test*M_test'/norm(M_test,'fro')^2; 

        sln = ext_prob_small(M_test(:));
        f_pred_test(ii) = sln(1) + lambda_z*norm(sln(2:end), p);

        f_test(ii) = trace((M_test+trace(M_test)*eye(n_small))*logm(M_test./trace(M_test)+eye(n_small)));
    end
    err_arr(n_small) = mean(abs(f_test - f_pred_test)./f_test);
end
save('quantEnt_err_plot_noExt.mat','err_arr')
%% Plot gen. error w/ and w/o ext.
load('quantEnt_err_plot_noExt.mat','err_arr')
figure, plot(err_arr, 'linewidth', 4) 
hold on
load('quantEnt_err_plot.mat','err_arr')
plot(err_arr, 'linewidth', 4) 
xline(n, '--k', 'linewidth', 3)
xlabel('n'), ylabel('error'), legend({'Free', 'Free + Compatible', 'data'},'box','off', 'location', 'southeast')
set(gca,'fontsize',18)
set(gca, 'yscale', 'log')
exportgraphics(gcf,'quantEnt_learn.eps'); 
