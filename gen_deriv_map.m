function L_d = gen_deriv_map(L, x, v, deg_list)

num_mons = size(deg_list,1); % number of monomials
L_d = zeros(num_mons, num_mons, size(L,3));
for ii = 1:size(L,3)
    x_mod = L(:,:,ii)*x; % list of modified monomials
    for jj = 1:num_mons
        v_mod = jacobian(v(jj), x)*x_mod;
        L_d(jj, :, ii) = coeffs_mod(v_mod, deg_list, x); % coefficients wrt original monomials
    end
end