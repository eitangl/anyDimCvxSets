function L_a = gen_algebra_map(L, x, deg_list)

k = max(sum(deg_list,2)); % max total degree
num_mons_x = size(deg_list,1); % number of monomials
num_mons_y = nchoosek(size(L,1) + k, k);
L_a = zeros(num_mons_y, num_mons_x, size(L,3));
for ii = 1:size(L,3)
    v_mod = monolist(L(:,:,ii)*x, k); % list of modified monomials
    for jj = 1:num_mons_y
        L_a(jj, :, ii) = coeffs_mod(v_mod(jj), deg_list, x); % coefficients wrt original monomials
    end
end