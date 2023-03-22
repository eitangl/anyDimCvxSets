function L_a = gen_algebra_map(L, x, deg_list)
% Generate matrices representing action of given linear maps on monomials
% by acting on each factor in a product.
% Inputs:
%  - L: 3D array of size n_y x n_x x m representing m linear maps R^{n_x} -> R^{n_y}
%  - x: a vector of length n_x of class sdpvar (see YALMIP)
%  - deg_list: matrix of size binom(n_x + k, k) x n_x whose rows
%  give the degree of monomials of degree <= k in each entry of x (output
%  of get_deg_list.m).
% 
% Outputs:
%  - L_a: 3D array of size binom(n_y+k, k) x binom(n_x + k, k) x m
%  s.t. L_a(:,:,ii) is the matrix representing the induced action
%  of L(:,:,ii) from polynomials of degree <= k in n_x variables to such 
%  polynomials in n_y variables .
% 
% Eitan Levin, March '23

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