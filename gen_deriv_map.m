function L_d = gen_deriv_map(L, x, v, deg_list)
% Generate matrices representing action of given linear maps on monomials
% by acting as derivations.
% Inputs:
%  - L: 3D array of size n_y x n_x x m representing m linear maps R^{n_x} -> R^{n_y}
%  - x: a vector of length n_x of class sdpvar (see YALMIP)
%  - v: a vector of length binom(n_x + k, k) of class sdpvar whose entries
%  are monomials in x of degree <= k (output of YALMIP's monolist)
%  - deg_list: matrix of size binom(n_x + k, k) x n_x whose rows
%  give the degree of monomials of degree <= k in each entry of x (output
%  of get_deg_list.m).
% 
% Outputs:
%  - L_a: 3D array of size binom(n_y+k, k) x binom(n_x + k, k) x m
%  s.t. L_a(:,:,ii) is the matrix representing the induced action
%  of L(:,:,ii) from polynomials of degree <= k in n_x variables to such 
%  polynomials in n_y variables, acting as derivations.
% 
% Eitan Levin, March '23

num_mons = size(deg_list,1); % number of monomials
L_d = zeros(num_mons, num_mons, size(L,3));
for ii = 1:size(L,3)
    x_mod = L(:,:,ii)*x; % list of modified monomials
    for jj = 1:num_mons
        v_mod = jacobian(v(jj), x)*x_mod; % action of L(:,:,ii) on (jj)th monomial as a derivation
        L_d(jj, :, ii) = coeffs_mod(v_mod, deg_list, x); % coefficients wrt original monomials
    end
end