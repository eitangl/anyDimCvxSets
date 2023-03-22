function c = coeffs_mod(p, deg_list, x)
% Wrapper for YALMIP's coefficients, returning coefficients of a 
% polynomial in a particular order
% Inputs:
%  - p: a polynomial of class sdpvar
%  - deg_list: matrix of size (num. of monomials) x (num. of variables)
%  each row of which gives the degree of a monomial in each variable
%  - x: vector of length (num. of variables) of class sdpvar
% 
% Outputs:
%  - c: vector of length (num. of monomials) giving the coefficient of the
%  monomials whose degrees are in deg_list in the polynomial p
%
% Eitan Levin, March '23

[c_p, v_p] = coefficients(p, x);

if isa(c_p, 'sdpvar')
    c = sdpvar(size(deg_list,1),1);
else
    c = zeros(size(deg_list,1),1);
end
for ii = 1:length(v_p)
    c(ismember(deg_list, degree(v_p(ii),x), 'rows')) = c_p(ii);
end

