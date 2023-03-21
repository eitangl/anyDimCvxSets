function c = coeffs_mod(p, deg_list, x)
% Modified version of YALMIP's coeffs, returning coefficients corresponding
% to monomials of given degrees.
% 

[c_p, v_p] = coefficients(p, x);

if isa(c_p, 'sdpvar')
    c = sdpvar(size(deg_list,1),1);
else
    c = zeros(size(deg_list,1),1);
end
for ii = 1:length(v_p)
    c(ismember(deg_list, degree(v_p(ii),x), 'rows')) = c_p(ii);
end

