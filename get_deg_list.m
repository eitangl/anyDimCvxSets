function deg_list = get_deg_list(p, x)
deg_list = zeros(length(p),length(x));
for ii = 1:length(p)
    deg_list(ii, :) = degree(p(ii), x)';
end
