function Tperm = gen_transpose_perm_mtx(n)
% n^2 x n^2 permutation matrix satisfying Tperm*(X(:)) = (X')(:)
%
% Eitan Levin, March '23

vecperm = reshape(1:n^2,n,n)'; vecperm = vecperm(:);
Tperm = speye(n^2); Tperm = Tperm(:,vecperm);