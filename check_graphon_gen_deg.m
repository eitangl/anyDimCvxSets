%% Check that degree 2 graphons generate degree 3
clear all, close all, clc
vec = @(x) x(:);

n = 2; % initial degree of graphons

% Basis for V_n:
E = zeros(2^n, 2^n, 2);
E(:,:,1) = diag([1, zeros(1, 2^n-1)]); 
E(:,:,2) = zeros(2^n); E(1,2,2)=1; E(2,1,2)=1;

% generate all order 2^{n+1} permutations:
I = eye(2^(n+1));
M = reshape(I(:,flipud(perms(1:2^(n+1))).'),2^(n+1),2^(n+1),factorial(2^(n+1))); 

% Embed basis into V_{n+1}, form its G_{n+1}-orbits
E_orbit = zeros((2^(n+1))^2, size(M,3), size(E,3));
for jj = 1:size(E,3)
    for ii = 1:size(M,3) 
        E_orbit(:,ii, jj) = vec(M(:,:,ii)*kron(E(:,:,jj),ones(2))*M(:,:,ii)');
    end
end
E_orbit = reshape(E_orbit,[], size(M,3)*size(E,3)); 

% check that span of orbits is all of S^{2^{n+1}}.
s = svd(E_orbit);
expected_rk = nchoosek(2^(n+1)+1,2); % dim S^{2^{n+1}}

display(['min sing. value = ' num2str(s(expected_rk))]) % this should be big

figure, semilogy(s, 'linewidth', 4), xline(expected_rk, '--k', 'linewidth', 3)
xlabel('$i$','interpreter','latex'), ylabel('$\sigma_i$','interpreter','latex'), legend({'Sing. val''s', 'expected rank'},'box','off', 'location', 'northeast')
set(gca,'fontsize',18)