function [psi_t] = ref_sol_spin_systems(X0,t0,t,B,d,n,unitary)

% Operator in matrix format
gen = superkron(d,B,n);

% inital data in vector format
m = length(size(X0{end})) - 1;
psi_0 = double(tenmat(full_tensor(X0),m+1,1:(m))).';

% solution at time t
if unitary
    psi_t = expm((t-t0)*1i*gen)*psi_0;
else
    psi_t = expm((t-t0)*gen)*psi_0;
end


end


function [M] = superkron(L,B,ndim)
% B is a cell array
[m,n] = size(B);

M = sparse(ndim^L,ndim^L);
for jj=1:m
    tmp = 1;
    for ii=n:-1:1
        if isempty(B{jj,ii}) == 1
            tmp = kron(tmp,speye(ndim,ndim));
        else
            tmp = kron(tmp,B{jj,ii});
        end
    end
    M = M + tmp;
end

end