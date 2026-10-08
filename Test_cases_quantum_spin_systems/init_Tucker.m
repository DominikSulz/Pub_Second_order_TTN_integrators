function [X,tau] = init_Tucker(d)

X = cell(1,d+2);
tau = cell(1,d);
r = 2;

for k=1:d
    U = [1;0];
    for l=1:r-1
        U = [U [0;1]];
    end
    X{k} = U;
end

C = zeros(2*ones(1,d));
C(1) = 1;
X{end-1} = 1;
X{end} = tensor(C,[2*ones(1,d) 1]);

end